"""Typed retrieval of structured, source-backed evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field

from local_rag.config import Settings


class RetrievalInput(BaseModel):
    """Validated input contract for evidence retrieval."""

    query: str = Field(min_length=1, description="Question to search for")


@dataclass(frozen=True, slots=True)
class Evidence:
    content: str
    source_url: str
    title: str
    relevance: float


class EvidenceRetriever(Protocol):
    def retrieve(self, request: RetrievalInput) -> tuple[Evidence, ...]: ...


class ChromaEvidenceRetriever:
    def __init__(self, settings: Settings) -> None:
        embeddings = OllamaEmbeddings(
            model=settings.embedding_model, base_url=settings.ollama_base_url
        )
        self._store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.database_location),
        )
        self._limit = settings.retrieval_limit
        self._threshold = settings.relevance_threshold

    def retrieve(self, request: RetrievalInput) -> tuple[Evidence, ...]:
        matches = self._store.similarity_search_with_relevance_scores(
            request.query,
            k=self._limit,
            score_threshold=self._threshold,
        )
        return tuple(
            Evidence(
                content=document.page_content,
                source_url=str(document.metadata.get("source", "")),
                title=str(document.metadata.get("title", "Untitled source")),
                relevance=float(score),
            )
            for document, score in matches
            if document.metadata.get("source")
        )


def create_retrieval_tool(retriever: EvidenceRetriever) -> BaseTool:
    @tool(args_schema=RetrievalInput)
    def retrieve(query: str) -> list[dict[str, object]]:
        """Return structured evidence and source metadata for a question."""

        return [asdict(item) for item in retriever.retrieve(RetrievalInput(query=query))]

    return retrieve
