"""Corpus loading and the embedding-generation entry point."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_rag.config import Settings, load_settings


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """A source document accepted by the local RAG indexer."""

    url: str
    title: str
    raw_text: str


def load_documents(dataset_path: str | Path) -> list[CorpusDocument]:
    """Load non-empty records from the documented JSONL corpus format."""

    documents: list[CorpusDocument] = []
    with Path(dataset_path).open(encoding="utf-8") as dataset:
        for line in dataset:
            if not line.strip():
                continue
            record = json.loads(line)
            documents.append(
                CorpusDocument(
                    url=record["url"],
                    title=record["title"],
                    raw_text=record["raw_text"],
                )
            )
    return documents


def generate_embeddings(settings: Settings) -> None:
    """Rebuild the configured Chroma collection using the current behavior."""

    documents = load_documents(settings.dataset_path)
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )

    if settings.database_location.exists():
        shutil.rmtree(settings.database_location)

    vector_store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.database_location),
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    for document in documents:
        print(document.url)
        chunks = text_splitter.create_documents(
            [document.raw_text],
            metadatas=[{"source": document.url, "title": document.title}],
        )
        vector_store.add_documents(
            documents=chunks,
            ids=[str(uuid4()) for _ in chunks],
        )


def main() -> None:
    """Run embedding generation from environment-backed settings."""

    generate_embeddings(load_settings())


if __name__ == "__main__":
    main()
