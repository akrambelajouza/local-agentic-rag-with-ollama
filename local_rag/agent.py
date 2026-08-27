"""Runtime assembly for the grounded local RAG assistant."""

from __future__ import annotations

from langchain.chat_models import init_chat_model

from local_rag.assistant import GroundedAssistant
from local_rag.config import Settings
from local_rag.retrieval import ChromaEvidenceRetriever
from local_rag.workflow import AgenticRetrievalWorkflow, ModelEvidenceJudge


def build_assistant(settings: Settings) -> GroundedAssistant:
    model = init_chat_model(
        settings.chat_model,
        model_provider=settings.model_provider,
        temperature=0,
        base_url=settings.ollama_base_url,
    )
    retriever = ChromaEvidenceRetriever(settings)
    workflow = AgenticRetrievalWorkflow(retriever, ModelEvidenceJudge(model))
    return GroundedAssistant(model, workflow)


def build_agent_executor(settings: Settings) -> GroundedAssistant:
    """Backward-compatible name for callers of the original module."""

    return build_assistant(settings)
