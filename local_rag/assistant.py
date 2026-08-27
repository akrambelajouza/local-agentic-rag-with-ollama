"""Grounded answer generation with citations controlled by retrieved metadata."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from local_rag.retrieval import Evidence, EvidenceRetriever, RetrievalInput


UNSUPPORTED_ANSWER = "The indexed collection does not contain enough evidence to answer that question."


@dataclass(frozen=True, slots=True)
class Citation:
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    text: str
    citations: tuple[Citation, ...]


class GroundedAssistant:
    def __init__(self, model: Any, retriever: EvidenceRetriever) -> None:
        self._model = model
        self._retriever = retriever

    def answer(
        self, question: str, history: Sequence[BaseMessage]
    ) -> GroundedAnswer:
        evidence = self._retriever.retrieve(RetrievalInput(query=question))
        if not evidence:
            return GroundedAnswer(UNSUPPORTED_ANSWER, ())
        response = self._model.invoke(
            [SystemMessage(content=_grounding_prompt(evidence)), *history, HumanMessage(question)]
        )
        citations = tuple(
            dict.fromkeys(Citation(item.title, item.source_url) for item in evidence)
        )
        return GroundedAnswer(_answer_text(response.content), citations)


def _grounding_prompt(evidence: tuple[Evidence, ...]) -> str:
    context = "\n\n".join(
        f"Evidence {index}\nTitle: {item.title}\nSource: {item.source_url}\nContent: {item.content}"
        for index, item in enumerate(evidence, start=1)
    )
    return (
        "Answer only from the evidence below. Do not use outside knowledge or invent "
        "sources. Return answer prose only, without links or a sources section. If it "
        "is insufficient, say the collection cannot answer.\n\n" + context
    )


_SOURCE_SECTION = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:sources?|citations?|references?)\s*:?\s*$"
)
_MARKDOWN_LINK = re.compile(r"\[[^\]\n]+\]\((?:https?://|www\.)[^)\s]+\)", re.I)
_URL = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.I)


def _answer_text(content: object) -> str:
    """Remove model-authored citation material before Markdown rendering."""

    text = str(content)
    source_heading = _SOURCE_SECTION.search(text)
    if source_heading:
        text = text[: source_heading.start()]
    text = _MARKDOWN_LINK.sub("", text)
    text = _URL.sub("", text)
    return text.strip()
