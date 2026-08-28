"""Grounded answer generation with citations controlled by retrieved metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from local_rag.retrieval import Evidence
from local_rag.workflow import RetrievalAttempt, RetrievalWorkflow, WorkflowEvent


UNSUPPORTED_ANSWER = "The indexed collection does not contain enough evidence to answer that question."


@dataclass(frozen=True, slots=True)
class Citation:
    title: str
    url: str
    excerpt: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "excerpt": self.excerpt}

    @classmethod
    def from_dict(cls, value: object) -> Citation | None:
        if not isinstance(value, Mapping) or "title" not in value or "url" not in value:
            return None
        return cls(
            title=str(value["title"]),
            url=str(value["url"]),
            excerpt=str(value.get("excerpt", "")),
        )


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    text: str
    citations: tuple[Citation, ...]
    events: tuple[WorkflowEvent, ...] = ()
    retrieval_attempts: tuple[RetrievalAttempt, ...] = ()


class GroundedAssistant:
    def __init__(
        self,
        model: Any,
        workflow: RetrievalWorkflow,
    ) -> None:
        self._model = model
        self._workflow = workflow

    def answer(
        self, question: str, history: Sequence[BaseMessage]
    ) -> GroundedAnswer:
        result = self._workflow.run(question)
        evidence = result.evidence
        events: tuple[WorkflowEvent, ...] = result.events
        if not result.sufficient:
            return GroundedAnswer(
                UNSUPPORTED_ANSWER, (), events, result.attempts
            )
        if not evidence:
            return GroundedAnswer(
                UNSUPPORTED_ANSWER, (), events, result.attempts
            )
        response = self._model.invoke(
            [SystemMessage(content=_grounding_prompt(evidence)), *history, HumanMessage(question)]
        )
        citations_by_source: dict[tuple[str, str], Citation] = {}
        for item in evidence:
            key = (item.title, item.source_url)
            citations_by_source.setdefault(
                key, Citation(item.title, item.source_url, _excerpt(item.content))
            )
        citations = tuple(citations_by_source.values())
        return GroundedAnswer(
            _answer_text(response.content), citations, events, result.attempts
        )


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


def _excerpt(content: str, limit: int = 280) -> str:
    compact = " ".join(content.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
