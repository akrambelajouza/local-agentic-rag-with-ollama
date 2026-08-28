"""Bounded, observable retrieval workflow with one optional query rewrite."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from local_rag.retrieval import Evidence, EvidenceRetriever, RetrievalInput


LOGGER = logging.getLogger(__name__)
MAX_RETRIEVAL_ATTEMPTS = 2


class SufficiencyDecision(BaseModel):
    """Structured model decision without private reasoning."""

    sufficient: bool
    rewritten_query: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    message: str


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    sufficient: bool
    evidence: tuple[Evidence, ...]
    events: tuple[WorkflowEvent, ...]


class EvidenceJudge(Protocol):
    def evaluate(
        self,
        question: str,
        query: str,
        evidence: tuple[Evidence, ...],
        *,
        can_retry: bool,
    ) -> SufficiencyDecision: ...


class RetrievalWorkflow(Protocol):
    def run(self, question: str) -> RetrievalResult: ...


class DirectRetrievalWorkflow:
    """Adapter for callers that need one retrieval without model evaluation."""

    def __init__(self, retriever: EvidenceRetriever) -> None:
        self._retriever = retriever

    def run(self, question: str) -> RetrievalResult:
        evidence = self._retriever.retrieve(RetrievalInput(query=question))
        return RetrievalResult(bool(evidence), evidence, ())


class ModelEvidenceJudge:
    """Adapter that asks the chat model for a typed sufficiency decision."""

    def __init__(self, model: object) -> None:
        self._decision_model = model.with_structured_output(SufficiencyDecision)  # type: ignore[attr-defined]

    def evaluate(
        self,
        question: str,
        query: str,
        evidence: tuple[Evidence, ...],
        *,
        can_retry: bool,
    ) -> SufficiencyDecision:
        context = "\n\n".join(
            f"Title: {item.title}\nContent: {item.content}" for item in evidence
        ) or "No evidence was retrieved."
        retry_instruction = (
            "If insufficient, provide one concise rewritten search query."
            if can_retry
            else "No retries remain; leave rewritten_query empty."
        )
        decision = self._decision_model.invoke(
            [
                SystemMessage(
                    "Decide whether the evidence can fully answer the question. "
                    "Return only the requested structured fields; do not provide reasoning."
                ),
                HumanMessage(
                    f"Question: {question}\nCurrent search: {query}\n"
                    f"{retry_instruction}\n\nEvidence:\n{context}"
                ),
            ]
        )
        if not isinstance(decision, SufficiencyDecision):
            raise ValueError("Model returned an invalid sufficiency decision")
        return decision


class AgenticRetrievalWorkflow:
    """Retrieve, assess, and retry at most once behind one small interface."""

    def __init__(self, retriever: EvidenceRetriever, judge: EvidenceJudge) -> None:
        self._retriever = retriever
        self._judge = judge

    def run(self, question: str) -> RetrievalResult:
        query = question
        events: list[WorkflowEvent] = []
        evidence: tuple[Evidence, ...] = ()

        for attempt in range(MAX_RETRIEVAL_ATTEMPTS):
            evidence = self._retriever.retrieve(RetrievalInput(query=query))
            can_retry = attempt + 1 < MAX_RETRIEVAL_ATTEMPTS
            try:
                decision = self._judge.evaluate(
                    question, query, evidence, can_retry=can_retry
                )
            except (TypeError, ValueError):
                return self._stop(
                    evidence,
                    events,
                    "Stopped safely after an invalid sufficiency decision.",
                )

            if decision.sufficient:
                event = WorkflowEvent("Evidence is sufficient; generating the answer.")
                events.append(event)
                LOGGER.info(event.message)
                return RetrievalResult(True, evidence, tuple(events))

            if not can_retry:
                return self._stop(
                    evidence,
                    events,
                    "Stopped after reaching the retrieval retry limit.",
                )

            rewritten_query = (decision.rewritten_query or "").strip()
            if not rewritten_query or rewritten_query == query:
                return self._stop(
                    evidence,
                    events,
                    "Stopped because the model could not produce a safe rewrite.",
                )

            event = WorkflowEvent("Retrying retrieval with a rewritten query.")
            events.append(event)
            LOGGER.info(event.message)
            query = rewritten_query

        return self._stop(evidence, events, "Stopped at the hard retrieval limit.")

    @staticmethod
    def _stop(
        evidence: tuple[Evidence, ...],
        events: list[WorkflowEvent],
        message: str,
    ) -> RetrievalResult:
        event = WorkflowEvent(message)
        events.append(event)
        LOGGER.info(event.message)
        return RetrievalResult(False, evidence, tuple(events))
