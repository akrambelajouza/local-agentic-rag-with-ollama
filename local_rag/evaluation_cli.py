"""CLI composition and live adapters for local RAG evaluation."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Protocol, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from local_rag.agent import build_assistant
from local_rag.assistant import UNSUPPORTED_ANSWER, Citation, GroundedAssistant
from local_rag.config import load_settings
from local_rag.evaluation import (
    METRIC_DEFINITIONS,
    EvaluationCase,
    EvaluationMetadata,
    EvaluationObservation,
    EvaluationThresholds,
    calculate_metrics,
    threshold_failures,
)
from local_rag.evaluation_io import load_evaluation_cases, sha256_file, write_reports


class ClaimSupportGrade(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)


class ClaimSupportJudge(Protocol):
    def find_unsupported_claims(
        self, answer: str, citations: tuple[Citation, ...]
    ) -> tuple[str, ...]: ...


class ModelClaimSupportJudge:
    """Adapter that grades answer claims against the cited evidence excerpts."""

    def __init__(self, model: object) -> None:
        self._grade_model = model.with_structured_output(ClaimSupportGrade)  # type: ignore[attr-defined]

    def find_unsupported_claims(
        self, answer: str, citations: tuple[Citation, ...]
    ) -> tuple[str, ...]:
        if answer.strip() == UNSUPPORTED_ANSWER:
            return ()
        if not citations:
            return (answer.strip(),)
        evidence = "\n\n".join(
            f"Source: {citation.title}\nExcerpt: {citation.excerpt}"
            for citation in citations
        )
        grade = self._grade_model.invoke(
            [
                SystemMessage(
                    "Identify each factual claim in the answer that is not supported "
                    "by the supplied evidence. Return claim text only, without reasoning."
                ),
                HumanMessage(f"Answer:\n{answer}\n\nEvidence:\n{evidence}"),
            ]
        )
        if not isinstance(grade, ClaimSupportGrade):
            raise ValueError("Model returned an invalid claim-support grade")
        return tuple(
            dict.fromkeys(
                claim.strip() for claim in grade.unsupported_claims if claim.strip()
            )
        )


def run_local_evaluation(
    assistant: GroundedAssistant,
    cases: Sequence[EvaluationCase],
    claim_judge: ClaimSupportJudge,
) -> tuple[EvaluationObservation, ...]:
    observations: list[EvaluationObservation] = []
    for case in cases:
        answer = assistant.answer(case.question, [])
        observations.append(
            EvaluationObservation(
                case.case_id,
                answer.text,
                answer.citations,
                answer.retrieval_attempts,
                claim_judge.find_unsupported_claims(answer.text, answer.citations),
            )
        )
    return tuple(observations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local RAG quality")
    parser.add_argument(
        "--cases", type=Path, default=Path("evaluation/questions.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/results/latest.json")
    )
    for definition in METRIC_DEFINITIONS:
        parser.add_argument(
            definition.option,
            dest=definition.threshold_field,
            type=_rate,
            default=definition.default_threshold,
        )
    args = parser.parse_args()

    started = perf_counter()
    settings = load_settings()
    cases = load_evaluation_cases(args.cases)
    grading_model = init_chat_model(
        settings.chat_model,
        model_provider=settings.model_provider,
        temperature=0,
        base_url=settings.ollama_base_url,
    )
    observations = run_local_evaluation(
        build_assistant(settings), cases, ModelClaimSupportJudge(grading_model)
    )
    metrics, scores = calculate_metrics(cases, observations)
    thresholds = EvaluationThresholds(
        **{
            definition.threshold_field: getattr(args, definition.threshold_field)
            for definition in METRIC_DEFINITIONS
        }
    )
    failures = threshold_failures(metrics, thresholds)
    metadata = EvaluationMetadata(
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        dataset_sha256=sha256_file(settings.dataset_path),
        evaluation_set_sha256=sha256_file(args.cases),
        duration_seconds=perf_counter() - started,
        run_at=datetime.now(timezone.utc).isoformat(),
    )
    human_path = write_reports(
        args.output,
        metadata,
        metrics,
        scores,
        thresholds=thresholds,
        failures=failures,
    )
    print(human_path.read_text(encoding="utf-8"))
    print(f"Machine report: {args.output}")
    raise SystemExit(1 if failures else 0)


def cli() -> None:
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error


def _rate(value: str) -> float:
    try:
        rate = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number from 0 to 1") from error
    if not 0 <= rate <= 1:
        raise argparse.ArgumentTypeError("must be from 0 to 1")
    return rate


if __name__ == "__main__":
    cli()
