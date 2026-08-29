"""Generate a deterministic reference report for the evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

from local_rag.assistant import UNSUPPORTED_ANSWER, Citation
from local_rag.evaluation import (
    EvaluationCase,
    EvaluationMetadata,
    EvaluationObservation,
    calculate_metrics,
    default_thresholds,
    threshold_failures,
)
from local_rag.evaluation_io import (
    load_evaluation_cases,
    sha256_file,
    write_reports,
)
from local_rag.workflow import RetrievalAttempt

REFERENCE_ANSWERS = {
    "python-creator": "Python was created by Guido van Rossum.",
    "python-name": "The name Python is a tribute to Monty Python.",
    "python-uses": "Common uses include web development, data science, and automation.",
    "python-list-comprehensions": (
        "A list comprehension is a concise way to create a list from an iterable."
    ),
}


def build_reference_observations(
    cases: tuple[EvaluationCase, ...],
) -> tuple[EvaluationObservation, ...]:
    observations: list[EvaluationObservation] = []
    for case in cases:
        if not case.answerable:
            observations.append(
                EvaluationObservation(case.case_id, UNSUPPORTED_ANSWER, ())
            )
            continue
        answer = REFERENCE_ANSWERS[case.case_id]
        citations = tuple(
            Citation("Included corpus source", source, answer)
            for source in case.expected_sources
        )
        observations.append(
            EvaluationObservation(
                case.case_id,
                answer,
                citations,
                (RetrievalAttempt(case.question, case.expected_sources),),
            )
        )
    return tuple(observations)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "datasets/data.txt"
    cases_path = root / "evaluation/questions.jsonl"
    cases = load_evaluation_cases(cases_path)
    metrics, scores = calculate_metrics(cases, build_reference_observations(cases))
    thresholds = default_thresholds()
    failures = threshold_failures(metrics, thresholds)
    machine_path = root / "evaluation/results/reference.json"
    human_path = write_reports(
        machine_path,
        EvaluationMetadata(
            chat_model="offline-reference-fixture",
            embedding_model="offline-reference-fixture",
            chunk_size=1000,
            chunk_overlap=200,
            dataset_sha256=sha256_file(dataset_path),
            evaluation_set_sha256=sha256_file(cases_path),
            duration_seconds=0.0,
            run_at="deterministic-offline-reference",
        ),
        metrics,
        scores,
        thresholds=thresholds,
        failures=failures,
    )
    print(f"Wrote {machine_path.relative_to(root)}")
    print(f"Wrote {human_path.relative_to(root)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
