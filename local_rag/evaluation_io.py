"""Evaluation-set loading and report presentation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from local_rag.evaluation import (
    METRIC_DEFINITIONS,
    CaseScore,
    EvaluationCase,
    EvaluationMetadata,
    EvaluationMetrics,
    EvaluationThresholds,
    default_thresholds,
    metric_display,
)


def load_evaluation_cases(path: Path) -> tuple[EvaluationCase, ...]:
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            answerable = item["answerable"]
            if not isinstance(answerable, bool):
                raise TypeError("answerable must be a boolean")
            case = EvaluationCase(
                case_id=str(item["id"]).strip(),
                question=str(item["question"]).strip(),
                answerable=answerable,
                expected_sources=tuple(
                    str(value) for value in item["expected_sources"]
                ),
                expected_answer_terms=tuple(
                    str(value) for value in item["expected_answer_terms"]
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid evaluation case on line {line_number}: {error}"
            ) from error
        if not case.case_id or not case.question or case.case_id in seen_ids:
            raise ValueError(
                f"Invalid or duplicate evaluation case on line {line_number}"
            )
        if case.answerable and (
            not case.expected_sources or not case.expected_answer_terms
        ):
            raise ValueError(
                f"Answerable case on line {line_number} lacks expectations"
            )
        seen_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("Evaluation set is empty")
    return tuple(cases)


def write_reports(
    machine_path: Path,
    metadata: EvaluationMetadata,
    metrics: EvaluationMetrics,
    scores: Sequence[CaseScore],
    *,
    thresholds: EvaluationThresholds | None = None,
    failures: Sequence[str],
) -> Path:
    effective_thresholds = thresholds or default_thresholds()
    machine_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": asdict(metadata),
        "metrics": asdict(metrics),
        "thresholds": asdict(effective_thresholds),
        "metric_definition": {
            "retrieval_hit_rate": "Expected source retrieved in any workflow attempt.",
            "unsupported_claims": (
                "Claim-level judgments against cited excerpts; uncited non-declined "
                "answers count as unsupported."
            ),
        },
        "threshold_failures": list(failures),
        "cases": [asdict(score) for score in scores],
    }
    machine_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    human_path = machine_path.with_suffix(".md")
    metric_lines = "\n".join(
        f"- {definition.label}: {metric_display(definition, metrics)}"
        for definition in METRIC_DEFINITIONS
    )
    failure_lines = "\n".join(f"- {failure}" for failure in failures) or "- None"
    human_path.write_text(
        "# RAG Evaluation Summary\n\n"
        f"Status: **{'PASS' if not failures else 'FAIL'}**\n\n"
        f"{metric_lines}\n"
        f"- Cases: {metrics.case_count}\n"
        f"- Duration: {metadata.duration_seconds:.2f}s\n\n"
        "## Threshold failures\n\n"
        f"{failure_lines}\n",
        encoding="utf-8",
    )
    return human_path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
