"""Evaluation domain types, metric calculation, and threshold policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from local_rag.assistant import UNSUPPORTED_ANSWER, Citation
from local_rag.workflow import RetrievalAttempt


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    question: str
    answerable: bool
    expected_sources: tuple[str, ...]
    expected_answer_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationObservation:
    case_id: str
    answer: str
    citations: tuple[Citation, ...]
    retrieval_attempts: tuple[RetrievalAttempt, ...] = ()
    unsupported_claims: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CaseScore:
    case_id: str
    retrieval_hit: bool | None
    answer_correct: bool
    unsupported_claims: tuple[str, ...]
    citation_hits: int
    citation_total: int
    answer: str
    retrieved_sources: tuple[str, ...]
    cited_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    retrieval_hit_rate: float
    answer_correctness: float
    unsupported_claims: int
    unsupported_claim_rate: float
    citation_accuracy: float
    case_count: int


@dataclass(frozen=True, slots=True)
class EvaluationThresholds:
    min_retrieval_hit_rate: float
    min_answer_correctness: float
    max_unsupported_claim_rate: float
    min_citation_accuracy: float


@dataclass(frozen=True, slots=True)
class EvaluationMetadata:
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    dataset_sha256: str
    evaluation_set_sha256: str
    duration_seconds: float
    run_at: str


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    metric_field: str
    label: str
    threshold_field: str
    direction: Literal["min", "max"]
    default_threshold: float
    count_field: str | None = None

    @property
    def option(self) -> str:
        return "--" + self.threshold_field.replace("_", "-")


METRIC_DEFINITIONS = (
    MetricDefinition(
        "retrieval_hit_rate",
        "Retrieval hit rate",
        "min_retrieval_hit_rate",
        "min",
        0.75,
    ),
    MetricDefinition(
        "answer_correctness",
        "Answer correctness",
        "min_answer_correctness",
        "min",
        0.75,
    ),
    MetricDefinition(
        "unsupported_claim_rate",
        "Unsupported claims",
        "max_unsupported_claim_rate",
        "max",
        0.0,
        count_field="unsupported_claims",
    ),
    MetricDefinition(
        "citation_accuracy",
        "Citation accuracy",
        "min_citation_accuracy",
        "min",
        0.75,
    ),
)


def default_thresholds() -> EvaluationThresholds:
    values = {
        definition.threshold_field: definition.default_threshold
        for definition in METRIC_DEFINITIONS
    }
    return EvaluationThresholds(**values)


def calculate_metrics(
    cases: Sequence[EvaluationCase],
    observations: Sequence[EvaluationObservation],
) -> tuple[EvaluationMetrics, tuple[CaseScore, ...]]:
    """Score completed observations without requiring a live local stack."""

    observations_by_id = {item.case_id: item for item in observations}
    if len(observations_by_id) != len(observations):
        raise ValueError("Evaluation observations contain duplicate case IDs")
    if set(observations_by_id) != {case.case_id for case in cases}:
        raise ValueError("Evaluation observations must match every evaluation case")

    scores: list[CaseScore] = []
    for case in cases:
        observation = observations_by_id[case.case_id]
        cited_sources = tuple(citation.url for citation in observation.citations)
        retrieved_sources = tuple(
            source
            for attempt in observation.retrieval_attempts
            for source in attempt.source_urls
        )
        expected_sources = set(case.expected_sources)
        retrieval_hit = (
            bool(expected_sources.intersection(retrieved_sources))
            if case.answerable
            else None
        )
        declined = _normalise(observation.answer) == _normalise(UNSUPPORTED_ANSWER)
        answer_correct = (
            declined
            if not case.answerable
            else not declined
            and all(
                _normalise(term) in _normalise(observation.answer)
                for term in case.expected_answer_terms
            )
        )
        unsupported_claims = observation.unsupported_claims
        if not declined and not cited_sources and not unsupported_claims:
            unsupported_claims = (observation.answer,)
        citation_hits = sum(source in expected_sources for source in cited_sources)
        scores.append(
            CaseScore(
                case.case_id,
                retrieval_hit,
                answer_correct,
                unsupported_claims,
                citation_hits,
                len(cited_sources),
                observation.answer,
                retrieved_sources,
                cited_sources,
            )
        )

    answerable_scores = [score for score in scores if score.retrieval_hit is not None]
    unsupported_count = sum(len(score.unsupported_claims) for score in scores)
    unsupported_case_count = sum(bool(score.unsupported_claims) for score in scores)
    citation_total = sum(score.citation_total for score in scores)
    metrics = EvaluationMetrics(
        retrieval_hit_rate=_ratio(
            sum(bool(score.retrieval_hit) for score in answerable_scores),
            len(answerable_scores),
        ),
        answer_correctness=_ratio(
            sum(score.answer_correct for score in scores), len(scores)
        ),
        unsupported_claims=unsupported_count,
        unsupported_claim_rate=_ratio(unsupported_case_count, len(scores)),
        citation_accuracy=_ratio(
            sum(score.citation_hits for score in scores), citation_total
        ),
        case_count=len(scores),
    )
    return metrics, tuple(scores)


def threshold_failures(
    metrics: EvaluationMetrics, thresholds: EvaluationThresholds
) -> tuple[str, ...]:
    failures: list[str] = []
    for definition in METRIC_DEFINITIONS:
        value = float(getattr(metrics, definition.metric_field))
        threshold = float(getattr(thresholds, definition.threshold_field))
        passed = (
            value >= threshold if definition.direction == "min" else value <= threshold
        )
        if not passed:
            comparison = "below" if definition.direction == "min" else "exceeds"
            failures.append(
                f"{definition.label.casefold()} {value:.1%} {comparison} {threshold:.1%}"
            )
    return tuple(failures)


def metric_display(definition: MetricDefinition, metrics: EvaluationMetrics) -> str:
    value = float(getattr(metrics, definition.metric_field))
    if definition.count_field:
        count = int(getattr(metrics, definition.count_field))
        return f"{count} ({value:.1%})"
    return f"{value:.1%}"


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
