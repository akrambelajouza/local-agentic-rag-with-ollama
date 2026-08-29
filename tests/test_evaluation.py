import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from local_rag.assistant import UNSUPPORTED_ANSWER, Citation, GroundedAnswer
from local_rag.evaluation import (
    EvaluationCase,
    EvaluationMetadata,
    EvaluationObservation,
    EvaluationThresholds,
    calculate_metrics,
    threshold_failures,
)
from local_rag.evaluation_cli import (
    ClaimSupportGrade,
    ModelClaimSupportJudge,
    _configured_digest,
    run_local_evaluation,
)
from local_rag.evaluation_io import load_evaluation_cases, write_reports
from local_rag.workflow import RetrievalAttempt


class EvaluationMetricTests(unittest.TestCase):
    def test_model_digest_matches_implicit_latest_tag(self) -> None:
        self.assertEqual(
            _configured_digest({"embed:latest": "sha256-value"}, "embed"),
            "sha256-value",
        )

    def test_claim_judge_detects_mixed_supported_and_unsupported_answer(self) -> None:
        model = MagicMock()
        model.with_structured_output.return_value.invoke.return_value = (
            ClaimSupportGrade(unsupported_claims=["The Moon is made of cheese."])
        )
        judge = ModelClaimSupportJudge(model)

        claims = judge.find_unsupported_claims(
            "Alpha is supported. The Moon is made of cheese.",
            (Citation("One", "source-one", "Alpha is supported."),),
        )

        self.assertEqual(claims, ("The Moon is made of cheese.",))

    def test_claim_judge_discards_claims_that_are_absent_from_the_answer(self) -> None:
        model = MagicMock()
        model.with_structured_output.return_value.invoke.return_value = (
            ClaimSupportGrade(
                unsupported_claims=[
                    "Python is named after a snake.",
                    "The Moon is made of cheese.",
                ]
            )
        )
        judge = ModelClaimSupportJudge(model)

        claims = judge.find_unsupported_claims(
            "Python is used for automation. The Moon is made of cheese.",
            (
                Citation(
                    "Uses",
                    "source-one",
                    "Python is popular for automation and scripting tasks.",
                ),
            ),
        )

        self.assertEqual(claims, ("The Moon is made of cheese.",))

    def test_claim_judge_retains_number_contradictions_despite_lexical_overlap(
        self,
    ) -> None:
        model = MagicMock()
        model.with_structured_output.return_value.invoke.return_value = (
            ClaimSupportGrade(unsupported_claims=["Python was created in 2001."])
        )
        judge = ModelClaimSupportJudge(model)

        claims = judge.find_unsupported_claims(
            "Python was created in 2001.",
            (Citation("History", "source-one", "Python was created in 1991."),),
        )

        self.assertEqual(claims, ("Python was created in 2001.",))

    def test_live_runner_captures_retrieval_attempts_and_claim_grades(self) -> None:
        assistant = MagicMock()
        attempt = RetrievalAttempt("Question one", ("source-one",))
        assistant.answer.return_value = GroundedAnswer(
            "Alpha", (Citation("One", "source-one", "Alpha"),), (), (attempt,)
        )
        claim_judge = MagicMock()
        claim_judge.find_unsupported_claims.return_value = ("Unsupported detail",)

        observations = run_local_evaluation(assistant, (self.cases[0],), claim_judge)

        self.assertEqual(observations[0].retrieval_attempts, (attempt,))
        self.assertEqual(observations[0].unsupported_claims, ("Unsupported detail",))

    def test_included_set_covers_answerable_and_unanswerable_questions(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]

        cases = load_evaluation_cases(repository_root / "evaluation/questions.jsonl")

        self.assertTrue(any(case.answerable for case in cases))
        self.assertTrue(any(not case.answerable for case in cases))
        self.assertTrue(all(case.expected_sources for case in cases if case.answerable))

    def setUp(self) -> None:
        self.cases = (
            EvaluationCase(
                "answerable-one", "Question one", True, ("source-one",), ("alpha",)
            ),
            EvaluationCase(
                "answerable-two", "Question two", True, ("source-two",), ("beta",)
            ),
            EvaluationCase("unanswerable", "Unknown", False, (), ()),
        )

    def test_calculates_retrieval_answer_and_citation_metrics(self) -> None:
        observations = (
            EvaluationObservation(
                "answerable-one",
                "Alpha is supported.",
                (Citation("One", "source-one", "Alpha is supported."),),
                (RetrievalAttempt("Question one", ("source-one",)),),
            ),
            EvaluationObservation(
                "answerable-two",
                "An incorrect response.",
                (Citation("Wrong", "wrong-source", "Other evidence."),),
                (RetrievalAttempt("Question two", ("wrong-source",)),),
            ),
            EvaluationObservation("unanswerable", UNSUPPORTED_ANSWER, ()),
        )

        metrics, scores = calculate_metrics(self.cases, observations)

        self.assertEqual(metrics.retrieval_hit_rate, 0.5)
        self.assertAlmostEqual(metrics.answer_correctness, 2 / 3)
        self.assertEqual(metrics.unsupported_claims, 0)
        self.assertEqual(metrics.unsupported_claim_rate, 0.0)
        self.assertEqual(metrics.annotated_source_coverage, 0.5)
        self.assertEqual(len(scores), 3)

    def test_retrieval_hit_is_independent_of_answer_and_citations(self) -> None:
        case = self.cases[0]
        observation = EvaluationObservation(
            case.case_id,
            UNSUPPORTED_ANSWER,
            (),
            (RetrievalAttempt(case.question, ("source-one",)),),
        )

        metrics, scores = calculate_metrics((case,), (observation,))

        self.assertEqual(metrics.retrieval_hit_rate, 1.0)
        self.assertTrue(scores[0].retrieval_hit)
        self.assertFalse(scores[0].answer_correct)

    def test_annotated_source_coverage_ignores_unlabelled_extra_sources(self) -> None:
        case = self.cases[0]
        observation = EvaluationObservation(
            case.case_id,
            "Alpha is supported.",
            (
                Citation("Expected", "source-one", "Alpha is supported."),
                Citation("Additional", "another-valid-source", "More context."),
            ),
            (RetrievalAttempt(case.question, ("source-one",)),),
        )

        metrics, scores = calculate_metrics((case,), (observation,))

        self.assertEqual(metrics.annotated_source_coverage, 1.0)
        self.assertEqual(scores[0].citation_hits, 1)
        self.assertEqual(scores[0].citation_total, 1)

    def test_counts_unsupported_claim_inside_an_otherwise_cited_answer(self) -> None:
        case = self.cases[0]
        observation = EvaluationObservation(
            case.case_id,
            "Alpha is supported, and the Moon is made of cheese.",
            (Citation("One", "source-one", "Alpha is supported."),),
            (RetrievalAttempt(case.question, ("source-one",)),),
            ("The Moon is made of cheese.",),
        )

        metrics, scores = calculate_metrics((case,), (observation,))

        self.assertEqual(metrics.unsupported_claims, 1)
        self.assertEqual(scores[0].unsupported_claims, ("The Moon is made of cheese.",))

    def test_counts_high_confidence_unsupported_answers(self) -> None:
        observations = (
            EvaluationObservation("answerable-one", "Alpha", ()),
            EvaluationObservation("answerable-two", UNSUPPORTED_ANSWER, ()),
            EvaluationObservation("unanswerable", "Paris.", ()),
        )

        metrics, _scores = calculate_metrics(self.cases, observations)

        self.assertEqual(metrics.unsupported_claims, 2)
        self.assertAlmostEqual(metrics.unsupported_claim_rate, 2 / 3)

    def test_threshold_failures_are_suitable_for_a_cli_exit_status(self) -> None:
        metrics, _scores = calculate_metrics(
            self.cases,
            (
                EvaluationObservation("answerable-one", UNSUPPORTED_ANSWER, ()),
                EvaluationObservation("answerable-two", UNSUPPORTED_ANSWER, ()),
                EvaluationObservation("unanswerable", UNSUPPORTED_ANSWER, ()),
            ),
        )

        failures = threshold_failures(
            metrics,
            EvaluationThresholds(
                min_retrieval_hit_rate=0.8,
                min_answer_correctness=0.8,
                max_unsupported_claim_rate=0.0,
                min_annotated_source_coverage=0.8,
            ),
        )

        self.assertEqual(len(failures), 3)
        self.assertTrue(any("retrieval hit rate" in failure for failure in failures))

    def test_writes_machine_and_human_readable_reports(self) -> None:
        observations = (
            EvaluationObservation(
                "answerable-one",
                "Alpha",
                (Citation("One", "source-one"),),
                (RetrievalAttempt("Question one", ("source-one",)),),
            ),
            EvaluationObservation("answerable-two", UNSUPPORTED_ANSWER, ()),
            EvaluationObservation("unanswerable", UNSUPPORTED_ANSWER, ()),
        )
        metrics, scores = calculate_metrics(self.cases, observations)
        metadata = EvaluationMetadata(
            source_revision="abcde12345",
            ollama_version="0.33.2",
            chat_model="chat-model",
            chat_model_digest="chat-digest",
            embedding_model="embed-model",
            embedding_model_digest="embed-digest",
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_limit=4,
            relevance_threshold=0.25,
            max_generation_tokens=512,
            dataset_sha256="abc123",
            evaluation_set_sha256="def456",
            duration_seconds=12.5,
            run_at="2026-08-28T12:00:00+00:00",
        )

        with TemporaryDirectory() as temporary_directory:
            machine_path = Path(temporary_directory) / "report.json"
            human_path = write_reports(
                machine_path, metadata, metrics, scores, failures=()
            )

            report = json.loads(machine_path.read_text(encoding="utf-8"))
            summary = human_path.read_text(encoding="utf-8")

        self.assertEqual(report["metadata"]["dataset_sha256"], "abc123")
        self.assertEqual(report["metadata"]["source_revision"], "abcde12345")
        self.assertEqual(report["metadata"]["retrieval_limit"], 4)
        self.assertEqual(report["metrics"]["retrieval_hit_rate"], 0.5)
        self.assertEqual(report["thresholds"]["min_answer_correctness"], 0.75)
        self.assertIn("RAG Evaluation Summary", summary)
        self.assertIn("Retrieval hit rate: 50.0%", summary)


if __name__ == "__main__":
    unittest.main()
