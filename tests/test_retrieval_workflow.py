import unittest
from unittest.mock import MagicMock

from local_rag.retrieval import Evidence, RetrievalInput
from local_rag.workflow import (
    AgenticRetrievalWorkflow,
    ModelEvidenceJudge,
    SufficiencyDecision,
)


class StubRetriever:
    def __init__(self, results: list[tuple[Evidence, ...]]) -> None:
        self.results = results
        self.queries: list[str] = []

    def retrieve(self, request: RetrievalInput) -> tuple[Evidence, ...]:
        self.queries.append(request.query)
        return self.results[len(self.queries) - 1]


class StubJudge:
    def __init__(self, decisions: list[object]) -> None:
        self.decisions = decisions
        self.calls = 0

    def evaluate(
        self,
        question: str,
        query: str,
        evidence: tuple[Evidence, ...],
        *,
        can_retry: bool,
    ) -> SufficiencyDecision:
        decision = self.decisions[self.calls]
        self.calls += 1
        if isinstance(decision, Exception):
            raise decision
        assert isinstance(decision, SufficiencyDecision)
        return decision


def evidence(label: str, relevance: float = 0.9) -> tuple[Evidence, ...]:
    return (Evidence(label, f"https://source.test/{label}", label, relevance),)


class AgenticRetrievalWorkflowTests(unittest.TestCase):
    def test_model_judge_rejects_malformed_structured_response(self) -> None:
        model = MagicMock()
        model.with_structured_output.return_value.invoke.return_value = {
            "unexpected": "tool output"
        }
        judge = ModelEvidenceJudge(model)

        with self.assertRaisesRegex(ValueError, "invalid sufficiency decision"):
            judge.evaluate(
                "Question",
                "Question",
                evidence("weak"),
                can_retry=True,
            )

    def test_sufficient_initial_retrieval_proceeds_without_rewrite(self) -> None:
        retriever = StubRetriever([evidence("initial")])
        judge = StubJudge([SufficiencyDecision(sufficient=True)])

        result = AgenticRetrievalWorkflow(retriever, judge).run("Question")

        self.assertTrue(result.sufficient)
        self.assertEqual(result.evidence, evidence("initial"))
        self.assertEqual(retriever.queries, ["Question"])

    def test_insufficient_retrieval_rewrites_once_and_uses_retry_evidence(self) -> None:
        retriever = StubRetriever([evidence("weak"), evidence("strong")])
        judge = StubJudge(
            [
                SufficiencyDecision(
                    sufficient=False, rewritten_query="more specific query"
                ),
                SufficiencyDecision(sufficient=True),
            ]
        )

        result = AgenticRetrievalWorkflow(retriever, judge).run("Question")

        self.assertTrue(result.sufficient)
        self.assertEqual(result.evidence, evidence("strong"))
        self.assertEqual(retriever.queries, ["Question", "more specific query"])
        self.assertEqual(
            [attempt.query for attempt in result.attempts],
            ["Question", "more specific query"],
        )
        self.assertEqual(result.attempts[0].source_urls, ("https://source.test/weak",))
        self.assertIn("Retrying retrieval", result.events[0].message)

    def test_failed_rewrite_stops_honestly_without_second_retrieval(self) -> None:
        retriever = StubRetriever([evidence("weak", 0.4)])
        judge = StubJudge([SufficiencyDecision(sufficient=False, rewritten_query="  ")])

        result = AgenticRetrievalWorkflow(retriever, judge).run("Question")

        self.assertFalse(result.sufficient)
        self.assertEqual(retriever.queries, ["Question"])
        self.assertIn("could not produce a safe rewrite", result.events[-1].message)

    def test_strong_evidence_recovers_when_judge_cannot_rewrite(self) -> None:
        strong = evidence("direct-answer", 0.8)
        retriever = StubRetriever([strong])
        judge = StubJudge([SufficiencyDecision(sufficient=False, rewritten_query=None)])

        result = AgenticRetrievalWorkflow(
            retriever, judge, strong_evidence_threshold=0.6
        ).run("Question")

        self.assertTrue(result.sufficient)
        self.assertEqual(result.evidence, strong)
        self.assertIn("strong relevance", result.events[-1].message)

    def test_retry_limit_stops_after_two_retrieval_attempts(self) -> None:
        retriever = StubRetriever([evidence("weak", 0.4), evidence("still-weak", 0.45)])
        judge = StubJudge(
            [
                SufficiencyDecision(sufficient=False, rewritten_query="retry"),
                SufficiencyDecision(sufficient=False, rewritten_query="third attempt"),
            ]
        )

        result = AgenticRetrievalWorkflow(retriever, judge).run("Question")

        self.assertFalse(result.sufficient)
        self.assertEqual(retriever.queries, ["Question", "retry"])
        self.assertIn("retry limit", result.events[-1].message)

    def test_strong_final_evidence_recovers_from_a_false_negative_judgment(
        self,
    ) -> None:
        strong = (
            Evidence(
                "The answer is stated directly.",
                "https://source.test/strong",
                "Strong source",
                0.8,
            ),
        )
        retriever = StubRetriever([evidence("weak"), strong])
        judge = StubJudge(
            [
                SufficiencyDecision(sufficient=False, rewritten_query="retry"),
                SufficiencyDecision(sufficient=False),
            ]
        )

        result = AgenticRetrievalWorkflow(
            retriever, judge, strong_evidence_threshold=0.6
        ).run("Question")

        self.assertTrue(result.sufficient)
        self.assertEqual(result.evidence, strong)
        self.assertIn("strong relevance", result.events[-1].message)

    def test_malformed_decision_stops_without_leaking_an_exception(self) -> None:
        retriever = StubRetriever([evidence("weak")])
        judge = StubJudge([ValueError("malformed tool call")])

        result = AgenticRetrievalWorkflow(retriever, judge).run("Question")

        self.assertFalse(result.sufficient)
        self.assertEqual(retriever.queries, ["Question"])
        self.assertIn("invalid sufficiency decision", result.events[-1].message)


if __name__ == "__main__":
    unittest.main()
