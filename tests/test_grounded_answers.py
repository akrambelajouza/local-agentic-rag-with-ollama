import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from local_rag.assistant import UNSUPPORTED_ANSWER, GroundedAssistant
from local_rag.retrieval import Evidence
from local_rag.workflow import DirectRetrievalWorkflow, RetrievalResult, WorkflowEvent


class GroundedAnswerTests(unittest.TestCase):
    def test_citation_excerpt_retains_support_late_in_a_chunk(self) -> None:
        model = MagicMock()
        model.invoke.return_value = AIMessage("Late supported fact.")
        retriever = MagicMock()
        retriever.retrieve.return_value = (
            Evidence(
                "A" * 500 + " Late supported fact.",
                "https://source.test",
                "Long source",
                0.9,
            ),
        )

        answer = GroundedAssistant(model, DirectRetrievalWorkflow(retriever)).answer(
            "Question?", []
        )

        self.assertIn("Late supported fact.", answer.citations[0].excerpt)

    def test_agentic_workflow_events_and_evidence_drive_the_answer(self) -> None:
        model = MagicMock()
        model.invoke.return_value = AIMessage("Answer")
        workflow = MagicMock()
        workflow.run.return_value = RetrievalResult(
            True,
            (Evidence("Strong", "https://source.test", "Source", 0.9),),
            (WorkflowEvent("Retrying retrieval with a rewritten query."),),
        )

        answer = GroundedAssistant(model, workflow).answer("Question?", [])

        self.assertEqual(answer.text, "Answer")
        self.assertEqual(answer.events, workflow.run.return_value.events)

    def test_insufficient_agentic_result_declines_without_generation(self) -> None:
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = RetrievalResult(
            False,
            (Evidence("Weak", "https://source.test", "Source", 0.4),),
            (WorkflowEvent("Stopped after reaching the retrieval retry limit."),),
        )

        answer = GroundedAssistant(model, workflow).answer("Question?", [])

        self.assertEqual(answer.text, UNSUPPORTED_ANSWER)
        self.assertEqual(answer.events, workflow.run.return_value.events)
        model.invoke.assert_not_called()

    def test_answer_is_grounded_and_citations_come_only_from_evidence(self) -> None:
        model = MagicMock()
        model.invoke.return_value = AIMessage(
            "Grounded answer. Invented citation: https://invented.test"
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = (
            Evidence("Known fact", "https://source.test", "Source title", 0.9),
        )

        answer = GroundedAssistant(model, DirectRetrievalWorkflow(retriever)).answer(
            "Question?", []
        )

        self.assertNotIn("https://invented.test", answer.text)
        self.assertEqual(
            [(item.title, item.url) for item in answer.citations],
            [("Source title", "https://source.test")],
        )

    def test_model_authored_sources_section_is_not_displayed(self) -> None:
        model = MagicMock()
        model.invoke.return_value = AIMessage(
            "Supported answer.\n\nSources\n- [Invented](https://invented.test)"
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = (
            Evidence("Known fact", "https://source.test", "Source title", 0.9),
        )

        answer = GroundedAssistant(model, DirectRetrievalWorkflow(retriever)).answer(
            "Question?", []
        )

        self.assertEqual(answer.text, "Supported answer.")

    def test_unsupported_question_declines_without_calling_model(self) -> None:
        model = MagicMock()
        retriever = MagicMock()
        retriever.retrieve.return_value = ()

        answer = GroundedAssistant(model, DirectRetrievalWorkflow(retriever)).answer(
            "Unknown?", []
        )

        self.assertEqual(answer.text, UNSUPPORTED_ANSWER)
        self.assertEqual(answer.citations, ())
        model.invoke.assert_not_called()

    def test_current_question_appears_once_after_role_correct_history(self) -> None:
        model = MagicMock()
        model.invoke.return_value = AIMessage("Answer")
        retriever = MagicMock()
        retriever.retrieve.return_value = (
            Evidence("Evidence", "https://source.test", "Source", 0.8),
        )
        history = [HumanMessage("Earlier question"), AIMessage("Earlier answer")]

        GroundedAssistant(model, DirectRetrievalWorkflow(retriever)).answer(
            "Current question", history
        )

        messages = model.invoke.call_args.args[0]
        self.assertEqual(messages[1:3], history)
        self.assertIsInstance(messages[-1], HumanMessage)
        self.assertEqual(messages[-1].content, "Current question")
        self.assertEqual(
            sum(message.content == "Current question" for message in messages), 1
        )


if __name__ == "__main__":
    unittest.main()
