import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from local_rag.app import get_assistant, render_app
from local_rag.assistant import Citation, GroundedAnswer
from local_rag.config import Settings
from local_rag.workflow import WorkflowEvent


class StreamlitStartupTests(unittest.TestCase):
    def setUp(self) -> None:
        settings_patcher = patch(
            "local_rag.app.load_settings", return_value=MagicMock(spec=Settings)
        )
        readiness_patcher = patch(
            "local_rag.app.assess_readiness",
            return_value=MagicMock(ready=False, checks=[]),
        )
        settings_patcher.start()
        readiness_patcher.start()
        self.addCleanup(settings_patcher.stop)
        self.addCleanup(readiness_patcher.stop)

    def test_initial_page_renders_without_sending_a_model_request(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"

        app = AppTest.from_file(str(app_path), default_timeout=30).run()

        self.assertEqual([exception.value for exception in app.exception], [])
        self.assertEqual([title.value for title in app.title], ["📚 Local RAG Chatbot"])
        self.assertEqual(len(app.chat_input), 1)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(
            any("readiness" in item.value.lower() for item in app.subheader)
        )
        rendered_text = "\n".join(item.value for item in app.markdown)
        self.assertIn("local document collection", rendered_text)
        self.assertIn("What is Python?", rendered_text)
        self.assertIn("Who created Python?", rendered_text)

    @patch("local_rag.app.assess_readiness")
    @patch("local_rag.app.st")
    def test_historical_answer_renders_its_verified_sources(
        self, streamlit: MagicMock, assess_readiness: MagicMock
    ) -> None:
        assess_readiness.return_value = MagicMock(ready=True, checks=[])
        streamlit.session_state.__contains__.return_value = True
        streamlit.button.return_value = False
        streamlit.session_state.pending_question = None
        streamlit.session_state.messages = [
            AIMessage(
                "Earlier answer",
                additional_kwargs={
                    "citations": [
                        {
                            "title": "Stored source",
                            "url": "https://source.test",
                            "excerpt": "Supporting excerpt from the indexed document.",
                        }
                    ]
                },
            )
        ]
        streamlit.chat_input.return_value = None

        render_app(citations_expanded=True)

        streamlit.markdown.assert_any_call("[https://source.test](https://source.test)")
        streamlit.expander.assert_called_once_with("Stored source", expanded=True)
        streamlit.caption.assert_any_call(
            "Supporting excerpt from the indexed document."
        )

    @patch("local_rag.app.assess_readiness")
    @patch("local_rag.app.st")
    def test_user_can_clear_the_conversation(
        self, streamlit: MagicMock, assess_readiness: MagicMock
    ) -> None:
        assess_readiness.return_value = MagicMock(ready=True, checks=[])
        streamlit.session_state.__contains__.return_value = True
        streamlit.session_state.messages = [AIMessage("Earlier answer")]
        streamlit.button.return_value = True
        streamlit.chat_input.return_value = None

        render_app()

        self.assertEqual(streamlit.session_state.messages, [])
        streamlit.rerun.assert_called_once_with()

    @patch("local_rag.app.assess_readiness")
    @patch("local_rag.app.st")
    def test_submission_shows_progress_and_saves_one_conversation_turn(
        self, streamlit: MagicMock, assess_readiness: MagicMock
    ) -> None:
        assess_readiness.return_value = MagicMock(ready=True, checks=[])
        streamlit.session_state.__contains__.return_value = True
        streamlit.session_state.messages = []
        streamlit.session_state.pending_question = None
        streamlit.button.return_value = False
        streamlit.chat_input.return_value = "What is Python?"
        assistant = MagicMock()
        assistant.answer.return_value = GroundedAnswer(
            "Python is a programming language.",
            (
                Citation(
                    "Python overview",
                    "https://source.test/python",
                    "Python is a high-level programming language.",
                ),
            ),
            (WorkflowEvent("Retrying retrieval with a rewritten query."),),
        )

        render_app(assistant_provider=lambda _settings: assistant)

        assistant.answer.assert_not_called()
        streamlit.rerun.assert_called_once_with()
        self.assertEqual(streamlit.session_state.pending_question, "What is Python?")

        streamlit.rerun.reset_mock()
        streamlit.chat_input.return_value = None
        render_app(assistant_provider=lambda _settings: assistant)

        assistant.answer.assert_called_once()
        self.assertEqual(
            [call.kwargs["disabled"] for call in streamlit.chat_input.call_args_list],
            [False, True],
        )
        streamlit.status.assert_called_once()
        progress = streamlit.status.return_value.__enter__.return_value
        progress.write.assert_any_call("Retrying retrieval with a rewritten query.")
        self.assertEqual(len(streamlit.session_state.messages), 2)
        self.assertEqual(
            streamlit.session_state.messages[-1].content,
            "Python is a programming language.",
        )

    @patch("local_rag.app.assess_readiness")
    @patch("local_rag.app.st")
    def test_expected_local_service_failure_shows_recovery_guidance(
        self, streamlit: MagicMock, assess_readiness: MagicMock
    ) -> None:
        assess_readiness.return_value = MagicMock(ready=True, checks=[])
        streamlit.session_state.__contains__.return_value = True
        streamlit.session_state.messages = []
        streamlit.session_state.pending_question = "What is Python?"
        streamlit.button.return_value = False
        streamlit.chat_input.return_value = None
        assistant = MagicMock()
        assistant.answer.side_effect = ConnectionError("Ollama stopped")

        render_app(assistant_provider=lambda _settings: assistant)

        error_text = " ".join(call.args[0] for call in streamlit.error.call_args_list)
        guidance = " ".join(call.args[0] for call in streamlit.caption.call_args_list)
        self.assertIn("local AI service is unavailable", error_text)
        self.assertIn("ollama serve", guidance)
        self.assertEqual(len(streamlit.session_state.messages), 2)

    @patch("local_rag.app.LOGGER")
    @patch("local_rag.app.assess_readiness")
    @patch("local_rag.app.st")
    def test_model_retrieval_and_generation_failures_remain_recoverable(
        self,
        streamlit: MagicMock,
        assess_readiness: MagicMock,
        _logger: MagicMock,
    ) -> None:
        assess_readiness.return_value = MagicMock(ready=True, checks=[])
        streamlit.session_state.__contains__.return_value = True
        streamlit.button.return_value = False
        streamlit.chat_input.return_value = "Question"

        for error in (
            RuntimeError("model failed"),
            ValueError("retrieval failed"),
            Exception("generation failed"),
        ):
            with self.subTest(error=error):
                streamlit.reset_mock()
                streamlit.session_state.__contains__.return_value = True
                streamlit.session_state.messages = []
                streamlit.session_state.pending_question = "Question"
                streamlit.button.return_value = False
                streamlit.chat_input.return_value = None
                assistant = MagicMock()
                assistant.answer.side_effect = error
                assistant_provider = MagicMock(return_value=assistant)

                render_app(assistant_provider=assistant_provider)

                self.assertTrue(streamlit.error.called)
                self.assertEqual(len(streamlit.session_state.messages), 2)

    @patch("local_rag.app.assess_readiness")
    @patch("local_rag.app.st")
    def test_in_flight_submission_completes_only_the_pending_turn(
        self, streamlit: MagicMock, assess_readiness: MagicMock
    ) -> None:
        assess_readiness.return_value = MagicMock(ready=True, checks=[])
        streamlit.session_state.__contains__.return_value = True
        streamlit.session_state.messages = []
        streamlit.session_state.pending_question = "Original question"
        streamlit.button.return_value = False
        streamlit.chat_input.return_value = "Duplicate question"
        assistant = MagicMock()
        assistant.answer.return_value = GroundedAnswer("Answer", ())

        render_app(assistant_provider=lambda _settings: assistant)

        assistant.answer.assert_called_once()
        self.assertEqual(
            streamlit.session_state.messages[0].content, "Original question"
        )
        self.assertEqual(len(streamlit.session_state.messages), 2)
        self.assertIsNone(streamlit.session_state.pending_question)

    @patch("local_rag.app.build_assistant")
    def test_stable_assistant_resources_are_reused_across_reruns(
        self, build_assistant: MagicMock
    ) -> None:
        settings = Settings(
            embedding_model="embed",
            chat_model="chat",
            model_provider="ollama",
            dataset_path=Path("data.txt"),
            database_location=Path("index"),
            collection_name="rag",
        )
        build_assistant.return_value = object()
        get_assistant.clear()
        self.addCleanup(get_assistant.clear)

        first = get_assistant(settings)
        second = get_assistant(settings)

        self.assertIs(first, second)
        build_assistant.assert_called_once_with(settings)


if __name__ == "__main__":
    unittest.main()
