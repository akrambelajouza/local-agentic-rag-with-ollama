import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from local_rag.app import render_app


class StreamlitStartupTests(unittest.TestCase):
    def test_initial_page_renders_without_sending_a_model_request(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"

        app = AppTest.from_file(str(app_path), default_timeout=30).run()

        self.assertEqual([exception.value for exception in app.exception], [])
        self.assertEqual([title.value for title in app.title], ["🦜 Agentic RAG Chatbot"])
        self.assertEqual(len(app.chat_input), 1)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(any("readiness" in item.value.lower() for item in app.subheader))

    @patch("local_rag.app.assess_readiness")
    @patch("local_rag.app.st")
    def test_historical_answer_renders_its_verified_sources(
        self, streamlit: MagicMock, assess_readiness: MagicMock
    ) -> None:
        assess_readiness.return_value = MagicMock(ready=True, checks=[])
        streamlit.session_state.__contains__.return_value = True
        streamlit.session_state.messages = [
            AIMessage(
                "Earlier answer",
                additional_kwargs={
                    "citations": [
                        {"title": "Stored source", "url": "https://source.test"}
                    ]
                },
            )
        ]
        streamlit.chat_input.return_value = None

        render_app()

        streamlit.markdown.assert_any_call(
            "- [Stored source](https://source.test)"
        )


if __name__ == "__main__":
    unittest.main()
