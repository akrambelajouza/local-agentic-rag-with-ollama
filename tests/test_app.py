import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitStartupTests(unittest.TestCase):
    def test_initial_page_renders_without_sending_a_model_request(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"

        app = AppTest.from_file(str(app_path), default_timeout=30).run()

        self.assertEqual([exception.value for exception in app.exception], [])
        self.assertEqual([title.value for title in app.title], ["🦜 Agentic RAG Chatbot"])
        self.assertEqual(len(app.chat_input), 1)


if __name__ == "__main__":
    unittest.main()
