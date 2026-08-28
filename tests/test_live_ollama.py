import os
import unittest

from local_rag.agent import build_assistant
from local_rag.config import load_settings
from local_rag.readiness import check_readiness


@unittest.skipUnless(
    os.getenv("RUN_LIVE_OLLAMA") == "1",
    "set RUN_LIVE_OLLAMA=1 to run the local Ollama smoke test",
)
class LiveOllamaSmokeTests(unittest.TestCase):
    def test_configured_stack_returns_a_grounded_sample_answer(self) -> None:
        settings = load_settings()
        report = check_readiness(settings)
        self.assertTrue(
            report.ready,
            "\n".join(
                f"{check.label}: {check.detail}"
                for check in report.checks
                if not check.ok
            ),
        )

        answer = build_assistant(settings).answer("Who created Python?", [])

        self.assertIn("Guido van Rossum", answer.text)
        self.assertTrue(answer.citations)
        self.assertTrue(
            all(
                citation.url.startswith("https://www.python.org/")
                for citation in answer.citations
            )
        )
