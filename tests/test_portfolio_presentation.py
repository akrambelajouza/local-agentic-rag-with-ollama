import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from local_rag.evaluation import calculate_metrics
from local_rag.evaluation_io import load_evaluation_cases
from local_rag.reference_evaluation import build_reference_observations


class PortfolioPresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.readme = (self.root / "README.md").read_text(encoding="utf-8")

    def test_portfolio_materials_cover_run_evaluate_and_license_paths(self) -> None:
        for required_text in (
            "git clone https://github.com/akrambelajouza/",
            "python -m local_rag.readiness",
            "python -m local_rag.ingestion",
            "streamlit run app.py",
            "python scripts/quality.py",
            "python -m local_rag.evaluation_cli",
            "```mermaid",
            "docs/portfolio-demo.svg",
        ):
            self.assertIn(required_text, self.readme)
        for required_path in (
            "LICENSE",
            "docs/portfolio-demo.svg",
            "docs/social-preview.md",
            "portfolio_demo.py",
        ):
            self.assertTrue((self.root / required_path).is_file())

    def test_reference_evaluation_is_complete_and_perfect_by_construction(self) -> None:
        cases = load_evaluation_cases(self.root / "evaluation/questions.jsonl")

        metrics, scores = calculate_metrics(cases, build_reference_observations(cases))

        self.assertEqual(len(scores), len(cases))
        self.assertEqual(metrics.retrieval_hit_rate, 1.0)
        self.assertEqual(metrics.answer_correctness, 1.0)
        self.assertEqual(metrics.unsupported_claim_rate, 0.0)
        self.assertEqual(metrics.annotated_source_coverage, 1.0)

    def test_no_ollama_portfolio_preview_is_explicitly_read_only(self) -> None:
        app = AppTest.from_file(
            str(self.root / "portfolio_demo.py"), default_timeout=30
        ).run()

        self.assertEqual([exception.value for exception in app.exception], [])
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(any("read-only" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
