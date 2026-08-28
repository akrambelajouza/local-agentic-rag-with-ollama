import tomllib
import unittest
from pathlib import Path


class QualityConfigurationTests(unittest.TestCase):
    def test_local_and_hosted_quality_commands_stay_aligned(self) -> None:
        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        readme = (root / "README.md").read_text(encoding="utf-8")
        workflow = (root / ".github/workflows/quality.yml").read_text(encoding="utf-8")

        self.assertTrue((root / "requirements-dev.lock").is_file())
        self.assertEqual(metadata["tool"]["coverage"]["report"]["fail_under"], 85)
        self.assertEqual(
            metadata["tool"]["ruff"]["lint"]["select"], ["E", "F", "I", "B"]
        )
        for command in (
            "python scripts/quality.py",
            "python -m ruff format --check .",
            "python -m ruff check .",
            "python -m unittest discover -v",
            "python -m unittest tests.test_app -v",
            "python -m coverage report",
            "python -m pip check",
        ):
            self.assertIn(command, readme)
        self.assertIn("python scripts/quality.py", workflow)
        self.assertIn('RUN_LIVE_OLLAMA: "0"', workflow)
        self.assertIn("cache: pip", workflow)
        self.assertIn("os: [ubuntu-latest, windows-latest]", workflow)
        self.assertIn('python-version: ["3.11", "3.12"]', workflow)
        self.assertNotIn("actions/checkout@v7", workflow)
        self.assertNotIn("actions/setup-python@v7", workflow)


if __name__ == "__main__":
    unittest.main()
