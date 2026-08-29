import json
import tomllib
import unittest
from pathlib import Path


class ReleaseMetadataTests(unittest.TestCase):
    def test_v1_release_metadata_and_validation_links_are_current(self) -> None:
        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        readme = (root / "README.md").read_text(encoding="utf-8")
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        validation = (root / "docs/release-validation.md").read_text(encoding="utf-8")
        live_report_path = root / "evaluation/results/v1.0.0.json"
        live_summary_path = root / "evaluation/results/v1.0.0.md"
        live_report = json.loads(live_report_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["project"]["version"], "1.0.0")
        self.assertIn("releases/tag/v1.0.0", readme)
        self.assertIn("docs/release-validation.md", readme)
        self.assertIn("v1.0.0", changelog)
        self.assertIn("Windows / Python 3.12", validation)
        self.assertIn("Ubuntu / Python 3.11 and 3.12", validation)
        self.assertIn("macOS", validation)
        self.assertNotIn("In progress", validation)
        self.assertTrue(live_summary_path.is_file())
        self.assertEqual(live_report["threshold_failures"], [])
        self.assertEqual(live_report["metrics"]["answer_correctness"], 1.0)
        self.assertEqual(live_report["metrics"]["unsupported_claims"], 0)
        self.assertIn("evaluation/results/v1.0.0.md", readme)


if __name__ == "__main__":
    unittest.main()
