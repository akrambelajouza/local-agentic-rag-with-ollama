import tomllib
import unittest
from pathlib import Path


class ProjectMetadataTests(unittest.TestCase):
    def test_documents_reproducible_install_run_and_test_commands(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads(
            (repository_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        readme = (repository_root / "README.md").read_text(encoding="utf-8")

        self.assertEqual(metadata["project"]["requires-python"], ">=3.11,<3.13")
        self.assertEqual(
            metadata["project"]["dependencies"],
            [
                "langchain==0.3.25",
                "langchain-chroma==0.2.4",
                "langchain-core==0.3.60",
                "langchain-ollama==0.3.3",
                "langchain-text-splitters==0.3.8",
                "pydantic==2.11.4",
                "pypdf==6.16.2",
                "python-dotenv==1.1.0",
                "streamlit==1.45.1",
            ],
        )
        self.assertTrue((repository_root / "requirements.lock").is_file())
        self.assertIn("pip install -r requirements.lock", readme)
        self.assertIn("python -m local_rag.ingestion", readme)
        self.assertIn("python -m local_rag.readiness", readme)
        self.assertIn("python -m local_rag.evaluation_cli", readme)
        self.assertIn("streamlit run app.py", readme)
        self.assertIn("python -m unittest discover -v", readme)
        self.assertIn("Upload one or more text-based PDFs", readme)
        self.assertIn("20 MB", readme)
        self.assertIn("require OCR before upload", readme)
        self.assertEqual(
            metadata["project"]["scripts"]["local-rag-evaluate"],
            "local_rag.evaluation_cli:cli",
        )


if __name__ == "__main__":
    unittest.main()
