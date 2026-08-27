import unittest
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from local_rag.config import ConfigurationError, Settings, load_settings


class SettingsTests(unittest.TestCase):
    def test_loads_typed_settings_from_an_explicit_mapping(self) -> None:
        settings = Settings.from_mapping(
            {
                "EMBEDDING_MODEL": "mxbai-embed-large",
                "CHAT_MODEL": "llama3.2:3b",
                "MODEL_PROVIDER": "ollama",
                "DATASET_STORAGE_FOLDER": "fixtures",
                "DATABASE_LOCATION": "index",
                "COLLECTION_NAME": "portfolio_rag",
            },
            base_directory=Path("workspace"),
        )

        self.assertEqual(settings.embedding_model, "mxbai-embed-large")
        self.assertEqual(settings.chat_model, "llama3.2:3b")
        self.assertEqual(settings.model_provider, "ollama")
        self.assertEqual(settings.dataset_path, Path("workspace/fixtures/data.txt"))
        self.assertEqual(settings.database_location, Path("workspace/index"))
        self.assertEqual(settings.collection_name, "portfolio_rag")

    def test_reports_all_missing_required_settings(self) -> None:
        with self.assertRaisesRegex(
            ConfigurationError,
            "CHAT_MODEL, COLLECTION_NAME, DATABASE_LOCATION",
        ):
            Settings.from_mapping(
                {
                    "EMBEDDING_MODEL": "mxbai-embed-large",
                    "MODEL_PROVIDER": "ollama",
                    "DATASET_STORAGE_FOLDER": "fixtures",
                }
            )

    def test_loads_dotenv_values_with_environment_overrides(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            env_file = root / ".env"
            env_file.write_text(
                "\n".join(
                    (
                        'EMBEDDING_MODEL="embed-from-file"',
                        'CHAT_MODEL="chat-from-file"',
                        'MODEL_PROVIDER="ollama"',
                        'DATASET_STORAGE_FOLDER="datasets"',
                        'DATABASE_LOCATION="chroma_db"',
                        'COLLECTION_NAME="rag_data"',
                    )
                ),
                encoding="utf-8",
            )

            with patch.dict(environ, {"CHAT_MODEL": "chat-from-environment"}, clear=True):
                settings = load_settings(env_file)

            self.assertEqual(settings.chat_model, "chat-from-environment")
            self.assertEqual(settings.embedding_model, "embed-from-file")
            self.assertEqual(settings.dataset_path, root / "datasets/data.txt")


if __name__ == "__main__":
    unittest.main()
