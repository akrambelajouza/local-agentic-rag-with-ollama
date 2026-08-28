import hashlib
import io
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from local_rag.config import Settings
from local_rag.readiness import (
    ReadinessReport,
    _collection_count,
    check_readiness,
    render_cli_report,
)


class ReadinessTests(unittest.TestCase):
    def _settings(self, root: Path) -> Settings:
        return Settings(
            embedding_model="embed-model",
            chat_model="chat-model:latest",
            model_provider="ollama",
            dataset_path=root / "datasets" / "data.txt",
            database_location=root / "chroma_db",
            collection_name="rag_data",
            ollama_base_url="http://localhost:11434",
        )

    def test_reports_a_ready_local_environment(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = self._settings(root)
            settings.dataset_path.parent.mkdir()
            settings.dataset_path.write_text(
                '{"url":"https://example.test","title":"Example","raw_text":"Text"}\n',
                encoding="utf-8",
            )
            settings.database_location.mkdir()
            report = check_readiness(
                settings,
                list_models=lambda _: {"embed-model:latest", "chat-model:latest"},
                collection_count=lambda *_: 3,
            )

        self.assertTrue(report.ready)
        self.assertTrue(all(check.ok for check in report.checks))

    def test_reports_each_partial_configuration_problem_with_an_action(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            settings = self._settings(Path(temporary_directory))
            report = check_readiness(
                settings,
                list_models=lambda _: {"chat-model:latest"},
                collection_count=lambda *_: 0,
            )

        failed = {check.key: check for check in report.checks if not check.ok}
        self.assertEqual(set(failed), {"dataset", "models", "collection"})
        self.assertTrue(all(check.action for check in failed.values()))

    def test_reports_unavailable_ollama_while_still_checking_collection(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = self._settings(root)
            settings.dataset_path.parent.mkdir()
            settings.dataset_path.write_text("{}\n", encoding="utf-8")
            settings.database_location.mkdir()

            def unavailable(_: str) -> set[str]:
                raise ConnectionError("connection refused")

            report = check_readiness(
                settings,
                list_models=unavailable,
                collection_count=lambda *_: 2,
            )

        self.assertFalse(report.ready)
        self.assertEqual(
            [check.key for check in report.checks if not check.ok],
            ["ollama", "models"],
        )
        self.assertIn("ollama serve", report.checks[2].action)
        self.assertTrue(report.checks[4].ok)

    def test_cli_report_returns_nonzero_and_prints_corrective_actions(self) -> None:
        report = ReadinessReport.configuration_failure("Missing CHAT_MODEL")
        output = io.StringIO()

        exit_code = render_cli_report(report, output)

        self.assertEqual(exit_code, 1)
        self.assertIn("ACTION:", output.getvalue())
        self.assertIn("Missing CHAT_MODEL", output.getvalue())

    def test_missing_collection_check_does_not_create_database_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = self._settings(root)
            settings.dataset_path.parent.mkdir()
            settings.dataset_path.write_text("{}\n", encoding="utf-8")
            settings.database_location.mkdir()

            report = check_readiness(
                settings,
                list_models=lambda _: {"embed-model:latest", "chat-model:latest"},
            )

            self.assertFalse(report.ready)
            self.assertEqual(list(settings.database_location.iterdir()), [])

    def test_populated_collection_check_is_read_only(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_location = Path(temporary_directory)
            database_file = database_location / "chroma.sqlite3"
            connection = sqlite3.connect(database_file)
            try:
                connection.executescript(
                    """
                    CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT NOT NULL);
                    CREATE TABLE segments (
                        id TEXT PRIMARY KEY,
                        collection TEXT NOT NULL
                    );
                    CREATE TABLE embeddings (
                        id INTEGER PRIMARY KEY,
                        segment_id TEXT NOT NULL
                    );
                    INSERT INTO collections VALUES ('collection-id', 'rag_data');
                    INSERT INTO segments VALUES ('segment-id', 'collection-id');
                    INSERT INTO embeddings (segment_id) VALUES ('segment-id');
                    INSERT INTO embeddings (segment_id) VALUES ('segment-id');
                    """
                )
            finally:
                connection.close()
            before = hashlib.sha256(database_file.read_bytes()).digest()

            count = _collection_count(database_location, "rag_data")

            after = hashlib.sha256(database_file.read_bytes()).digest()
            self.assertEqual(count, 2)
            self.assertEqual(after, before)
            self.assertEqual(
                [path.name for path in database_location.iterdir()],
                ["chroma.sqlite3"],
            )


if __name__ == "__main__":
    unittest.main()
