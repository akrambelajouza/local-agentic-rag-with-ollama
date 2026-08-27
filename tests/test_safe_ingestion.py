import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.documents import Document

from local_rag.config import Settings
from local_rag.ingestion import (
    CorpusDocument,
    CorpusValidationError,
    IngestionError,
    deterministic_chunk_id,
    generate_embeddings,
    load_documents,
    _build_index,
    _build_index_with_embeddings,
    _promote_index,
    _run_builder_in_subprocess,
)


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


def build_real_test_index(settings: Settings, destination: Path) -> tuple[int, int]:
    documents = load_documents(settings.dataset_path)
    return _build_index_with_embeddings(
        settings, destination, documents, FakeEmbeddings()
    )


def fail_with_large_error(_settings: Settings, _destination: Path) -> tuple[int, int]:
    raise RuntimeError("large-error:" + "x" * 1_000_000)


class SafeIngestionTests(unittest.TestCase):
    def _settings(self, root: Path) -> Settings:
        return Settings(
            embedding_model="embed-model", chat_model="chat-model",
            model_provider="ollama", dataset_path=root / "data.txt",
            database_location=root / "index", collection_name="rag_data",
        )

    def test_validates_complete_dataset_before_building(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory) / "data.txt"
            dataset.write_text(
                '{"url":"one","title":"One","raw_text":"Text"}\n'
                '{"url":"two","title":"","raw_text":"Text"}\nnot-json\n',
                encoding="utf-8",
            )
            with self.assertRaises(CorpusValidationError) as raised:
                load_documents(dataset)
        self.assertIn("line 2", str(raised.exception))
        self.assertIn("line 3", str(raised.exception))

    def test_rejects_duplicate_document_ids(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory) / "data.txt"
            record = '{"url":"one","title":"One","raw_text":"Text"}\n'
            dataset.write_text(record + record, encoding="utf-8")

            with self.assertRaisesRegex(CorpusValidationError, "duplicates document"):
                load_documents(dataset)

    def test_successful_build_replaces_previous_index(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = self._settings(root)
            settings.database_location.mkdir()
            (settings.database_location / "old.txt").write_text("old", encoding="utf-8")

            def build(_settings: Settings, staging: Path) -> tuple[int, int]:
                (staging / "new.txt").write_text("new", encoding="utf-8")
                return 2, 5

            summary = generate_embeddings(
                settings, build_index=build, clock=iter((10.0, 12.5)).__next__
            )
            self.assertFalse((settings.database_location / "old.txt").exists())
            self.assertEqual((settings.database_location / "new.txt").read_text(), "new")
            self.assertEqual((summary.document_count, summary.chunk_count), (2, 5))
            self.assertEqual(summary.duration_seconds, 2.5)

    def test_failed_build_preserves_previous_index(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = self._settings(root)
            settings.database_location.mkdir()
            marker = settings.database_location / "working.txt"
            marker.write_text("keep", encoding="utf-8")

            def fail(_settings: Settings, staging: Path) -> tuple[int, int]:
                (staging / "partial.txt").write_text("partial", encoding="utf-8")
                raise RuntimeError("embedding failed")

            with self.assertRaisesRegex(IngestionError, "previous index preserved"):
                generate_embeddings(settings, build_index=fail)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertEqual([path.name for path in root.iterdir()], ["index"])

    def test_failed_promotion_restores_previous_index(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            active = root / "index"
            staging = root / "staging"
            active.mkdir()
            staging.mkdir()
            (active / "working.txt").write_text("keep", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")
            original_replace = Path.replace

            def fail_staging_replace(path: Path, target: Path) -> Path:
                if path == staging:
                    raise OSError("persistence rename failed")
                return original_replace(path, target)

            with patch.object(Path, "replace", fail_staging_replace):
                with self.assertRaisesRegex(OSError, "persistence rename failed"):
                    _promote_index(staging, active)

            self.assertEqual((active / "working.txt").read_text(), "keep")
            self.assertTrue(staging.exists())

    def test_rebuild_can_refuse_to_replace_existing_index(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = self._settings(root)
            settings.database_location.mkdir()
            settings = Settings(
                embedding_model=settings.embedding_model,
                chat_model=settings.chat_model,
                model_provider=settings.model_provider,
                dataset_path=settings.dataset_path,
                database_location=settings.database_location,
                collection_name=settings.collection_name,
                rebuild_index=False,
            )

            with self.assertRaisesRegex(IngestionError, "REBUILD_INDEX=true"):
                generate_embeddings(settings, build_index=lambda *_: self.fail())

    def test_chunk_ids_are_deterministic(self) -> None:
        document = CorpusDocument("https://example.test", "Example", "body")
        first = deterministic_chunk_id(document, 0, "chunk")
        self.assertEqual(first, deterministic_chunk_id(document, 0, "chunk"))
        self.assertNotEqual(first, deterministic_chunk_id(document, 1, "chunk"))

    @patch("local_rag.ingestion.RecursiveCharacterTextSplitter")
    @patch("local_rag.ingestion.Chroma")
    @patch("local_rag.ingestion.OllamaEmbeddings")
    def test_repeated_builds_send_identical_ids(
        self, _embeddings, chroma, splitter
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = self._settings(root)
            settings.dataset_path.write_text(
                '{"url":"https://example.test","title":"Example","raw_text":"body"}\n',
                encoding="utf-8",
            )
            splitter.return_value.create_documents.return_value = [
                Document(page_content="chunk", metadata={"source": "https://example.test"})
            ]

            _build_index(settings, root / "first")
            first_ids = chroma.return_value.add_documents.call_args.kwargs["ids"]
            chroma.return_value.add_documents.reset_mock()
            _build_index(settings, root / "second")
            second_ids = chroma.return_value.add_documents.call_args.kwargs["ids"]

        self.assertEqual(first_ids, second_ids)

    def test_real_chroma_build_can_be_promoted_and_repeated(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = self._settings(root)
            settings.dataset_path.write_text(
                '{"url":"https://example.test","title":"Example","raw_text":"body"}\n',
                encoding="utf-8",
            )

            first_staging = root / "first-staging"
            first_staging.mkdir()
            self.assertEqual(
                _run_builder_in_subprocess(
                    build_real_test_index, settings, first_staging
                ),
                (1, 1),
            )
            _promote_index(first_staging, settings.database_location)
            first_ids = self._stored_embedding_ids(settings.database_location)

            second_staging = root / "second-staging"
            second_staging.mkdir()
            _run_builder_in_subprocess(build_real_test_index, settings, second_staging)
            _promote_index(second_staging, settings.database_location)
            second_ids = self._stored_embedding_ids(settings.database_location)

            self.assertEqual(len(first_ids), 1)
            self.assertEqual(second_ids, first_ids)

    def test_large_child_error_does_not_deadlock(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(IngestionError, "large-error"):
                _run_builder_in_subprocess(
                    fail_with_large_error,
                    self._settings(root),
                    root / "staging",
                )

    @staticmethod
    def _stored_embedding_ids(database_location: Path) -> list[str]:
        database_file = database_location / "chroma.sqlite3"
        uri = f"{database_file.resolve().as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        try:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT embedding_id FROM embeddings ORDER BY embedding_id"
                )
            ]
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
