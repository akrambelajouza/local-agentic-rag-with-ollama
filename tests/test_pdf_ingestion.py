import shutil
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from local_rag.config import Settings
from local_rag.ingestion import IngestionError, load_documents
from local_rag.pdf_ingestion import PdfUpload, ingest_pdf_uploads


class _TextPage:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


class _Reader:
    is_encrypted = False

    def __init__(self, pages: tuple[_TextPage, ...]) -> None:
        self.pages = pages


def _real_pdf_with_text(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    content = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content.set_data(f"BT /F1 12 Tf 72 200 Td ({escaped}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(content)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _real_blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class PdfIngestionTests(unittest.TestCase):
    def test_password_protected_pdf_is_rejected_before_ingestion(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                embedding_model="embed-model",
                chat_model="chat-model",
                model_provider="ollama",
                dataset_path=root / "data.txt",
                database_location=root / "index",
                collection_name="rag_data",
            )
            reader = _Reader(())
            reader.is_encrypted = True

            with patch("local_rag.pdf_ingestion.PdfReader", return_value=reader):
                with self.assertRaisesRegex(ValueError, "password-protected"):
                    ingest_pdf_uploads(
                        settings,
                        [PdfUpload("private.pdf", b"%PDF-locked")],
                        build_index=lambda *_: self.fail("index must not start"),
                    )

            self.assertFalse(settings.dataset_path.exists())

    def test_scanned_or_blank_pdf_explains_that_ocr_is_required(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                embedding_model="embed-model",
                chat_model="chat-model",
                model_provider="ollama",
                dataset_path=root / "data.txt",
                database_location=root / "index",
                collection_name="rag_data",
            )

            with self.assertRaisesRegex(ValueError, "require OCR"):
                ingest_pdf_uploads(
                    settings,
                    [PdfUpload("scan.pdf", _real_blank_pdf())],
                    build_index=lambda *_: self.fail("index must not start"),
                )

            self.assertFalse(settings.dataset_path.exists())
            self.assertFalse(settings.database_location.exists())

    def test_real_pdf_text_is_extracted_by_the_supported_parser(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                embedding_model="embed-model",
                chat_model="chat-model",
                model_provider="ollama",
                dataset_path=root / "data.txt",
                database_location=root / "index",
                collection_name="rag_data",
            )

            def build_index(candidate: Settings, destination: Path) -> tuple[int, int]:
                documents = load_documents(candidate.dataset_path)
                (destination / "built.txt").write_text("complete", encoding="utf-8")
                return len(documents), len(documents)

            ingest_pdf_uploads(
                settings,
                [PdfUpload("facts.pdf", _real_pdf_with_text("Mars has two moons."))],
                build_index=build_index,
            )
            documents = load_documents(settings.dataset_path)

        self.assertEqual(len(documents), 1)
        self.assertIn("Mars has two moons.", documents[0].raw_text)

    def test_uploaded_pdf_pages_are_added_to_the_corpus_and_index(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                embedding_model="embed-model",
                chat_model="chat-model",
                model_provider="ollama",
                dataset_path=root / "data.txt",
                database_location=root / "index",
                collection_name="rag_data",
            )
            settings.dataset_path.write_text(
                '{"url":"existing","title":"Existing","raw_text":"Original"}\n',
                encoding="utf-8",
            )

            def build_index(candidate: Settings, destination: Path) -> tuple[int, int]:
                documents = load_documents(candidate.dataset_path)
                (destination / "built.txt").write_text("complete", encoding="utf-8")
                return len(documents), len(documents)

            reader = _Reader(
                (_TextPage("First page knowledge."), _TextPage("Second page facts."))
            )
            with patch("local_rag.pdf_ingestion.PdfReader", return_value=reader):
                summary = ingest_pdf_uploads(
                    settings,
                    [PdfUpload("handbook.pdf", b"%PDF-test")],
                    build_index=build_index,
                )

            documents = load_documents(settings.dataset_path)

        self.assertEqual(summary.uploaded_file_count, 1)
        self.assertEqual(summary.extracted_page_count, 2)
        self.assertEqual(summary.added_document_count, 2)
        self.assertEqual(
            [document.title for document in documents],
            ["Existing", "handbook.pdf (page 1)", "handbook.pdf (page 2)"],
        )
        self.assertEqual(documents[1].raw_text, "First page knowledge.")
        self.assertIn("handbook.pdf#page=1", documents[1].url)

    def test_oversized_pdf_is_rejected_before_corpus_or_index_changes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                embedding_model="embed-model",
                chat_model="chat-model",
                model_provider="ollama",
                dataset_path=root / "data.txt",
                database_location=root / "index",
                collection_name="rag_data",
            )
            original = b'{"url":"existing","title":"Existing","raw_text":"Original"}\n'
            settings.dataset_path.write_bytes(original)
            settings.database_location.mkdir()
            (settings.database_location / "old.txt").write_text("old", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "20 MB"):
                ingest_pdf_uploads(
                    settings,
                    [PdfUpload("too-large.pdf", b"%PDF-" + b"x" * (20 * 1024 * 1024))],
                    build_index=lambda *_: self.fail("index build must not start"),
                )

            self.assertEqual(settings.dataset_path.read_bytes(), original)
            self.assertEqual(
                (settings.database_location / "old.txt").read_text(encoding="utf-8"),
                "old",
            )

    def test_reuploading_the_same_pdf_is_an_idempotent_no_op(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                embedding_model="embed-model",
                chat_model="chat-model",
                model_provider="ollama",
                dataset_path=root / "data.txt",
                database_location=root / "index",
                collection_name="rag_data",
            )
            upload = PdfUpload("handbook.pdf", b"%PDF-same")
            reader = _Reader((_TextPage("Stable knowledge."),))

            def initial_build(
                candidate: Settings, destination: Path
            ) -> tuple[int, int]:
                count = len(load_documents(candidate.dataset_path))
                (destination / "built.txt").write_text("initial", encoding="utf-8")
                return count, count

            with patch("local_rag.pdf_ingestion.PdfReader", return_value=reader):
                ingest_pdf_uploads(settings, [upload], build_index=initial_build)
                original_corpus = settings.dataset_path.read_bytes()
                summary = ingest_pdf_uploads(
                    settings,
                    [upload],
                    build_index=lambda *_: self.fail(
                        "duplicate must not rebuild index"
                    ),
                )

            self.assertEqual(summary.added_document_count, 0)
            self.assertEqual(summary.skipped_duplicate_count, 1)
            self.assertEqual(settings.dataset_path.read_bytes(), original_corpus)
            self.assertEqual(
                (settings.database_location / "built.txt").read_text(encoding="utf-8"),
                "initial",
            )

    def test_renaming_the_same_pdf_does_not_duplicate_its_pages(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                embedding_model="embed-model",
                chat_model="chat-model",
                model_provider="ollama",
                dataset_path=root / "data.txt",
                database_location=root / "index",
                collection_name="rag_data",
            )
            content = b"%PDF-identical-content"
            reader = _Reader((_TextPage("Stable knowledge."),))

            def initial_build(
                candidate: Settings, destination: Path
            ) -> tuple[int, int]:
                count = len(load_documents(candidate.dataset_path))
                (destination / "built.txt").write_text("initial", encoding="utf-8")
                return count, count

            with patch("local_rag.pdf_ingestion.PdfReader", return_value=reader):
                ingest_pdf_uploads(
                    settings,
                    [PdfUpload("original.pdf", content)],
                    build_index=initial_build,
                )
                summary = ingest_pdf_uploads(
                    settings,
                    [PdfUpload("renamed.pdf", content)],
                    build_index=lambda *_: self.fail(
                        "renamed duplicate must not rebuild"
                    ),
                )
            documents = load_documents(settings.dataset_path)

        self.assertEqual(summary.added_document_count, 0)
        self.assertEqual(summary.skipped_duplicate_count, 1)
        self.assertEqual(
            [document.title for document in documents], ["original.pdf (page 1)"]
        )

    def test_duplicate_files_in_one_upload_batch_are_indexed_once(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                embedding_model="embed-model",
                chat_model="chat-model",
                model_provider="ollama",
                dataset_path=root / "data.txt",
                database_location=root / "index",
                collection_name="rag_data",
            )
            upload = PdfUpload("handbook.pdf", b"%PDF-same-batch")
            reader = _Reader((_TextPage("Stable knowledge."),))

            def build_index(candidate: Settings, destination: Path) -> tuple[int, int]:
                count = len(load_documents(candidate.dataset_path))
                (destination / "built.txt").write_text("complete", encoding="utf-8")
                return count, count

            with patch("local_rag.pdf_ingestion.PdfReader", return_value=reader):
                summary = ingest_pdf_uploads(
                    settings, [upload, upload], build_index=build_index
                )
            documents = load_documents(settings.dataset_path)

        self.assertEqual(summary.uploaded_file_count, 2)
        self.assertEqual(summary.added_document_count, 1)
        self.assertEqual(summary.skipped_duplicate_count, 1)
        self.assertEqual(len(documents), 1)

    def test_duplicate_pdf_rebuilds_when_the_local_index_is_missing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                embedding_model="embed-model",
                chat_model="chat-model",
                model_provider="ollama",
                dataset_path=root / "data.txt",
                database_location=root / "index",
                collection_name="rag_data",
            )
            upload = PdfUpload("handbook.pdf", b"%PDF-recover")
            reader = _Reader((_TextPage("Recoverable knowledge."),))

            def build_index(candidate: Settings, destination: Path) -> tuple[int, int]:
                count = len(load_documents(candidate.dataset_path))
                (destination / "built.txt").write_text("complete", encoding="utf-8")
                return count, count

            with patch("local_rag.pdf_ingestion.PdfReader", return_value=reader):
                ingest_pdf_uploads(settings, [upload], build_index=build_index)
                shutil.rmtree(settings.database_location)
                summary = ingest_pdf_uploads(
                    settings, [upload], build_index=build_index
                )

            self.assertEqual(summary.added_document_count, 0)
            self.assertEqual(summary.skipped_duplicate_count, 1)
            self.assertIsNotNone(summary.index)
            self.assertTrue((settings.database_location / "built.txt").is_file())

    def test_failed_index_rebuild_restores_the_previous_corpus_and_index(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                embedding_model="embed-model",
                chat_model="chat-model",
                model_provider="ollama",
                dataset_path=root / "data.txt",
                database_location=root / "index",
                collection_name="rag_data",
            )
            original = b'{"url":"existing","title":"Existing","raw_text":"Original"}\n'
            settings.dataset_path.write_bytes(original)
            settings.database_location.mkdir()
            (settings.database_location / "old.txt").write_text("old", encoding="utf-8")
            reader = _Reader((_TextPage("New PDF knowledge."),))

            def fail_build(_settings: Settings, _destination: Path) -> tuple[int, int]:
                raise RuntimeError("embedding service stopped")

            with patch("local_rag.pdf_ingestion.PdfReader", return_value=reader):
                with self.assertRaisesRegex(IngestionError, "previous index preserved"):
                    ingest_pdf_uploads(
                        settings,
                        [PdfUpload("handbook.pdf", b"%PDF-new")],
                        build_index=fail_build,
                    )

            self.assertEqual(settings.dataset_path.read_bytes(), original)
            self.assertEqual(
                (settings.database_location / "old.txt").read_text(encoding="utf-8"),
                "old",
            )


if __name__ == "__main__":
    unittest.main()
