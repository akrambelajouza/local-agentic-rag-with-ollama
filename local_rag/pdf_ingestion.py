"""Local PDF extraction and transactional corpus ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from pypdf import PdfReader

from local_rag.config import Settings
from local_rag.ingestion import (
    BuildIndex,
    CorpusDocument,
    IngestionSummary,
    generate_embeddings,
    load_documents,
)

MAX_PDF_BYTES = 20 * 1024 * 1024


class PdfIngestionError(ValueError):
    """Raised when uploaded PDF content cannot be safely ingested."""


@dataclass(frozen=True, slots=True)
class PdfUpload:
    filename: str
    content: bytes


@dataclass(frozen=True, slots=True)
class PdfIngestionSummary:
    uploaded_file_count: int
    extracted_page_count: int
    added_document_count: int
    skipped_duplicate_count: int
    index: IngestionSummary | None


def ingest_pdf_uploads(
    settings: Settings,
    uploads: Iterable[PdfUpload],
    *,
    build_index: BuildIndex | None = None,
) -> PdfIngestionSummary:
    """Extract PDFs, merge their pages into the corpus, and rebuild the index."""

    uploaded = tuple(uploads)
    if not uploaded:
        raise PdfIngestionError("Select at least one PDF to ingest.")

    extracted = [
        document for upload in uploaded for document in _extract_pdf_documents(upload)
    ]
    existing = (
        load_documents(settings.dataset_path) if settings.dataset_path.exists() else []
    )
    known_ids = {_document_identity(document) for document in existing}
    added = []
    for document in extracted:
        identity = _document_identity(document)
        if identity in known_ids:
            continue
        known_ids.add(identity)
        added.append(document)
    skipped = len(extracted) - len(added)
    if not added and settings.database_location.exists():
        return PdfIngestionSummary(
            uploaded_file_count=len(uploaded),
            extracted_page_count=len(extracted),
            added_document_count=0,
            skipped_duplicate_count=skipped,
            index=None,
        )
    merged = [*existing, *added]

    original = (
        settings.dataset_path.read_bytes() if settings.dataset_path.exists() else None
    )
    _write_corpus_atomically(settings.dataset_path, merged)
    try:
        index = generate_embeddings(settings, build_index=build_index)
    except Exception:
        _restore_corpus(settings.dataset_path, original)
        raise

    return PdfIngestionSummary(
        uploaded_file_count=len(uploaded),
        extracted_page_count=len(extracted),
        added_document_count=len(added),
        skipped_duplicate_count=skipped,
        index=index,
    )


def _extract_pdf_documents(upload: PdfUpload) -> list[CorpusDocument]:
    filename = _safe_filename(upload.filename)
    if len(upload.content) > MAX_PDF_BYTES:
        raise PdfIngestionError(f"{filename} is larger than the 20 MB upload limit.")
    if not upload.content.startswith(b"%PDF-"):
        raise PdfIngestionError(f"{filename} is not a valid PDF file.")
    try:
        reader = PdfReader(BytesIO(upload.content))
        if reader.is_encrypted:
            raise PdfIngestionError(
                f"{filename} is password-protected; upload an unlocked PDF."
            )
        digest = hashlib.sha256(upload.content).hexdigest()
        source = f"local-pdf://{digest}/{quote(filename)}"
        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                documents.append(
                    CorpusDocument(
                        f"{source}#page={page_number}",
                        f"{filename} (page {page_number})",
                        text,
                    )
                )
    except PdfIngestionError:
        raise
    except Exception as error:
        raise PdfIngestionError(f"Could not read {filename}: {error}") from error
    if not documents:
        raise PdfIngestionError(
            f"{filename} contains no extractable text; scanned PDFs require OCR first."
        )
    return documents


def _safe_filename(filename: str) -> str:
    cleaned = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    return cleaned or "uploaded.pdf"


def _document_identity(document: CorpusDocument) -> str:
    parsed = urlparse(document.url)
    if parsed.scheme == "local-pdf" and parsed.netloc:
        page = parse_qs(parsed.fragment).get("page", [""])[0]
        if page:
            return f"local-pdf:{parsed.netloc}:page={page}"
    return document.document_id


def _write_corpus_atomically(path: Path, documents: list[CorpusDocument]) -> None:
    payload = "".join(
        json.dumps(
            {
                "url": document.url,
                "title": document.title,
                "raw_text": document.raw_text,
            },
            ensure_ascii=False,
        )
        + "\n"
        for document in documents
    ).encode("utf-8")
    _replace_file_atomically(path, payload)


def _replace_file_atomically(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _restore_corpus(path: Path, original: bytes | None) -> None:
    if original is None:
        path.unlink(missing_ok=True)
        return
    _replace_file_atomically(path, original)
