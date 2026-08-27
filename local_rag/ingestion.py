"""Validated, deterministic, transactional corpus ingestion."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import shutil
import sys
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_rag.config import Settings, load_settings


class CorpusValidationError(ValueError):
    """Raised after the entire corpus has been checked for invalid records."""


class IngestionError(RuntimeError):
    """Raised when a safe index rebuild cannot be completed."""


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    url: str
    title: str
    raw_text: str

    @property
    def document_id(self) -> str:
        payload = f"{self.url}\0{self.title}\0{self.raw_text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    document_count: int
    chunk_count: int
    failure_count: int
    embedding_model: str
    destination: Path
    duration_seconds: float


BuildIndex = Callable[[Settings, Path], tuple[int, int]]


def load_documents(dataset_path: str | Path) -> list[CorpusDocument]:
    """Validate every JSONL record before returning any corpus documents."""

    documents: list[CorpusDocument] = []
    errors: list[str] = []
    document_lines: dict[str, int] = {}
    path = Path(dataset_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CorpusValidationError(f"Cannot read dataset {path}: {error}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {line_number}: invalid JSON ({error.msg})")
            continue
        if not isinstance(record, dict):
            errors.append(f"line {line_number}: expected a JSON object")
            continue
        invalid_fields = [
            field
            for field in ("url", "title", "raw_text")
            if not isinstance(record.get(field), str) or not record[field].strip()
        ]
        if invalid_fields:
            errors.append(
                f"line {line_number}: missing or empty fields: {', '.join(invalid_fields)}"
            )
            continue
        document = CorpusDocument(
            record["url"].strip(), record["title"].strip(), record["raw_text"]
        )
        if document.document_id in document_lines:
            errors.append(
                f"line {line_number}: duplicates document on line "
                f"{document_lines[document.document_id]}"
            )
            continue
        document_lines[document.document_id] = line_number
        documents.append(document)

    if not documents and not errors:
        errors.append("dataset contains no documents")
    if errors:
        raise CorpusValidationError("Dataset validation failed:\n- " + "\n- ".join(errors))
    return documents


def generate_embeddings(
    settings: Settings,
    *,
    build_index: BuildIndex | None = None,
    clock: Callable[[], float] = perf_counter,
) -> IngestionSummary:
    """Build in isolation and promote only a fully successful index."""

    if settings.database_location.exists() and not settings.rebuild_index:
        raise IngestionError(
            f"Index already exists at {settings.database_location}; set REBUILD_INDEX=true to replace it."
        )
    started_at = clock()
    destination = settings.database_location
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.build-", dir=destination.parent)
    )
    try:
        if build_index is None:
            document_count, chunk_count = _run_builder_in_subprocess(
                _build_index, settings, staging
            )
        else:
            document_count, chunk_count = build_index(settings, staging)
        _promote_index(staging, destination)
    except Exception as error:
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError as cleanup_error:
                raise IngestionError(
                    f"{error}; staging cleanup also failed: {cleanup_error}"
                ) from error
        if isinstance(error, (CorpusValidationError, IngestionError)):
            raise
        raise IngestionError(f"Index build failed; previous index preserved: {error}") from error
    return IngestionSummary(
        document_count, chunk_count, 0, settings.embedding_model,
        destination, clock() - started_at,
    )


def deterministic_chunk_id(document: CorpusDocument, index: int, content: str) -> str:
    payload = f"{document.document_id}\0{index}\0{content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def format_summary(summary: IngestionSummary) -> str:
    return (
        f"Documents: {summary.document_count}\nChunks: {summary.chunk_count}\n"
        f"Failures: {summary.failure_count}\nEmbedding model: {summary.embedding_model}\n"
        f"Destination: {summary.destination}\nDuration: {summary.duration_seconds:.2f}s"
    )


def main() -> None:
    try:
        summary = generate_embeddings(load_settings())
    except Exception as error:
        print(f"Ingestion failed\nFailures: 1\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(format_summary(summary))


def _build_index(settings: Settings, destination: Path) -> tuple[int, int]:
    documents = load_documents(settings.dataset_path)
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model, base_url=settings.ollama_base_url
    )
    return _build_index_with_embeddings(settings, destination, documents, embeddings)


def _build_index_with_embeddings(
    settings: Settings,
    destination: Path,
    documents: list[CorpusDocument],
    embeddings,
) -> tuple[int, int]:
    vector_store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(destination),
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = []
    identifiers: list[str] = []
    for document in documents:
        document_chunks = splitter.create_documents(
            [document.raw_text],
            metadatas=[{
                "source": document.url,
                "title": document.title,
                "document_id": document.document_id,
            }],
        )
        chunks.extend(document_chunks)
        identifiers.extend(
            deterministic_chunk_id(document, index, chunk.page_content)
            for index, chunk in enumerate(document_chunks)
        )
    for start in range(0, len(chunks), settings.batch_size):
        stop = start + settings.batch_size
        vector_store.add_documents(chunks[start:stop], ids=identifiers[start:stop])
    return len(documents), len(chunks)


def _run_builder_in_subprocess(
    builder: BuildIndex, settings: Settings, destination: Path
) -> tuple[int, int]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_builder_worker,
        args=(builder, settings, destination, sender),
    )
    process.start()
    sender.close()
    try:
        try:
            succeeded, payload = receiver.recv()
        except EOFError as error:
            process.join()
            raise IngestionError(
                f"Index builder exited with code {process.exitcode} without a result."
            ) from error
        process.join()
        if succeeded:
            return payload
        raise IngestionError(payload)
    finally:
        receiver.close()


def _builder_worker(builder, settings: Settings, destination: Path, sender) -> None:
    try:
        sender.send((True, builder(settings, destination)))
    except BaseException:
        sender.send((False, traceback.format_exc()))
    finally:
        sender.close()


def _promote_index(staging: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup-{uuid4().hex}")
    had_active_index = destination.exists()
    if had_active_index:
        destination.replace(backup)
    try:
        staging.replace(destination)
    except Exception:
        if had_active_index and backup.exists():
            backup.replace(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


if __name__ == "__main__":
    main()
