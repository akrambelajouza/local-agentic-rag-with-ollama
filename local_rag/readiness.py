"""Actionable readiness checks shared by the CLI and Streamlit app."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from urllib.error import URLError
from urllib.request import urlopen

from local_rag.config import ConfigurationError, Settings, load_settings


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    key: str
    label: str
    ok: bool
    detail: str
    action: str = ""


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    checks: tuple[ReadinessCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.ok for check in self.checks)

    @classmethod
    def configuration_failure(cls, detail: str) -> ReadinessReport:
        configuration = ReadinessCheck(
            "configuration",
            "Configuration",
            False,
            detail,
            "Copy .env.example to .env and fill in every required value.",
        )
        blocked = tuple(
            ReadinessCheck(
                key,
                label,
                False,
                "Not checked because configuration is invalid.",
                "Fix configuration, then run `python -m local_rag.readiness` again.",
            )
            for key, label in (
                ("dataset", "Dataset"),
                ("ollama", "Ollama"),
                ("models", "Models"),
                ("collection", "Vector collection"),
            )
        )
        return cls((configuration, *blocked))


ModelProbe = Callable[[str], set[str]]
CollectionProbe = Callable[[Path, str], int]


def assess_readiness(env_file: str | Path = ".env") -> ReadinessReport:
    try:
        settings = load_settings(env_file)
    except (ConfigurationError, OSError) as error:
        return ReadinessReport.configuration_failure(str(error))
    return check_readiness(settings)


def check_readiness(
    settings: Settings,
    *,
    list_models: ModelProbe | None = None,
    collection_count: CollectionProbe | None = None,
) -> ReadinessReport:
    model_probe = list_models or _list_ollama_models
    collection_probe = collection_count or _collection_count
    checks: list[ReadinessCheck] = [
        ReadinessCheck(
            "configuration", "Configuration", True, "Required settings loaded."
        ),
        _check_dataset(settings.dataset_path),
    ]
    try:
        available_models = model_probe(settings.ollama_base_url)
    except (ConnectionError, OSError, TimeoutError, URLError, ValueError) as error:
        checks.extend(
            (
                ReadinessCheck(
                    "ollama",
                    "Ollama",
                    False,
                    f"Cannot reach {settings.ollama_base_url}: {error}",
                    "Start Ollama with `ollama serve` and verify OLLAMA_BASE_URL.",
                ),
                ReadinessCheck(
                    "models",
                    "Models",
                    False,
                    "Cannot inspect models while Ollama is unavailable.",
                    "Start Ollama, then pull the CHAT_MODEL and EMBEDDING_MODEL values.",
                ),
            )
        )
        checks.append(_check_collection(settings, collection_probe))
        return ReadinessReport(tuple(checks))

    checks.append(ReadinessCheck("ollama", "Ollama", True, "Ollama is reachable."))
    missing_models = [
        model
        for model in (settings.chat_model, settings.embedding_model)
        if not _model_is_available(model, available_models)
    ]
    checks.append(
        ReadinessCheck(
            "models",
            "Models",
            not missing_models,
            "Configured chat and embedding models are available."
            if not missing_models
            else f"Missing models: {', '.join(missing_models)}",
            ""
            if not missing_models
            else "Run "
            + " and ".join(f"`ollama pull {model}`" for model in missing_models)
            + ".",
        )
    )
    checks.append(_check_collection(settings, collection_probe))
    return ReadinessReport(tuple(checks))


def render_cli_report(report: ReadinessReport, output: TextIO = sys.stdout) -> int:
    for check in report.checks:
        marker = "PASS" if check.ok else "FAIL"
        print(f"[{marker}] {check.label}: {check.detail}", file=output)
        if not check.ok and check.action:
            print(f"       ACTION: {check.action}", file=output)
    print(
        "\nReady." if report.ready else "\nNot ready; complete the actions above.",
        file=output,
    )
    return 0 if report.ready else 1


def main() -> None:
    raise SystemExit(render_cli_report(assess_readiness()))


def _check_dataset(dataset_path: Path) -> ReadinessCheck:
    if dataset_path.is_file() and dataset_path.stat().st_size > 0:
        return ReadinessCheck("dataset", "Dataset", True, f"Found {dataset_path}.")
    return ReadinessCheck(
        "dataset",
        "Dataset",
        False,
        f"No non-empty dataset found at {dataset_path}.",
        "Add a JSONL corpus at that path or update DATASET_STORAGE_FOLDER.",
    )


def _check_collection(settings: Settings, probe: CollectionProbe) -> ReadinessCheck:
    if not settings.database_location.exists():
        count = 0
    else:
        try:
            count = probe(settings.database_location, settings.collection_name)
        except Exception as error:
            return ReadinessCheck(
                "collection",
                "Vector collection",
                False,
                f"Cannot inspect collection {settings.collection_name!r}: {error}",
                "Run `python -m local_rag.ingestion` to create a fresh collection.",
            )
    if count > 0:
        return ReadinessCheck(
            "collection",
            "Vector collection",
            True,
            f"Collection contains {count} chunks.",
        )
    return ReadinessCheck(
        "collection",
        "Vector collection",
        False,
        f"Collection {settings.collection_name!r} is missing or empty.",
        "Run `python -m local_rag.ingestion` to build the collection.",
    )


def _list_ollama_models(base_url: str) -> set[str]:
    endpoint = f"{base_url.rstrip('/')}/api/tags"
    with urlopen(endpoint, timeout=3) as response:
        payload = json.load(response)
    return {
        str(model.get("name") or model.get("model"))
        for model in payload.get("models", ())
        if model.get("name") or model.get("model")
    }


def _collection_count(database_location: Path, collection_name: str) -> int:
    database_file = database_location / "chroma.sqlite3"
    if not database_file.is_file():
        return 0
    read_only_uri = f"{database_file.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(read_only_uri, uri=True)
    try:
        row = connection.execute(
            """
            SELECT COUNT(*)
            FROM embeddings AS embedding
            JOIN segments AS segment ON segment.id = embedding.segment_id
            JOIN collections AS collection ON collection.id = segment.collection
            WHERE collection.name = ?
            """,
            (collection_name,),
        ).fetchone()
    finally:
        connection.close()
    return int(row[0]) if row else 0


def _model_is_available(configured: str, available: set[str]) -> bool:
    return configured in available or (
        ":" not in configured and f"{configured}:latest" in available
    )


if __name__ == "__main__":
    main()
