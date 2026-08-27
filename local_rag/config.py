"""Application configuration loaded without creating runtime resources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import environ
from pathlib import Path

from dotenv import dotenv_values


class ConfigurationError(ValueError):
    """Raised when required application settings are missing or invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings shared by ingestion and chat entry points."""

    embedding_model: str
    chat_model: str
    model_provider: str
    dataset_path: Path
    database_location: Path
    collection_name: str
    ollama_base_url: str = "http://localhost:11434"

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, str | None],
        *,
        base_directory: Path | None = None,
    ) -> Settings:
        required_names = (
            "CHAT_MODEL",
            "COLLECTION_NAME",
            "DATABASE_LOCATION",
            "DATASET_STORAGE_FOLDER",
            "EMBEDDING_MODEL",
            "MODEL_PROVIDER",
        )
        missing = sorted(name for name in required_names if not _clean(values.get(name)))
        if missing:
            raise ConfigurationError(
                f"Missing required settings: {', '.join(missing)}"
            )

        root = base_directory or Path.cwd()
        dataset_folder = Path(_required(values, "DATASET_STORAGE_FOLDER"))
        database_location = Path(_required(values, "DATABASE_LOCATION"))

        return cls(
            embedding_model=_required(values, "EMBEDDING_MODEL"),
            chat_model=_required(values, "CHAT_MODEL"),
            model_provider=_required(values, "MODEL_PROVIDER"),
            dataset_path=_resolve(root, dataset_folder / "data.txt"),
            database_location=_resolve(root, database_location),
            collection_name=_required(values, "COLLECTION_NAME"),
            ollama_base_url=_clean(values.get("OLLAMA_BASE_URL"))
            or "http://localhost:11434",
        )


def load_settings(env_file: str | Path = ".env") -> Settings:
    """Load settings from a dotenv file, overridden by the process environment."""

    env_path = Path(env_file)
    values: dict[str, str | None] = dict(dotenv_values(env_path))
    values.update(environ)
    base_directory = env_path.parent if env_path.parent != Path("") else Path.cwd()
    return Settings.from_mapping(values, base_directory=base_directory)


def _clean(value: str | None) -> str:
    return value.strip() if value else ""


def _required(values: Mapping[str, str | None], name: str) -> str:
    return _clean(values.get(name))


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path
