"""Application settings loaded from environment variables and .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = _project_path(os.getenv("DATA_DIR", "data"))
    chroma_dir: Path = _project_path(os.getenv("CHROMA_DIR", "chroma_db"))
    collection_name: str = os.getenv("COLLECTION_NAME", "rag_documents")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))


settings = Settings()