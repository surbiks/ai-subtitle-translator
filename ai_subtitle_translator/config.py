import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str | None = None) -> str | None:
    """Read an env var, returning None if empty or missing."""
    val = os.getenv(key, default)
    return val if val else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val else default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val else default


@dataclass
class ChunkConfig:
    max_lines: int = _env_int("CHUNK_MAX_LINES", 18)
    max_chars: int = _env_int("CHUNK_MAX_CHARS", 1500)
    time_gap_threshold_ms: int = _env_int("CHUNK_TIME_GAP_MS", 2500)
    overlap_lines: int = _env_int("CHUNK_OVERLAP_LINES", 2)


@dataclass
class TranslatorConfig:
    model: str = _env("OPENAI_MODEL", "gpt-4o-mini")  # type: ignore[assignment]
    api_key: str | None = _env("OPENAI_API_KEY")
    base_url: str | None = _env("OPENAI_BASE_URL")
    max_concurrency: int = _env_int("MAX_CONCURRENCY", 5)
    max_retries: int = _env_int("MAX_RETRIES", 3)
    retry_base_delay: float = _env_float("RETRY_BASE_DELAY", 1.0)
    temperature: float = _env_float("OPENAI_TEMPERATURE", 0.3)


@dataclass
class AppConfig:
    chunk: ChunkConfig
    translator: TranslatorConfig

    @classmethod
    def default(cls) -> "AppConfig":
        return cls(chunk=ChunkConfig(), translator=TranslatorConfig())
