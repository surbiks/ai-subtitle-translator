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


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class ChunkConfig:
    max_lines: int = _env_int("CHUNK_MAX_LINES", 18)
    max_chars: int = _env_int("CHUNK_MAX_CHARS", 1500)
    time_gap_threshold_ms: int = _env_int("CHUNK_TIME_GAP_MS", 2500)
    context_lines: int = _env_int("CHUNK_CONTEXT_LINES", 3)


@dataclass
class TranslatorConfig:
    provider: str = _env("PROVIDER", "openai")  # type: ignore[assignment]  # "openai" or "anthropic"
    target_language: str = _env("TARGET_LANGUAGE", "Persian (Farsi)")  # type: ignore[assignment]
    model: str = _env("OPENAI_MODEL", "gpt-4o-mini")  # type: ignore[assignment]
    api_key: str | None = _env("OPENAI_API_KEY")
    base_url: str | None = _env("OPENAI_BASE_URL")
    anthropic_api_key: str | None = _env("ANTHROPIC_API_KEY")
    anthropic_base_url: str | None = _env("ANTHROPIC_BASE_URL")
    anthropic_model: str = _env("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")  # type: ignore[assignment]
    anthropic_temperature: float = _env_float("ANTHROPIC_TEMPERATURE", 0.3)
    max_concurrency: int = _env_int("MAX_CONCURRENCY", 5)
    max_retries: int = _env_int("MAX_RETRIES", 3)
    retry_base_delay: float = _env_float("RETRY_BASE_DELAY", 1.0)
    temperature: float = _env_float("OPENAI_TEMPERATURE", 0.3)
    enable_refinement: bool = _env_bool("ENABLE_REFINEMENT", False)
    enable_postprocess: bool = _env_bool("ENABLE_POSTPROCESS", True)
    glossary_path: str | None = _env("GLOSSARY_PATH")
    cache_path: str | None = _env("CACHE_PATH")


@dataclass
class AppConfig:
    chunk: ChunkConfig
    translator: TranslatorConfig

    @classmethod
    def default(cls) -> "AppConfig":
        return cls(chunk=ChunkConfig(), translator=TranslatorConfig())
