"""Multi-provider routing with rate-limit / quota awareness.

Lets the user define several AI backends (e.g. two accounts, each a different
``copilot``/``codex`` provider + model + key) in a JSON file and route requests
across them. A request switches to the next provider when the current one hits a
rate-limit/quota error **or** a user-configured per-minute / per-day cap.

``RoutingProvider`` implements the same ``chat(system, messages, model,
temperature)`` interface as the single backends in ``translator.py``, so it is a
drop-in replacement — every existing call site (translator, memory, discover)
works unchanged. Usage/limit state is kept in memory for the lifetime of one run
and reset on the next process start, which matches real daily-quota rollover.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ai_subtitle_translator import translator as _t

logger = logging.getLogger(__name__)

STRATEGIES = ("failover", "round_robin")
PROVIDER_TYPES = ("copilot", "codex")

_DEFAULT_CONCURRENCY = 5
_DEFAULT_COOLDOWN = 60.0
# Hard cap on how long a single chat() call will block waiting for a cooling
# provider before giving up (the chunk then fails and stays resumable).
_MAX_WAIT_SECONDS = 60.0

# Substrings (lowercased) that mark a rate-limit / quota error from either
# backend. The codex backend raises ``RuntimeError("codex HTTP 429: ...")``.
_RATE_LIMIT_MARKERS = (
    "http 429",
    "rate limit",
    "rate_limit",
    "quota",
    "too many requests",
    "insufficient_quota",
)


# -- Config model & loader --


@dataclass
class ProviderSpec:
    name: str
    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    api_mode: str
    send_temperature: bool
    requests_per_minute: int  # 0 = unlimited
    requests_per_day: int  # 0 = unlimited
    concurrency: int
    cooldown_seconds: float


@dataclass
class ProvidersFile:
    strategy: str
    specs: list[ProviderSpec]


def _clean(value: Any) -> str | None:
    """Return a non-empty string or None."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def _limit_int(limits: dict[str, Any], key: str, default: int, *, minimum: int, name: str) -> int:
    raw = limits.get(key, default)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool) or int(raw) != raw:
        raise ValueError(f"providers file: '{name}'.limits.{key} must be an integer")
    value = int(raw)
    if value < minimum:
        raise ValueError(f"providers file: '{name}'.limits.{key} must be >= {minimum}")
    return value


def load_providers_file(path: str | Path) -> ProvidersFile:
    """Parse and validate a providers JSON file.

    Applies env fallbacks (``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``) and the
    codex/copilot ``send_temperature`` defaults that mirror ``_build_provider``.
    Raises ``ValueError`` (prefixed ``providers file:``) on any problem.
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"providers file: not found: {p}") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"providers file: could not read {p}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("providers file: top level must be a JSON object")

    strategy = data.get("strategy", "failover")
    if strategy not in STRATEGIES:
        raise ValueError(f"providers file: strategy must be one of {STRATEGIES}")

    raw_providers = data.get("providers")
    if not isinstance(raw_providers, list) or not raw_providers:
        raise ValueError("providers file: 'providers' must be a non-empty array")

    specs: list[ProviderSpec] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw_providers):
        if not isinstance(entry, dict):
            raise ValueError(f"providers file: providers[{i}] must be an object")

        name = _clean(entry.get("name"))
        if not name:
            raise ValueError(f"providers file: providers[{i}] is missing 'name'")
        if name in seen:
            raise ValueError(f"providers file: duplicate provider name '{name}'")
        seen.add(name)

        provider = entry.get("provider", "copilot")
        if provider not in PROVIDER_TYPES:
            raise ValueError(f"providers file: '{name}'.provider must be one of {PROVIDER_TYPES}")

        model = _clean(entry.get("model"))
        if not model:
            raise ValueError(f"providers file: '{name}' is missing 'model'")

        send_temperature = entry.get("send_temperature")
        if send_temperature is None:
            send_temperature = provider != "codex"  # copilot True, codex False
        if not isinstance(send_temperature, bool):
            raise ValueError(f"providers file: '{name}'.send_temperature must be a boolean")

        limits = entry.get("limits", {})
        if not isinstance(limits, dict):
            raise ValueError(f"providers file: '{name}'.limits must be an object")

        cooldown = limits.get("cooldown_seconds", _DEFAULT_COOLDOWN)
        if not isinstance(cooldown, (int, float)) or isinstance(cooldown, bool) or cooldown < 0:
            raise ValueError(f"providers file: '{name}'.limits.cooldown_seconds must be a number >= 0")

        specs.append(
            ProviderSpec(
                name=name,
                provider=provider,
                model=model,
                api_key=_clean(entry.get("api_key")) or os.getenv("OPENAI_API_KEY"),
                base_url=_clean(entry.get("base_url")) or os.getenv("OPENAI_BASE_URL"),
                api_mode=entry.get("api_mode", "auto"),
                send_temperature=send_temperature,
                requests_per_minute=_limit_int(limits, "requests_per_minute", 0, minimum=0, name=name),
                requests_per_day=_limit_int(limits, "requests_per_day", 0, minimum=0, name=name),
                concurrency=_limit_int(limits, "concurrency", _DEFAULT_CONCURRENCY, minimum=1, name=name),
                cooldown_seconds=float(cooldown),
            )
        )

    return ProvidersFile(strategy=strategy, specs=specs)


# -- Error classification --


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True when ``exc`` is a rate-limit / quota error from either backend.

    Handles the openai SDK (``RateLimitError`` / HTTP 429) and the codex
    backend's ``RuntimeError("codex HTTP 429: ...")``. Never raises. Excludes
    ``ModelNotSupportedError`` (an endpoint-capability error, not throttling).
    """
    if isinstance(exc, _t.ModelNotSupportedError):
        return False
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return True
    except Exception:
        pass
    if getattr(exc, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Best-effort Retry-After (seconds) from an openai error's response headers."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
    except Exception:
        return None
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None  # HTTP-date form not supported; fall back to configured cooldown


# -- Runtime state & router --


def _build_backend(spec: ProviderSpec):
    """Construct the concrete backend for a spec, reusing translator's classes."""
    if spec.provider == "codex":
        return _t._CodexProvider(
            api_key=spec.api_key,
            base_url=spec.base_url,
            send_temperature=spec.send_temperature,
        )
    return _t._OpenAIProvider(
        api_key=spec.api_key,
        base_url=spec.base_url,
        api_mode=spec.api_mode,
        send_temperature=spec.send_temperature,
    )


@dataclass
class _ProviderRuntime:
    """In-memory availability/usage state for one provider (one run)."""

    spec: ProviderSpec
    backend: Any
    semaphore: asyncio.Semaphore
    request_times: deque[float] = field(default_factory=deque)  # monotonic, rolling 60s
    daily_count: int = 0
    total_count: int = 0
    cooldown_until: float = 0.0

    def _trim(self, now: float) -> None:
        while self.request_times and now - self.request_times[0] >= 60.0:
            self.request_times.popleft()

    def _at_daily_cap(self) -> bool:
        return bool(self.spec.requests_per_day) and self.daily_count >= self.spec.requests_per_day

    def is_available(self, now: float) -> bool:
        if now < self.cooldown_until:
            return False
        if self._at_daily_cap():
            return False
        if self.spec.requests_per_minute:
            self._trim(now)
            if len(self.request_times) >= self.spec.requests_per_minute:
                return False
        return True

    def next_available_at(self, now: float) -> float:
        """Earliest monotonic time this provider could serve again (inf if never)."""
        if self._at_daily_cap():
            return math.inf  # not recoverable within this run
        candidates: list[float] = []
        if self.cooldown_until > now:
            candidates.append(self.cooldown_until)
        if self.spec.requests_per_minute:
            self._trim(now)
            if len(self.request_times) >= self.spec.requests_per_minute and self.request_times:
                candidates.append(self.request_times[0] + 60.0)
        return min(candidates) if candidates else now

    def record_request(self, now: float) -> None:
        if self.spec.requests_per_minute:
            self.request_times.append(now)
        self.daily_count += 1
        self.total_count += 1

    def start_cooldown(self, now: float, retry_after: float | None = None) -> None:
        delay = retry_after if (retry_after and retry_after > 0) else self.spec.cooldown_seconds
        self.cooldown_until = max(self.cooldown_until, now + delay)


class RoutingProvider:
    """Routes ``chat`` calls across multiple backends with failover + caps.

    Drop-in for the single-backend ``_ChatProvider``: the ``model`` argument is
    ignored (each backend uses its own ``spec.model``) and ``temperature`` is
    forwarded as-is (each backend's own ``send_temperature`` decides the wire).
    """

    def __init__(
        self,
        providers_file: ProvidersFile,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._strategy = providers_file.strategy
        self._runtimes = [
            _ProviderRuntime(
                spec=spec,
                backend=_build_backend(spec),
                semaphore=asyncio.Semaphore(spec.concurrency),
            )
            for spec in providers_file.specs
        ]
        self._lock = asyncio.Lock()
        self._rr_cursor = 0
        self._now = time_fn or time.monotonic
        logger.info(
            "Routing across %d providers (strategy=%s): %s",
            len(self._runtimes), self._strategy,
            ", ".join(f"{rt.spec.name}->{rt.spec.model}" for rt in self._runtimes),
        )

    async def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        # Two passes: try every currently-eligible provider; if none are
        # eligible, wait once (bounded) for the soonest to recover, then retry.
        for attempt in range(2):
            while True:
                runtime = await self._select_and_reserve(self._now())
                if runtime is None:
                    break
                try:
                    async with runtime.semaphore:
                        return await runtime.backend.chat(
                            system=system,
                            messages=messages,
                            model=runtime.spec.model,
                            temperature=temperature,
                        )
                except Exception as exc:
                    if not _is_rate_limit_error(exc):
                        raise  # transient/other errors are the caller's retry job
                    retry_after = _retry_after_seconds(exc)
                    runtime.start_cooldown(self._now(), retry_after)
                    logger.info(
                        "Provider '%s' rate-limited%s — switching providers",
                        runtime.spec.name,
                        " (Retry-After)" if retry_after else "",
                    )
                    continue  # try the next eligible provider

            if attempt == 0:
                wait = self._soonest_wait(self._now())
                if wait is None:
                    break  # everything is at its daily cap — no point waiting
                logger.warning(
                    "All providers busy/cooling — waiting %.1fs before retry", wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError("all providers exhausted (rate-limited or over cap)")

    async def _select_and_reserve(self, now: float) -> _ProviderRuntime | None:
        """Pick the next eligible provider per strategy and reserve a request.

        The critical section holds no ``await`` (selection + counter updates are
        synchronous), so it is atomic under asyncio's cooperative scheduling; the
        backend network call happens outside the lock to preserve parallelism.
        """
        async with self._lock:
            n = len(self._runtimes)
            for offset in range(n):
                idx = (self._rr_cursor + offset) % n if self._strategy == "round_robin" else offset
                runtime = self._runtimes[idx]
                if runtime.is_available(now):
                    runtime.record_request(now)
                    if self._strategy == "round_robin":
                        self._rr_cursor = (idx + 1) % n
                    return runtime
            return None

    def _soonest_wait(self, now: float) -> float | None:
        """Bounded seconds until the soonest provider recovers, or None if all
        are permanently exhausted (daily cap) for this run."""
        finite = [t for t in (rt.next_available_at(now) for rt in self._runtimes) if t != math.inf]
        if not finite:
            return None
        return min(max(0.0, min(finite) - now), _MAX_WAIT_SECONDS)

    def usage_report(self) -> dict[str, int]:
        """Per-provider request counts for this run (for end-of-run logging)."""
        return {rt.spec.name: rt.total_count for rt in self._runtimes}
