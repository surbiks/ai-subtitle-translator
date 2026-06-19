"""
Async subtitle translator using the OpenAI API.

Features:
- Context-aware prompting (previous chunk as read-only context)
- Glossary injection
- Subtitle compression guidance
- Correction retry on invalid JSON
- Optional refinement pass
- Cache integration
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import httpx

from dataclasses import dataclass
from typing import Any, Protocol
from ai_subtitle_translator.ass_tags import has_tags, restore_tags, strip_tags
from collections.abc import Callable, Iterable

from ai_subtitle_translator.cache import TranslationCache
from ai_subtitle_translator.classify import LineKind, classify
from ai_subtitle_translator.config import TranslatorConfig
from ai_subtitle_translator.discover import extract_candidates, propose_translations
from ai_subtitle_translator.glossary import Glossary
from ai_subtitle_translator.memory import StoryMemory
from ai_subtitle_translator.parser import Subtitle
from ai_subtitle_translator.postprocess import postprocess_persian

logger = logging.getLogger(__name__)

# -- Prompt builders --


def _is_persian(language: str) -> bool:
    """True when the target language is Persian/Farsi."""
    lang = language.lower()
    return "persian" in lang or "farsi" in lang


_PERSIAN_STYLE_BLOCK = """

PERSIAN (FARSI) STYLE
- Use everyday spoken Iranian Persian (محاوره‌ای), not written/literary forms:
  می‌خوام، نمی‌دونم، بریم، چی‌کار، آره — not می‌خواهم، نمی‌دانم، برویم.
- Pick «تو» vs «شما» from the speakers' relationship and stay consistent.
- Prefer common Persian words over heavy formal Arabic-loan vocabulary.
- Use Persian punctuation (؟ ، ؛) and «…» for quotes; use the half-space (ZWNJ)
  correctly: می‌خوام، کتاب‌ها، خونه‌ام.
- Make it sound like real Persian movie subtitles: short, casual, natural."""


def _build_system_prompt(
    language: str,
    glossary: Glossary | None = None,
    film_context: str = "",
) -> str:
    glossary_section = glossary.build_prompt_section() if glossary and not glossary.is_empty else ""
    persian_section = _PERSIAN_STYLE_BLOCK if _is_persian(language) else ""

    return f"""You are a professional subtitle translator. Translate the subtitles into natural,
idiomatic, conversational {language} as spoken in real films and TV.

TRANSLATION QUALITY
- Translate meaning, not words. Localize idioms, jokes, and slang to their natural
  {language} equivalent. Never translate literally.
- Match each line's tone and register (casual, formal, angry, tender) and keep a
  given speaker's register consistent across the scene.
- Keep lines short and readable at subtitle speed; prefer phrasing as short as or
  shorter than the source.
- Translate faithfully: do not censor, soften, summarize, add, or omit content.

KEEP VERBATIM (do not translate or alter)
- Markup: HTML-like tags such as <i>...</i>, ASS tags like {{\\an8}}, and music notes ♪.
- Proper nouns, brand names, and numbers — unless the glossary says otherwise.

STRUCTURE (critical)
- Input is a JSON array of objects with "id" and "text".
- Return a JSON array with EXACTLY the same number of objects and the SAME "id" values.
- One input item = one output item. Never merge or split items, even when a sentence
  spans several items — translate each so it reads correctly in its own position.
- A "Previous context" block may be supplied for continuity only — never translate
  or include those lines.

OUTPUT
- Output ONLY the JSON array. No explanations, no markdown, no code fences.{persian_section}{glossary_section}"""


def _build_user_message(
    payload: list[dict[str, Any]],
    context: list[Subtitle] | None = None,
    memory_block: str = "",
    retry_note: str = "",
) -> str:
    """Build user message with optional rolling-summary and previous-chunk context."""
    parts: list[str] = []

    if retry_note:
        parts.append(retry_note)

    if memory_block:
        parts.append(memory_block)

    if context:
        ctx_lines = [f'  - [{s.id}] "{s.text}"' for s in context]
        parts.append(
            "Previous context (for continuity only, do NOT translate these):\n"
            + "\n".join(ctx_lines)
            + "\n"
        )

    parts.append("Translate the following:\n" + json.dumps(payload, ensure_ascii=False))
    return "\n".join(parts)


def _build_refinement_prompt(language: str) -> str:
    """Second-pass editor prompt, specialized for the target language."""
    return f"""You are a {language} subtitle editor. Improve this translated subtitle text:
- Make it more natural and conversational
- Fix any awkward phrasing
- Keep it concise for subtitle readability
- Do NOT change the JSON structure or the "id" values

Input and output: a JSON array of objects with "id" (int) and "text" (string).
Return ONLY the improved JSON array."""

_CORRECTION_PROMPT = (
    "Your previous output was not valid JSON. "
    "Return ONLY a valid JSON array of objects with \"id\" (int) and \"text\" (string) fields. "
    "No markdown, no explanation, just the JSON array."
)

# Prepended to the user message when retrying chunks that failed a prior run,
# so the model knows it must produce a real translation (not echo the source).
_RETRY_NOTE = (
    "These lines failed a previous translation attempt. Translate every item "
    "fully into the target language now. Never return the original source text "
    "as the translation.\n"
)


# -- Provider abstraction --


class _ChatProvider(Protocol):
    """Minimal interface for an LLM chat call."""

    async def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str: ...


class ModelNotSupportedError(Exception):
    """Raised when a model doesn't support the chat completions endpoint."""


def _is_unsupported_model_error(exc: Exception) -> bool:
    """Check if an API error indicates the model doesn't support the endpoint."""
    err = getattr(exc, "body", None) or getattr(exc, "response", None)
    if isinstance(exc, dict):
        err = exc
    if isinstance(err, dict):
        code = str(err.get("code", ""))
        message = str(err.get("message", ""))
    elif isinstance(err, str):
        code = ""
        message = err
    else:
        code = str(getattr(err, "code", "") or "")
        message = str(getattr(err, "message", "") or "") + str(exc)

    return "unsupported_api_for_model" in code or "not accessible via" in message


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    """Check if an API error indicates the model rejects an explicit temperature."""
    msg = str(exc).lower()
    if "temperature" not in msg:
        return False
    return (
        "unsupported" in msg
        or "does not support" in msg
        or "only the default" in msg
    )


def _build_responses_input(
    system: str, messages: list[dict[str, str]]
) -> str:
    """Convert chat messages to a single input string for the Responses API."""
    parts: list[str] = []
    if system:
        parts.append(f"System: {system}")
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


class _OpenAIProvider:
    """OpenAI provider supporting both chat.completions and the Responses API.

    The API surface is chosen by ``api_mode``:
      - "chat":      always use chat.completions (e.g. gpt-5-mini)
      - "responses": always use the Responses API (e.g. gpt-5.4-mini)
      - "auto":      try chat.completions, fall back to Responses when the model
                     isn't supported there (default)

    ``send_temperature`` controls whether the temperature parameter is sent at
    all — some models reject any explicit temperature, so set it to False for
    them. As a safety net, a temperature-rejection error also triggers a
    one-time retry without it, so a misconfigured setting degrades gracefully.
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        api_mode: str = "auto",
        send_temperature: bool = True,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._api_mode = api_mode
        self._send_temperature = send_temperature
        self._use_responses: dict[str, bool] = {}  # model → True once responses is required

    async def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        temp = temperature if self._send_temperature else None

        if self._api_mode == "responses" or self._use_responses.get(model):
            return await self._chat_via_responses(system, messages, model, temp)

        try:
            return await self._chat_via_completions(system, messages, model, temp)
        except Exception as exc:
            # In "chat" mode the endpoint is pinned by the user — don't fall back.
            if self._api_mode != "chat" and _is_unsupported_model_error(exc):
                logger.info(
                    "Model '%s' unsupported on /chat/completions, falling back to /responses",
                    model,
                )
                self._use_responses[model] = True
                return await self._chat_via_responses(system, messages, model, temp)
            raise

    async def _chat_via_completions(
        self,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None,
    ) -> str:
        api_messages = [{"role": "system", "content": system}, *messages]
        try:
            return await self._do_completions_call(api_messages, model, temperature)
        except Exception as exc:
            if temperature is not None and _is_unsupported_temperature_error(exc):
                logger.debug("Temperature rejected for '%s' on /chat/completions, retrying without", model)
                return await self._do_completions_call(api_messages, model, None)
            raise

    async def _do_completions_call(
        self,
        api_messages: list[dict[str, str]],
        model: str,
        temperature: float | None,
    ) -> str:
        kwargs: dict[str, Any] = {"model": model, "messages": api_messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        return response.choices[0].message.content or ""

    async def _chat_via_responses(
        self,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None,
    ) -> str:
        """Use the OpenAI Responses API for models that don't support chat.completions."""
        try:
            return await self._do_responses_call(system, messages, model, temperature)
        except Exception as exc:
            if temperature is not None and _is_unsupported_temperature_error(exc):
                logger.debug("Temperature rejected for '%s' on /responses, retrying without", model)
                return await self._do_responses_call(system, messages, model, None)
            raise

    async def _do_responses_call(
        self,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": system or None,
            "input": _build_responses_input("", messages),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await self._client.responses.create(**kwargs)
        # Responses API returns output as a list of content blocks
        text_parts: list[str] = []
        for item in response.output:
            if hasattr(item, "content") and item.content:
                for block in item.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
        return "\n".join(text_parts)


def _to_codex_input(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert chat messages to the Responses API list-input shape.

    The system prompt is carried separately via ``instructions``, so it is not
    included here. Assistant turns (from the correction-retry flow) use
    ``output_text`` content; everything else uses ``input_text``.
    """
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content_type = "output_text" if role == "assistant" else "input_text"
        items.append(
            {
                "role": role,
                "content": [{"type": content_type, "text": msg.get("content", "")}],
            }
        )
    return items


def _extract_codex_text(lines: Iterable[str]) -> str:
    """Accumulate assistant text from codex SSE ``data:`` lines.

    Concatenates every ``response.output_text.delta`` fragment, falling back to
    the ``response.output_text.done`` text if no deltas were seen. Raises on
    error/failed events; unknown events (rate limits, lifecycle) are ignored.
    """
    parts: list[str] = []
    done_text: str | None = None

    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            evt = json.loads(data)
        except json.JSONDecodeError:
            continue

        etype = evt.get("type")
        if etype == "response.output_text.delta":
            parts.append(evt.get("delta", ""))
        elif etype == "response.output_text.done":
            done_text = evt.get("text")
        elif etype == "response.completed":
            break
        elif etype in ("response.failed", "error") or evt.get("error"):
            raise RuntimeError(f"codex stream error: {data[:300]}")

    return "".join(parts) or (done_text or "")


class _CodexProvider:
    """OpenAI-compatible proxy that speaks only the *streaming* Responses API.

    Requires ``stream=true`` and a list-shaped ``input``, and emits some
    non-standard SSE events (e.g. ``codex.rate_limits``) which we ignore.
    Returns the full assistant text once the stream completes, so the rest of
    the translator (JSON parsing, retry, cache) is unchanged.
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        send_temperature: bool = False,
    ) -> None:
        self._api_key = api_key or "dummy"
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._send_temperature = send_temperature

    async def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        body: dict[str, Any] = {
            "model": model,
            "stream": True,
            "instructions": system or None,
            "input": _to_codex_input(messages),
        }
        if self._send_temperature and temperature is not None:
            body["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/responses"

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")
                    raise RuntimeError(f"codex HTTP {resp.status_code}: {detail[:300]}")
                lines = [line async for line in resp.aiter_lines()]

        return _extract_codex_text(lines)


def _build_provider(config: TranslatorConfig) -> _ChatProvider:
    """Create the provider backend selected by ``config.provider``.

    When a providers file is configured, route across multiple backends instead
    of a single one. Imported lazily to avoid an import cycle.
    """
    if config.providers_path:
        from ai_subtitle_translator.providers import RoutingProvider, load_providers_file

        return RoutingProvider(load_providers_file(config.providers_path))
    if config.provider == "codex":
        return _CodexProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            send_temperature=config.send_temperature,
        )
    # "copilot" — chat.completions with non-streaming Responses fallback
    return _OpenAIProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        api_mode=config.api_mode,
        send_temperature=config.send_temperature,
    )


# -- Translator --


@dataclass
class ChunkOutcome:
    """Result of translating a single chunk.

    ``subtitles`` always holds the best text to use: the translation on success,
    or the original chunk on failure (so the final file is always complete).
    ``ok`` reports whether the translation actually succeeded and validated.
    """

    index: int
    ok: bool
    subtitles: list[Subtitle]
    error: str | None = None


class Translator:
    def __init__(
        self,
        config: TranslatorConfig | None = None,
        glossary: Glossary | None = None,
        cache: TranslationCache | None = None,
    ) -> None:
        self._cfg = config or TranslatorConfig()
        self._provider = _build_provider(self._cfg)
        self._semaphore = asyncio.Semaphore(self._cfg.max_concurrency)
        self._glossary = glossary
        self._cache = cache or TranslationCache()

        self._model = self._cfg.model
        self._temperature = self._cfg.temperature

        # Run-level state (Phase 2)
        self._memory: StoryMemory | None = (
            StoryMemory() if self._cfg.enable_memory else None
        )
        self._film_context: str = (self._cfg.register_override or "").strip()

        # Resume support: set when retrying chunks that failed a prior run, and
        # a guard so one-time run setup (glossary discovery / register probe)
        # runs at most once even if multiple entry points are called.
        self._retry_mode = False
        self._prepared = False

        self._system_prompt = _build_system_prompt(
            self._cfg.target_language, glossary, self._film_context,
        )
        self._refinement_prompt = _build_refinement_prompt(self._cfg.target_language)

    @property
    def cache(self) -> TranslationCache:
        return self._cache

    async def translate_chunks(
        self,
        chunks: list[list[Subtitle]],
        contexts: list[list[Subtitle] | None] | None = None,
    ) -> list[list[Subtitle]]:
        """Translate all chunks, falling back to the original text for any chunk
        that fails. Kept for backward compatibility; use translate_chunks_detailed
        for per-chunk success reporting and resume support."""
        outcomes = await self.translate_chunks_detailed(chunks, contexts)
        by_index = {o.index: o for o in outcomes}
        return [by_index[i].subtitles for i in range(len(chunks))]

    async def translate_chunks_detailed(
        self,
        chunks: list[list[Subtitle]],
        contexts: list[list[Subtitle] | None] | None = None,
        targets: Iterable[int] | None = None,
        progress_callback: Callable[[ChunkOutcome], None] | None = None,
        retry_mode: bool = False,
    ) -> list[ChunkOutcome]:
        """Translate chunks and report per-chunk outcomes.

        ``targets`` restricts work to specific chunk indices (resume retries only
        failed/pending chunks); None means all chunks. ``retry_mode`` tells the
        model these chunks failed before. When given, ``progress_callback`` runs
        synchronously as each chunk finishes so the caller can persist progress
        for crash safety. Returns one ChunkOutcome per translated index."""
        if contexts is None:
            contexts = [None] * len(chunks)

        self._retry_mode = retry_mode
        await self._prepare_run(chunks)

        if targets is None:
            target_indices = list(range(len(chunks)))
        else:
            target_indices = sorted(i for i in set(targets) if 0 <= i < len(chunks))

        # Rolling-memory batching only applies to a full, in-order run.
        if self._memory is not None and targets is None:
            outcomes = await self._translate_in_batches(chunks, contexts, progress_callback)
        else:
            tasks = [
                self._run_chunk(i, chunks[i], contexts[i], progress_callback=progress_callback)
                for i in target_indices
            ]
            outcomes = list(await asyncio.gather(*tasks))

        self._log_provider_usage()
        return outcomes

    def _log_provider_usage(self) -> None:
        """Log per-provider request counts when routing across multiple backends."""
        report = getattr(self._provider, "usage_report", None)
        if not callable(report):
            return
        counts = report()
        if counts:
            logger.info(
                "Provider usage: %s",
                ", ".join(f"{name}={n}" for name, n in counts.items()),
            )

    async def _prepare_run(self, chunks: list[list[Subtitle]]) -> None:
        """One-time per-run setup: auto-glossary discovery and register probe."""
        if self._prepared:
            return
        self._prepared = True

        # Auto-discover glossary terms from the source (Phase 3.1).
        if self._cfg.auto_glossary:
            await self._discover_glossary(chunks)

        # One-shot register probe (skipped if register_override was set or
        # auto_probe is off).
        if self._cfg.auto_probe and not self._film_context:
            sample = [s for chunk in chunks[:3] for s in chunk]
            probed = await self._probe_register(sample)
            if probed:
                self._film_context = probed
                self._system_prompt = _build_system_prompt(
                    self._cfg.target_language, self._glossary, self._film_context,
                )
                logger.info("Detected film register: %s", probed.replace("\n", " | "))

    async def _translate_in_batches(
        self,
        chunks: list[list[Subtitle]],
        contexts: list[list[Subtitle] | None],
        progress_callback: Callable[[ChunkOutcome], None] | None = None,
    ) -> list[ChunkOutcome]:
        """Batched-parallel: translate N chunks sharing the current summary,
        then fold that batch's source text into the summary before the next."""
        assert self._memory is not None
        batch_size = max(1, self._cfg.memory_update_interval)
        outcomes: list[ChunkOutcome] = []

        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            memory_block = self._memory.context_block()
            tasks = [
                self._run_chunk(
                    i, chunks[i], contexts[i],
                    memory_block=memory_block,
                    progress_callback=progress_callback,
                )
                for i in range(start, end)
            ]
            outcomes.extend(await asyncio.gather(*tasks))

            batch_subs = [s for i in range(start, end) for s in chunks[i]]
            await self._memory.update(
                self._provider, self._model, batch_subs, temperature=0.0,
            )

        return outcomes

    async def _run_chunk(
        self,
        index: int,
        chunk: list[Subtitle],
        context: list[Subtitle] | None,
        memory_block: str = "",
        progress_callback: Callable[[ChunkOutcome], None] | None = None,
    ) -> ChunkOutcome:
        """Translate one chunk into a ChunkOutcome, catching failures and
        validating the result. Never raises."""
        try:
            translated = await self._translate_chunk(
                index, chunk, context, memory_block=memory_block
            )
            error = _validate_translation(chunk, translated, self._should_translate)
            if error is None:
                outcome = ChunkOutcome(index=index, ok=True, subtitles=translated)
            else:
                logger.warning("Chunk %d failed validation: %s", index, error)
                outcome = ChunkOutcome(
                    index=index, ok=False, subtitles=list(chunk), error=error,
                )
        except Exception as exc:
            logger.error("Chunk %d failed permanently: %s", index, exc)
            outcome = ChunkOutcome(
                index=index, ok=False, subtitles=list(chunk), error=str(exc),
            )

        if progress_callback is not None:
            progress_callback(outcome)
        return outcome

    async def _discover_glossary(self, chunks: list[list[Subtitle]]) -> None:
        """Extract candidate terms from source text, ask the provider for
        translations, and merge non-conflicting entries into the active
        glossary. User entries always win."""
        all_subs = [s for chunk in chunks for s in chunk]
        candidates = extract_candidates(
            all_subs, self._cfg.auto_glossary_min_occurrences,
        )
        if not candidates:
            logger.info("Auto-glossary: no candidate terms found")
            return

        preview = ", ".join(candidates[:5])
        more = f", … (+{len(candidates) - 5})" if len(candidates) > 5 else ""
        logger.info("Auto-glossary: %d candidate terms (%s%s)", len(candidates), preview, more)

        proposed = await propose_translations(
            self._provider, self._model, self._cfg.target_language, candidates,
        )
        if not proposed:
            logger.info("Auto-glossary: no translations returned")
            return

        if self._glossary is None:
            self._glossary = Glossary()
        added = self._glossary.extend(proposed, user_wins=True)
        if added > 0:
            logger.info("Auto-glossary: added %d new entries", added)
            self._system_prompt = _build_system_prompt(
                self._cfg.target_language, self._glossary, self._film_context,
            )
        else:
            logger.info("Auto-glossary: all proposed terms already present")

    async def _enforce_glossary(
        self,
        index: int,
        chunk: list[Subtitle],
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """One targeted retry for translations that omit required glossary terms."""
        if self._glossary is None or self._glossary.is_empty:
            return items
        chunk_by_id = {s.id: s for s in chunk}

        violations: list[dict[str, Any]] = []
        for item in items:
            try:
                item_id = int(item["id"])
            except (KeyError, TypeError, ValueError):
                continue
            orig = chunk_by_id.get(item_id)
            text = item.get("text")
            if orig is None or not isinstance(text, str):
                continue
            missing = self._glossary.check_compliance(orig.text, text)
            if missing:
                violations.append({
                    "id": item_id,
                    "current": text,
                    "required_terms": [
                        {"term": t, "must_use": e} for t, e in missing
                    ],
                })

        if not violations:
            return items

        logger.info(
            "Chunk %d: %d translations missing glossary terms — requesting fix",
            index, len(violations),
        )
        fix_msg = (
            "These translations did not use the required glossary terms. "
            "Re-translate each item, using EVERY listed glossary term EXACTLY "
            "as given in 'must_use'. Preserve meaning otherwise and respect "
            'the same max_chars budget. Output ONLY a JSON array of '
            '{"id": int, "text": string}.\n\n'
            + json.dumps(violations, ensure_ascii=False)
        )
        try:
            content = await self._provider.chat(
                system=self._system_prompt,
                messages=[{"role": "user", "content": fix_msg}],
                model=self._model,
                temperature=self._temperature,
            )
            revised = _parse_response(content, len(violations))
        except Exception as exc:
            logger.warning(
                "Chunk %d: glossary retry failed (%s) — keeping non-compliant",
                index, exc,
            )
            return items

        revised_by_id: dict[int, str] = {}
        for r in revised:
            try:
                rid = int(r["id"])
            except (KeyError, TypeError, ValueError):
                continue
            rtext = r.get("text")
            if isinstance(rtext, str) and rtext.strip():
                revised_by_id[rid] = rtext

        merged: list[dict[str, Any]] = []
        for item in items:
            try:
                item_id = int(item["id"])
            except (KeyError, TypeError, ValueError):
                merged.append(item)
                continue
            if item_id in revised_by_id:
                merged.append({"id": item_id, "text": revised_by_id[item_id]})
            else:
                merged.append(item)
        return merged

    async def _probe_register(self, sample: list[Subtitle]) -> str:
        """Ask the model to summarize genre + register from a small sample."""
        sample_text = "\n".join(s.text for s in sample[:20] if s.text.strip())
        if not sample_text:
            return ""
        try:
            response = await self._provider.chat(
                system="You are a translation prep assistant.",
                messages=[{
                    "role": "user",
                    "content": f"{_PROBE_PROMPT}\n\nLines:\n{sample_text}",
                }],
                model=self._model,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning(
                "Register probe failed (%s) — proceeding without film context", exc,
            )
            return ""
        return response.strip()

    async def _translate_chunk(
        self,
        index: int,
        chunk: list[Subtitle],
        context: list[Subtitle] | None,
        memory_block: str = "",
    ) -> list[Subtitle]:
        """Translate a single chunk with semaphore, cache, retry, and post-processing."""
        async with self._semaphore:
            # Classify each line so we can route cues/lyrics differently and
            # honor the user's translate_cues / translate_lyrics flags.
            kinds = [classify(s.text) for s in chunk]
            translate_mask = [self._should_translate(k) for k in kinds]
            items_to_translate = [s for s, m in zip(chunk, translate_mask) if m]

            if not items_to_translate:
                logger.info("Chunk %d: no translatable lines, preserving all", index)
                return [self._copy_with_text(s, s.text) for s in chunk]

            # Cache is bypassed entirely when rolling memory is on: cached
            # translations don't account for the current summary state.
            use_cache = self._memory is None

            if use_cache:
                all_cached = all(self._cache.has(s.text) for s in items_to_translate)
                if all_cached:
                    logger.info("Chunk %d fully cached, skipping API call", index)
                    return self._build_from_cache(chunk, kinds, translate_mask)

            # Pre-strip ASS override tags so the model only sees visible text.
            # The tag map is kept per-id so we can restore tags at the end.
            ass_meta_by_id: dict[int, tuple[list[tuple[int, str]], int]] = {}
            text_overrides: dict[int, str] = {}
            for s in items_to_translate:
                if has_tags(s.text):
                    clean, tags = strip_tags(s.text)
                    ass_meta_by_id[s.id] = (tags, len(clean))
                    text_overrides[s.id] = clean

            payload = self._build_payload(
                chunk, kinds, translate_mask,
                text_overrides=text_overrides or None,
            )
            retry_note = _RETRY_NOTE if self._retry_mode else ""
            user_msg = _build_user_message(
                payload, context, memory_block=memory_block, retry_note=retry_note,
            )
            translated_items = await self._call_with_retry_and_correction(
                index, user_msg, len(payload)
            )

            # Refinement pass (optional)
            if self._cfg.enable_refinement:
                translated_items = await self._refine(index, translated_items)

            # One compression retry for over-budget translations.
            if self._cfg.enforce_cps:
                translated_items = await self._enforce_cps_budget(
                    index, chunk, translated_items
                )

            # One targeted retry for missing glossary terms (Phase 3.2).
            if self._cfg.enforce_glossary and self._glossary is not None and not self._glossary.is_empty:
                translated_items = await self._enforce_glossary(
                    index, chunk, translated_items
                )

            # Align translations back to the original chunk by id (not position).
            # The model may drop, duplicate, or reorder items; positional zip
            # would silently misalign translations to the wrong subtitle.
            by_id: dict[int, str] = {}
            for item in translated_items:
                try:
                    item_id = int(item["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    by_id[item_id] = text

            # Build result with post-processing
            result: list[Subtitle] = []
            for orig, will_translate in zip(chunk, translate_mask):
                if not will_translate:
                    # Cue/lyric the user opted out of translating — preserve source.
                    result.append(self._copy_with_text(orig, orig.text))
                    continue

                translated_text = by_id.get(orig.id)
                if translated_text is None:
                    # The model dropped this item — keep the source text.
                    result.append(self._copy_with_text(orig, orig.text))
                    continue

                # Apply Persian post-processing if target is Persian
                if _is_persian(self._cfg.target_language):
                    translated_text = postprocess_persian(translated_text)

                if "\n" in orig.text:
                    translated_text = _restore_multiline(translated_text, orig.text)

                meta = ass_meta_by_id.get(orig.id)
                if meta is not None:
                    tags, clean_len = meta
                    translated_text = restore_tags(translated_text, tags, clean_len)

                if use_cache:
                    self._cache.put(orig.text, translated_text)
                result.append(self._copy_with_text(orig, translated_text))

            return result

    # -- Per-chunk helpers --

    def _should_translate(self, kind: LineKind) -> bool:
        if not self._cfg.translate_cues and kind in (
            LineKind.SOUND_CUE, LineKind.STAGE_DIR,
        ):
            return False
        if not self._cfg.translate_lyrics and kind == LineKind.LYRICS:
            return False
        return True

    def _is_persian_target(self) -> bool:
        lang = self._cfg.target_language.lower()
        return "persian" in lang or "farsi" in lang

    def _char_budget(self, sub: Subtitle) -> int:
        """Per-line character budget derived from on-screen duration and CPS target."""
        duration_sec = max(0.5, (sub.end_ms - sub.start_ms) / 1000.0)
        budget = int(duration_sec * self._cfg.cps_target)
        return max(self._cfg.cps_min_chars, budget)

    def _build_payload(
        self,
        chunk: list[Subtitle],
        kinds: list[LineKind],
        translate_mask: list[bool],
        text_overrides: dict[int, str] | None = None,
    ) -> list[dict[str, Any]]:
        overrides = text_overrides or {}
        payload: list[dict[str, Any]] = []
        for s, kind, m in zip(chunk, kinds, translate_mask):
            if not m:
                continue
            src = overrides.get(s.id, s.text)
            item: dict[str, Any] = {"id": s.id, "text": src.replace("\n", " ")}
            if s.speaker:
                item["speaker"] = s.speaker
            if kind != LineKind.DIALOG:
                item["kind"] = kind.value
            if self._cfg.enforce_cps:
                item["max_chars"] = self._char_budget(s)
            payload.append(item)
        return payload

    def _copy_with_text(self, orig: Subtitle, text: str) -> Subtitle:
        return Subtitle(
            id=orig.id,
            start=orig.start,
            end=orig.end,
            text=text,
            speaker=orig.speaker,
            metadata=orig.metadata,
        )

    async def _enforce_cps_budget(
        self,
        index: int,
        chunk: list[Subtitle],
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """One compression retry for translations that exceed their char budget."""
        chunk_by_id = {s.id: s for s in chunk}

        overruns: list[dict[str, Any]] = []
        for item in items:
            try:
                item_id = int(item["id"])
            except (KeyError, TypeError, ValueError):
                continue
            orig = chunk_by_id.get(item_id)
            text = item.get("text")
            if orig is None or not isinstance(text, str):
                continue
            budget = self._char_budget(orig)
            visible_len = len(text.replace("‌", "").strip())
            if visible_len > int(budget * self._cfg.cps_tolerance):
                overruns.append({
                    "id": item_id,
                    "max_chars": budget,
                    "current": text,
                    "current_length": visible_len,
                })

        if not overruns:
            return items

        logger.info(
            "Chunk %d: %d translations exceed budget — requesting compression",
            index, len(overruns),
        )

        compression_msg = (
            "The following translations exceed their max_chars budget. "
            "Compress each one to fit, preserving meaning. Drop fillers, "
            "use shorter synonyms, contract clauses. Output ONLY a JSON array "
            'of {"id": int, "text": string} with the same ids.\n\n'
            + json.dumps(overruns, ensure_ascii=False)
        )
        try:
            content = await self._provider.chat(
                system=self._system_prompt,
                messages=[{"role": "user", "content": compression_msg}],
                model=self._model,
                temperature=self._temperature,
            )
            revised = _parse_response(content, len(overruns))
        except Exception as exc:
            logger.warning(
                "Chunk %d: CPS compression failed (%s), keeping over-budget translations",
                index, exc,
            )
            return items

        revised_by_id: dict[int, str] = {}
        for r in revised:
            try:
                rid = int(r["id"])
            except (KeyError, TypeError, ValueError):
                continue
            rtext = r.get("text")
            if isinstance(rtext, str) and rtext.strip():
                revised_by_id[rid] = rtext

        merged: list[dict[str, Any]] = []
        for item in items:
            try:
                item_id = int(item["id"])
            except (KeyError, TypeError, ValueError):
                merged.append(item)
                continue
            if item_id in revised_by_id:
                merged.append({"id": item_id, "text": revised_by_id[item_id]})
            else:
                merged.append(item)
        return merged

    async def _call_with_retry_and_correction(
        self,
        chunk_index: int,
        user_msg: str,
        expected_count: int,
    ) -> list[dict[str, Any]]:
        """
        Call the API with retries. On JSON parse failure, send a correction
        prompt asking the model to fix its output.
        """
        last_exc: Exception | None = None
        messages: list[dict[str, str]] = [
            {"role": "user", "content": user_msg},
        ]

        for attempt in range(1, self._cfg.max_retries + 1):
            # Initialize so the parse_exc handler can reference `content` even
            # if the chat call itself raises a ValueError before it's assigned.
            content = ""
            try:
                logger.info(
                    "Chunk %d: attempt %d/%d",
                    chunk_index, attempt, self._cfg.max_retries,
                )
                content = await self._provider.chat(
                    system=self._system_prompt,
                    messages=messages,
                    model=self._model,
                    temperature=self._temperature,
                )

                # Try to parse
                items = _parse_response(content, expected_count)
                logger.info("Chunk %d: success on attempt %d", chunk_index, attempt)
                return items

            except (ValueError, json.JSONDecodeError) as parse_exc:
                # JSON was invalid — ask model to correct.
                # If content is empty here, the chat call itself failed; treat
                # it as a transient error and retry without poisoning the message
                # history with an empty assistant turn.
                logger.warning(
                    "Chunk %d attempt %d: invalid JSON — sending correction prompt",
                    chunk_index, attempt,
                )
                if content:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": _CORRECTION_PROMPT})
                last_exc = parse_exc

            except ModelNotSupportedError:
                raise

            except Exception as exc:
                last_exc = exc
                delay = self._cfg.retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Chunk %d attempt %d failed: %s — retrying in %.1fs",
                    chunk_index, attempt, exc, delay,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Chunk {chunk_index} failed after {self._cfg.max_retries} attempts"
        ) from last_exc

    async def _refine(
        self, chunk_index: int, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Two-step refinement: critique the translations, then revise only
        the items the critique flagged. If the critique is empty, skip the
        revise call entirely."""
        if not items:
            return items

        lang = self._cfg.target_language
        items_json = json.dumps(items, ensure_ascii=False)

        try:
            logger.info("Chunk %d: refinement pass", chunk_index)
            payload = json.dumps(items, ensure_ascii=False)
            content = await self._provider.chat(
                system=self._refinement_prompt,
                messages=[{"role": "user", "content": payload}],
                model=self._model,
                temperature=0.0,
            )
            refined = _parse_response(content, len(items))
            logger.info("Chunk %d: refinement successful", chunk_index)
            return refined
        except ModelNotSupportedError:
            raise
        except Exception as exc:
            logger.warning(
                "Chunk %d: critique failed (%s) — skipping refinement", chunk_index, exc,
            )
            return items

        if _critique_is_empty(critique):
            logger.info("Chunk %d: critique returned no issues — keeping items", chunk_index)
            return items

        try:
            logger.info("Chunk %d: refinement revise", chunk_index)
            revised_raw = await self._provider.chat(
                system=_REVISE_SYSTEM_PROMPT.format(language=lang),
                messages=[{
                    "role": "user",
                    "content": _REVISE_USER_PROMPT.format(
                        items=items_json, critique=critique.strip(),
                    ),
                }],
                model=self._model,
                temperature=self._temperature,
            )
            revised = _parse_response(revised_raw, len(items))
            logger.info("Chunk %d: refinement applied", chunk_index)
            return revised
        except Exception as exc:
            logger.warning(
                "Chunk %d: revise failed (%s) — keeping original items", chunk_index, exc,
            )
            return items

    def _build_from_cache(
        self,
        chunk: list[Subtitle],
        kinds: list[LineKind] | None = None,
        translate_mask: list[bool] | None = None,
    ) -> list[Subtitle]:
        if kinds is None:
            kinds = [classify(s.text) for s in chunk]
        if translate_mask is None:
            translate_mask = [self._should_translate(k) for k in kinds]

        result: list[Subtitle] = []
        for orig, will_translate in zip(chunk, translate_mask):
            if will_translate:
                text = self._cache.get(orig.text) or orig.text
            else:
                text = orig.text
            result.append(self._copy_with_text(orig, text))
        return result


# -- Helpers --


# Runs of 2+ letters (any script). Used to distinguish prose that should be
# translated from names/interjections/symbols that legitimately stay verbatim.
_LETTER_RUN_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def _looks_translatable(text: str) -> bool:
    """Heuristic: does this line contain prose a translator would actually change?

    Returns False for content that legitimately stays identical across languages —
    symbols, numbers, music markers, and single proper-noun / interjection tokens
    such as "Carla!" or "Okay." — so an all-names chunk isn't mistaken for an
    untranslated one (see requirement 15's non-translatable-content exception).
    """
    words = _LETTER_RUN_RE.findall(text)
    if not words:
        return False  # only symbols / digits / music notes
    if len(words) == 1 and words[0][:1].isupper():
        return False  # a single capitalized token — likely a name or interjection
    return True


def _validate_translation(
    original: list[Subtitle],
    translated: list[Subtitle],
    should_translate: Callable[[LineKind], bool],
) -> str | None:
    """Validate a translated chunk; return an error string, or None if OK.

    Treats as failures: empty output, a changed subtitle count, or a chunk where
    nothing was translated yet at least one line is real prose that came back
    identical to the source (the model echoed the source instead of translating).
    Lines the user opted out of translating (cues/lyrics) and non-translatable
    content (names, symbols, music markers) are excluded, so a chunk of only
    names like "Carla!" legitimately passes.
    """
    if not translated:
        return "empty translation"
    if len(translated) != len(original):
        return f"subtitle count changed ({len(original)} -> {len(translated)})"

    translatable = [
        (o, t)
        for o, t in zip(original, translated)
        if should_translate(classify(o.text))
    ]
    any_changed = any(t.text.strip() != o.text.strip() for o, t in translatable)
    unchanged_prose = [
        o
        for o, t in translatable
        if t.text.strip() == o.text.strip() and _looks_translatable(o.text)
    ]
    if unchanged_prose and not any_changed:
        return "translation identical to source"
    return None


def _critique_is_empty(raw: str) -> bool:
    """True when the critique step returned no actionable issues."""
    text = raw.strip()
    if not text:
        return True
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return False
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return False
    if isinstance(data, list):
        if len(data) == 0:
            return True
        # Treat items with no real issues list as a no-op too.
        return all(
            isinstance(item, dict)
            and not [i for i in (item.get("issues") or []) if str(i).strip()]
            for item in data
        )
    return False


def _parse_response(raw: str, expected_count: int) -> list[dict[str, Any]]:
    """
    Safely extract a JSON array from the model response.
    Handles markdown-wrapped responses like ```json ... ```.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON array in the text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            data = json.loads(text[start : end + 1])
        else:
            raise ValueError(f"Could not parse JSON from response: {text[:200]}")

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    if len(data) != expected_count:
        logger.warning(
            "Expected %d items but got %d — using available items",
            expected_count, len(data),
        )

    # Validate each item has required fields
    for item in data:
        if "id" not in item or "text" not in item:
            raise ValueError(f"Item missing 'id' or 'text': {item}")

    return data


_CLAUSE_BREAK_RE = re.compile(r"[،؛.,;:]\s")


def _restore_multiline(translated: str, original: str) -> str:
    """Restore the original line count, preferring clause-boundary splits
    (Persian or Latin punctuation followed by whitespace). Falls back to an
    even word-count split when there aren't enough clause breaks to honor."""
    n_lines = original.count("\n") + 1
    if n_lines <= 1:
        return translated

    candidates = [m.end() for m in _CLAUSE_BREAK_RE.finditer(translated)]
    if len(candidates) < n_lines - 1:
        return _split_by_words(translated, n_lines)

    # Pick the n-1 split points whose positions are closest to even spacing.
    target_lengths = [len(translated) * i // n_lines for i in range(1, n_lines)]
    chosen: list[int] = []
    remaining = list(candidates)
    for target in target_lengths:
        best = min(remaining, key=lambda c: abs(c - target))
        chosen.append(best)
        remaining.remove(best)
    chosen.sort()

    lines: list[str] = []
    prev = 0
    for split in chosen:
        lines.append(translated[prev:split].strip())
        prev = split
    lines.append(translated[prev:].strip())

    lines = [ln for ln in lines if ln]
    return "\n".join(lines) if lines else translated


def _split_by_words(translated: str, n_lines: int) -> str:
    """Even word-count split — fallback when clause boundaries don't exist."""
    words = translated.split()
    if len(words) <= 1:
        return translated

    per_line = max(1, len(words) // n_lines)
    lines: list[str] = []
    for i in range(n_lines):
        start = i * per_line
        if i == n_lines - 1:
            lines.append(" ".join(words[start:]))
        else:
            lines.append(" ".join(words[start : start + per_line]))

    lines = [ln for ln in lines if ln]
    return "\n".join(lines) if lines else translated
