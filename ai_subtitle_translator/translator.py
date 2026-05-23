"""
Async subtitle translator with multi-provider support (OpenAI & Anthropic).

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
from typing import Any, Protocol

from ai_subtitle_translator.ass_tags import has_tags, restore_tags, strip_tags
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


def _build_system_prompt(
    language: str,
    glossary: Glossary | None = None,
    film_context: str = "",
) -> str:
    glossary_section = glossary.build_prompt_section() if glossary and not glossary.is_empty else ""
    context_section = ""
    if film_context.strip():
        context_section = (
            "\n\nFILM CONTEXT:\n"
            f"{film_context.strip()}\n"
            f"Tune register, tone, and word choice to fit this context consistently."
        )

    return f"""You are a professional subtitle translator specializing in {language}.

Your task is to translate subtitles into natural, fluent, and simple {language}.

STRICT RULES:
- Keep translations natural and conversational (like real spoken {language})
- Avoid literal translation
- Use simple and clear {language}
- Preserve the exact number of subtitle items
- Do NOT merge or split lines
- Keep alignment with original meaning
- Do NOT add explanations
- Output MUST be valid JSON
- Keep translations concise to fit subtitle reading speed

STYLE:
- Use modern spoken {language}
- Avoid formal/literary tone
- Keep sentences short and natural
- Make it sound like {language} movie subtitles

SPEAKER CONSISTENCY:
- Some items include a "speaker" field (character name).
- Maintain a consistent register (formal vs informal) per speaker across the
  ENTIRE file. Pick register from relationship cues in the dialog
  (boss/employee → formal, friends → informal, parent → child → mixed).
- Never switch register for the same speaker mid-conversation unless the
  source clearly signals it.
- Within a single line, a leading dash ("-" / "–" / "—") marks a speaker
  change between two turns. Translate each turn naturally and keep the dash.

LINE KINDS (the "kind" field, default is dialog when absent):
- "dialog" — translate naturally as spoken {language}.
- "sound_cue" — bracketed sounds like [music]. Translate inside brackets,
  e.g. [music] → [موسیقی]. Keep the brackets.
- "stage_dir" — parenthetical action like (whispering). Translate inside the
  parentheses. Keep the parentheses.
- "screen_text" — on-screen text (signs, titles, captions). Translate
  naturally without conversational softening.
- "lyrics" — translate poetically; preserve any ♪ markers.

LENGTH BUDGET:
- Some items have "max_chars". The translation MUST fit within that budget.
- Compress aggressively when needed: drop fillers ("you know", "well", "I
  mean"), contract clauses, use shorter synonyms.
- Meaning preservation > literal preservation.{glossary_section}{context_section}

INPUT:
JSON array of subtitle objects with "id" and "text" fields, plus optional
"speaker", "kind", and "max_chars" fields.

OUTPUT:
JSON array with the same "id" fields and translated "text" fields. Nothing else."""


def _build_user_message(
    payload: list[dict[str, Any]],
    context: list[Subtitle] | None = None,
    memory_block: str = "",
) -> str:
    """Build user message with optional rolling-summary and previous-chunk context."""
    parts: list[str] = []

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


_PROBE_PROMPT = (
    "Given these subtitle lines from a film, in 2 short lines describe "
    "(1) genre / setting and (2) register (formal, conversational, period, "
    "slangy, poetic, technical, etc.). Be concrete and brief — these notes "
    "tune a downstream translator."
)

_CRITIQUE_SYSTEM_PROMPT = (
    "You are a {language} subtitle editor. Identify SPECIFIC issues only. "
    "Do not rewrite. Do not praise. Be terse."
)

_CRITIQUE_USER_PROMPT = (
    "Find issues in these translations (naturalness, register, length, "
    "glossary, awkward phrasing). Output a JSON array of "
    '{{"id": int, "issues": [string]}} for items with problems. '
    "Empty array if all are fine.\n\n"
    "{payload}"
)

_REVISE_SYSTEM_PROMPT = (
    "You revise {language} subtitles given specific criticism. Apply every "
    "flagged fix, change nothing else, preserve item ids exactly."
)

_REVISE_USER_PROMPT = (
    "Original translations:\n{items}\n\n"
    "Criticism:\n{critique}\n\n"
    "Output the revised JSON array of {{\"id\": int, \"text\": string}}, "
    "same ids, fixing every flagged issue. JSON only."
)

_CORRECTION_PROMPT = (
    "Your previous output was not valid JSON. "
    "Return ONLY a valid JSON array of objects with \"id\" (int) and \"text\" (string) fields. "
    "No markdown, no explanation, just the JSON array."
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


class _OpenAIProvider:
    """OpenAI-compatible provider (works with any OpenAI-compatible endpoint)."""

    def __init__(self, api_key: str | None, base_url: str | None) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        api_messages = [{"role": "system", "content": system}, *messages]
        response = await self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=api_messages,  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""


class _AnthropicProvider:
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        max_tokens: int = 4096,
    ) -> None:
        from anthropic import AsyncAnthropic

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._max_tokens = max_tokens

    async def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        response = await self._client.messages.create(
            model=model,
            max_tokens=self._max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,  # type: ignore[arg-type]
        )
        return response.content[0].text


def _build_provider(config: TranslatorConfig) -> _ChatProvider:
    """Create the appropriate provider based on config."""
    if config.provider == "anthropic":
        return _AnthropicProvider(
            api_key=config.anthropic_api_key,
            base_url=config.anthropic_base_url,
            max_tokens=config.anthropic_max_tokens,
        )
    return _OpenAIProvider(api_key=config.api_key, base_url=config.base_url)


# -- Translator --


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

        # Resolve active model/temperature based on provider
        if self._cfg.provider == "anthropic":
            self._model = self._cfg.anthropic_model
            self._temperature = self._cfg.anthropic_temperature
        else:
            self._model = self._cfg.model
            self._temperature = self._cfg.temperature

        # Run-level state (Phase 2)
        self._memory: StoryMemory | None = (
            StoryMemory() if self._cfg.enable_memory else None
        )
        self._film_context: str = (self._cfg.register_override or "").strip()

        self._system_prompt = _build_system_prompt(
            self._cfg.target_language, glossary, self._film_context,
        )

    @property
    def cache(self) -> TranslationCache:
        return self._cache

    async def translate_chunks(
        self,
        chunks: list[list[Subtitle]],
        contexts: list[list[Subtitle] | None] | None = None,
    ) -> list[list[Subtitle]]:
        """Translate all chunks, optionally with a one-shot register probe and
        a rolling story summary updated every N chunks."""
        if contexts is None:
            contexts = [None] * len(chunks)

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

        if self._memory is not None:
            return await self._translate_in_batches(chunks, contexts)

        # Full-parallel path (memory disabled).
        tasks = [
            self._translate_chunk(i, chunk, contexts[i])
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        translated: list[list[Subtitle]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Chunk %d failed permanently: %s", i, result)
                translated.append(chunks[i])  # fallback: original text
            else:
                translated.append(result)

        return translated

    async def _translate_in_batches(
        self,
        chunks: list[list[Subtitle]],
        contexts: list[list[Subtitle] | None],
    ) -> list[list[Subtitle]]:
        """Batched-parallel: translate N chunks sharing the current summary,
        then fold that batch's source text into the summary before the next."""
        assert self._memory is not None
        batch_size = max(1, self._cfg.memory_update_interval)
        translated: list[list[Subtitle]] = []

        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            memory_block = self._memory.context_block()
            tasks = [
                self._translate_chunk(i, chunks[i], contexts[i], memory_block=memory_block)
                for i in range(start, end)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for offset, result in enumerate(results):
                idx = start + offset
                if isinstance(result, Exception):
                    logger.error("Chunk %d failed permanently: %s", idx, result)
                    translated.append(chunks[idx])
                else:
                    translated.append(result)

            batch_subs = [s for i in range(start, end) for s in chunks[i]]
            await self._memory.update(
                self._provider, self._model, batch_subs, temperature=0.0,
            )

        return translated

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
            user_msg = _build_user_message(payload, context, memory_block=memory_block)
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

                translated_text = by_id.get(orig.id, orig.text)

                if self._cfg.enable_postprocess and self._is_persian_target():
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
            logger.info("Chunk %d: refinement critique", chunk_index)
            critique = await self._provider.chat(
                system=_CRITIQUE_SYSTEM_PROMPT.format(language=lang),
                messages=[{
                    "role": "user",
                    "content": _CRITIQUE_USER_PROMPT.format(payload=items_json),
                }],
                model=self._model,
                temperature=0.0,
            )
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
