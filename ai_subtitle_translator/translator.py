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
from typing import Any, Protocol

from ai_subtitle_translator.cache import TranslationCache
from ai_subtitle_translator.config import TranslatorConfig
from ai_subtitle_translator.glossary import Glossary
from ai_subtitle_translator.parser import Subtitle
from ai_subtitle_translator.postprocess import postprocess_persian

logger = logging.getLogger(__name__)

# -- Prompt builders --


def _build_system_prompt(
    language: str,
    glossary: Glossary | None = None,
) -> str:
    glossary_section = glossary.build_prompt_section() if glossary and not glossary.is_empty else ""

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
- Make it sound like {language} movie subtitles{glossary_section}

INPUT:
JSON array of subtitle objects with "id" and "text" fields.

OUTPUT:
JSON array with the same "id" fields and translated "text" fields. Nothing else."""


def _build_user_message(
    payload: list[dict[str, Any]],
    context: list[Subtitle] | None = None,
) -> str:
    """Build user message with optional previous-chunk context."""
    parts: list[str] = []

    if context:
        ctx_lines = [f'  - [{s.id}] "{s.text}"' for s in context]
        parts.append(
            "Previous context (for continuity only, do NOT translate these):\n"
            + "\n".join(ctx_lines)
            + "\n"
        )

    parts.append("Translate the following:\n" + json.dumps(payload, ensure_ascii=False))
    return "\n".join(parts)


_REFINEMENT_PROMPT = """You are a Persian subtitle editor. Improve this translated subtitle text:
- Make it more natural and conversational
- Fix any awkward phrasing
- Keep it concise for subtitle readability
- Do NOT change the JSON structure

Input and output: JSON array of {{"id": int, "text": string}}.
Return ONLY the improved JSON array."""

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

    def __init__(self, api_key: str | None, base_url: str | None) -> None:
        from anthropic import AsyncAnthropic

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)

    async def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        response = await self._client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=system,
            messages=messages,  # type: ignore[arg-type]
        )
        return response.content[0].text


def _build_provider(config: TranslatorConfig) -> _ChatProvider:
    """Create the appropriate provider based on config."""
    if config.provider == "anthropic":
        return _AnthropicProvider(api_key=config.anthropic_api_key, base_url=config.anthropic_base_url)
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

        self._system_prompt = _build_system_prompt(
            self._cfg.target_language, glossary
        )

    @property
    def cache(self) -> TranslationCache:
        return self._cache

    async def translate_chunks(
        self,
        chunks: list[list[Subtitle]],
        contexts: list[list[Subtitle] | None] | None = None,
    ) -> list[list[Subtitle]]:
        """Translate all chunks in parallel (bounded by semaphore)."""
        if contexts is None:
            contexts = [None] * len(chunks)

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

    async def _translate_chunk(
        self,
        index: int,
        chunk: list[Subtitle],
        context: list[Subtitle] | None,
    ) -> list[Subtitle]:
        """Translate a single chunk with semaphore, cache, retry, and post-processing."""
        async with self._semaphore:
            # Check cache for every line
            all_cached = all(self._cache.has(s.text) for s in chunk)
            if all_cached:
                logger.info("Chunk %d fully cached, skipping API call", index)
                return self._build_from_cache(chunk)

            # Build payload (join multi-line text with space for cleaner translation)
            payload = [{"id": s.id, "text": s.text.replace("\n", " ")} for s in chunk]

            user_msg = _build_user_message(payload, context)
            raw = await self._call_with_retry_and_correction(index, user_msg, len(chunk))
            translated_items = raw  # already parsed

            # Refinement pass (optional)
            if self._cfg.enable_refinement:
                translated_items = await self._refine(index, translated_items)

            # Build result with post-processing
            result: list[Subtitle] = []
            for orig, trans in zip(chunk, translated_items):
                translated_text = trans.get("text", orig.text)

                # Apply Persian post-processing if target is Persian
                if "persian" in self._cfg.target_language.lower() or "farsi" in self._cfg.target_language.lower():
                    translated_text = postprocess_persian(translated_text)

                # Restore multi-line structure if original was multi-line
                if "\n" in orig.text:
                    translated_text = _restore_multiline(translated_text, orig.text)

                self._cache.put(orig.text, translated_text)
                result.append(Subtitle(
                    id=orig.id,
                    start=orig.start,
                    end=orig.end,
                    text=translated_text,
                    metadata=orig.metadata,
                ))

            return result

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
                # JSON was invalid — ask model to correct
                logger.warning(
                    "Chunk %d attempt %d: invalid JSON — sending correction prompt",
                    chunk_index, attempt,
                )
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
        """Optional second pass: send translations back for fluency improvement."""
        try:
            logger.info("Chunk %d: refinement pass", chunk_index)
            payload = json.dumps(items, ensure_ascii=False)
            content = await self._provider.chat(
                system=_REFINEMENT_PROMPT,
                messages=[{"role": "user", "content": payload}],
                model=self._model,
                temperature=self._temperature,
            )
            refined = _parse_response(content, len(items))
            logger.info("Chunk %d: refinement successful", chunk_index)
            return refined
        except Exception as exc:
            logger.warning("Chunk %d: refinement failed (%s), using original", chunk_index, exc)
            return items

    def _build_from_cache(self, chunk: list[Subtitle]) -> list[Subtitle]:
        return [
            Subtitle(
                id=s.id,
                start=s.start,
                end=s.end,
                text=self._cache.get(s.text) or s.text,
                metadata=s.metadata,
            )
            for s in chunk
        ]


# -- Helpers --


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


def _restore_multiline(translated: str, original: str) -> str:
    """
    If the original subtitle was multi-line, try to split the translated
    text into the same number of lines (split roughly by midpoint).
    """
    original_lines = original.split("\n")
    n_lines = len(original_lines)

    if n_lines <= 1:
        return translated

    # Split translated text into roughly equal parts
    words = translated.split()
    if len(words) <= 1:
        return translated

    # Distribute words across lines as evenly as possible
    per_line = max(1, len(words) // n_lines)
    lines: list[str] = []
    for i in range(n_lines):
        start = i * per_line
        if i == n_lines - 1:
            lines.append(" ".join(words[start:]))
        else:
            lines.append(" ".join(words[start : start + per_line]))

    # Filter out empty lines
    lines = [ln for ln in lines if ln]
    return "\n".join(lines) if lines else translated
