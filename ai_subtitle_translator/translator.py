"""Async OpenAI translator with concurrency control, retries, and caching."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI

from ai_subtitle_translator.config import TranslatorConfig
from ai_subtitle_translator.parser import Subtitle

logger = logging.getLogger(__name__)


def _build_system_prompt(language: str) -> str:
    return f"""You are a professional subtitle translator specializing in {language}.

Your task is to translate subtitles into natural, fluent, and simple {language}.

STRICT RULES:
- Keep the translation natural and conversational (like real spoken {language})
- Avoid literal translation
- Use simple and clear {language}
- Preserve the exact number of subtitle items
- Do NOT merge or split lines
- Keep alignment with original meaning
- Do NOT add explanations
- Output MUST be valid JSON

STYLE:
- Use modern spoken {language}
- Avoid formal/literary tone
- Keep sentences short and natural
- Make it sound like {language} movie subtitles

INPUT:
JSON array of subtitle objects with "id" and "text" fields.

OUTPUT:
JSON array with the same "id" fields and translated "text" fields. Nothing else."""


class Translator:
    def __init__(self, config: TranslatorConfig | None = None) -> None:
        self._cfg = config or TranslatorConfig()
        self._client = AsyncOpenAI(
            api_key=self._cfg.api_key,      # None → falls back to OPENAI_API_KEY env
            base_url=self._cfg.base_url,     # None → default OpenAI endpoint
        )
        self._semaphore = asyncio.Semaphore(self._cfg.max_concurrency)
        self._system_prompt = _build_system_prompt(self._cfg.target_language)
        # Simple in-memory cache: source text -> translated text
        self._cache: dict[str, str] = {}

    async def translate_chunks(
        self,
        chunks: list[list[Subtitle]],
        overlap: int = 0,
    ) -> list[list[Subtitle]]:
        """Translate all chunks in parallel (bounded by semaphore)."""
        tasks = [
            self._translate_chunk(i, chunk, overlap)
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        translated: list[list[Subtitle]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Chunk %d failed permanently: %s", i, result)
                # Fallback: return original subtitles untranslated
                chunk = chunks[i]
                if overlap > 0 and i > 0:
                    chunk = chunk[overlap:]
                translated.append(chunk)
            else:
                translated.append(result)

        return translated

    async def _translate_chunk(
        self,
        index: int,
        chunk: list[Subtitle],
        overlap: int,
    ) -> list[Subtitle]:
        """Translate a single chunk with semaphore, retry, and cache."""
        async with self._semaphore:
            # Check if every line in the chunk is cached
            all_cached = all(sub.text in self._cache for sub in chunk)
            if all_cached:
                logger.info("Chunk %d fully cached, skipping API call", index)
                subs = _apply_cache(chunk, self._cache)
                if overlap > 0 and index > 0:
                    subs = subs[overlap:]
                return subs

            payload = [{"id": s.id, "text": s.text} for s in chunk]
            raw = await self._call_api_with_retry(index, payload)
            translated_items = _parse_response(raw, expected_count=len(chunk))

            # Build result subtitles and populate cache
            result: list[Subtitle] = []
            for orig, trans in zip(chunk, translated_items):
                translated_text = trans.get("text", orig.text)
                self._cache[orig.text] = translated_text
                result.append(
                    Subtitle(
                        id=orig.id,
                        start=orig.start,
                        end=orig.end,
                        text=translated_text,
                    )
                )

            # Strip overlap items (they were only there for context)
            if overlap > 0 and index > 0:
                result = result[overlap:]

            return result

    async def _call_api_with_retry(
        self, chunk_index: int, payload: list[dict[str, Any]]
    ) -> str:
        """Call the OpenAI API with exponential backoff retries."""
        last_exc: Exception | None = None
        user_content = json.dumps(payload, ensure_ascii=False)

        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                logger.info(
                    "Chunk %d: attempt %d/%d (%d items)",
                    chunk_index, attempt, self._cfg.max_retries, len(payload),
                )
                response = await self._client.chat.completions.create(
                    model=self._cfg.model,
                    temperature=self._cfg.temperature,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                content = response.choices[0].message.content or ""
                logger.info("Chunk %d: success on attempt %d", chunk_index, attempt)
                return content

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


def _parse_response(raw: str, expected_count: int) -> list[dict[str, Any]]:
    """
    Safely extract a JSON array from the model response.
    Handles markdown-wrapped responses like ```json ... ```.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Last resort: try to find a JSON array in the text
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

    return data


def _apply_cache(chunk: list[Subtitle], cache: dict[str, str]) -> list[Subtitle]:
    """Build translated subtitles from cache."""
    return [
        Subtitle(id=s.id, start=s.start, end=s.end, text=cache[s.text])
        for s in chunk
    ]
