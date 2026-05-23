"""Run-level story memory: a rolling 2–3 sentence summary kept up to date as
chunks are translated, so the model can maintain character/context continuity
across a long file without re-paying the full transcript in every prompt."""

from __future__ import annotations

import logging
from typing import Protocol

from ai_subtitle_translator.parser import Subtitle

logger = logging.getLogger(__name__)


class _ChatProvider(Protocol):
    async def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str: ...


_SUMMARY_SYSTEM_PROMPT = (
    "You maintain a short rolling summary for a subtitle translation job. "
    "Output only the updated summary text — no preamble, no JSON, no quotes."
)


class StoryMemory:
    """A rolling free-text summary updated periodically as chunks are translated."""

    def __init__(self, max_chars: int = 600) -> None:
        self._summary: str = ""
        self._max_chars = max_chars

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def is_empty(self) -> bool:
        return not self._summary

    def context_block(self) -> str:
        """Prompt fragment to inject into each chunk's user message."""
        if not self._summary:
            return ""
        return (
            "STORY SO FAR (for character/context continuity, do NOT translate):\n"
            f"{self._summary}\n"
        )

    async def update(
        self,
        provider: _ChatProvider,
        model: str,
        new_subtitles: list[Subtitle],
        temperature: float = 0.0,
    ) -> None:
        """Fold the source text of new_subtitles into the rolling summary."""
        if not new_subtitles:
            return

        new_scene = "\n".join(s.text for s in new_subtitles if s.text.strip())
        if not new_scene:
            return

        user_msg = (
            f"Current summary:\n{self._summary or '(none yet)'}\n\n"
            f"New scene (source subtitle lines):\n{new_scene}\n\n"
            "Update the summary to 2–3 sentences capturing characters, "
            "relationships, setting, and current situation. Keep it under "
            f"{self._max_chars} characters. If the current summary is empty, "
            "write the first version. Output only the new summary."
        )
        try:
            response = await provider.chat(
                system=_SUMMARY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                model=model,
                temperature=temperature,
            )
        except Exception as exc:
            logger.warning("StoryMemory update failed (%s) — keeping prior summary", exc)
            return

        updated = response.strip()
        if not updated:
            return
        if len(updated) > self._max_chars * 2:
            updated = updated[: self._max_chars * 2].rstrip()
        self._summary = updated
        logger.info("StoryMemory updated (%d chars)", len(self._summary))
