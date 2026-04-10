"""Smart chunking: groups subtitles into translation-friendly chunks."""

from __future__ import annotations

from ai_subtitle_translator.config import ChunkConfig
from ai_subtitle_translator.parser import Subtitle


def chunk_subtitles(
    subtitles: list[Subtitle],
    cfg: ChunkConfig | None = None,
) -> list[list[Subtitle]]:
    """
    Split subtitles into chunks using a hybrid strategy:
      1. Time-gap based splitting (dialogue boundaries)
      2. Size limits (max lines and max characters)
      3. Optional overlap for context preservation
    """
    if not subtitles:
        return []

    cfg = cfg or ChunkConfig()

    # Step 1: split into dialogue groups by time gap
    dialogue_groups = _split_by_time_gap(subtitles, cfg.time_gap_threshold_ms)

    # Step 2: enforce size limits within each group
    sized_chunks: list[list[Subtitle]] = []
    for group in dialogue_groups:
        sized_chunks.extend(_split_by_size(group, cfg.max_lines, cfg.max_chars))

    # Step 3: add overlap between adjacent chunks for context
    if cfg.overlap_lines > 0 and len(sized_chunks) > 1:
        sized_chunks = _add_overlap(sized_chunks, cfg.overlap_lines)

    return sized_chunks


def _split_by_time_gap(
    subtitles: list[Subtitle], threshold_ms: int
) -> list[list[Subtitle]]:
    """Start a new group whenever the gap between consecutive subtitles exceeds threshold."""
    groups: list[list[Subtitle]] = [[subtitles[0]]]

    for prev, curr in zip(subtitles, subtitles[1:]):
        gap = curr.start_ms - prev.end_ms
        if gap > threshold_ms:
            groups.append([curr])
        else:
            groups[-1].append(curr)

    return groups


def _split_by_size(
    group: list[Subtitle], max_lines: int, max_chars: int
) -> list[list[Subtitle]]:
    """Break a group further if it exceeds line or character limits."""
    chunks: list[list[Subtitle]] = []
    current: list[Subtitle] = []
    current_chars = 0

    for sub in group:
        sub_chars = len(sub.text)

        # Would adding this subtitle exceed limits?
        would_exceed_lines = len(current) >= max_lines
        would_exceed_chars = current_chars + sub_chars > max_chars and current

        if would_exceed_lines or would_exceed_chars:
            chunks.append(current)
            current = [sub]
            current_chars = sub_chars
        else:
            current.append(sub)
            current_chars += sub_chars

    if current:
        chunks.append(current)

    return chunks


def _add_overlap(
    chunks: list[list[Subtitle]], overlap: int
) -> list[list[Subtitle]]:
    """
    Prepend the last N subtitles of the previous chunk to the next chunk.
    Overlap subtitles carry context but are stripped after translation.
    """
    result: list[list[Subtitle]] = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        # Mark overlap items by prepending them; the translator module
        # will know to discard the first `overlap` items from the response.
        result.append(prev_tail + chunks[i])

    return result
