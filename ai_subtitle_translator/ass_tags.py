"""ASS subtitle override-tag preservation.

ASS dialogue text can carry inline formatting like ``{\\i1}italic{\\i0}``,
``{\\b1}bold{\\b0}``, and positional metadata like ``{\\pos(x,y)}`` or
``{\\an5}``. Passing these tags through to the translation prompt usually
mangles them — models either reorder, drop, or "translate" the contents.

The fix is to strip overrides before translation (sending only visible text
to the model) and restore them afterwards:

  - Inline formatting tags are placed back at proportionally-mapped offsets
    in the translated text.
  - Positional tags (``\\pos``, ``\\move``, ``\\an``, ``\\org``, etc.) are
    line-level metadata and are reattached at the very start of the line.
"""

from __future__ import annotations

import re

_ASS_TAG_RE = re.compile(r"\{[^}]+\}")

_POSITIONAL_PREFIXES: tuple[str, ...] = (
    "{\\pos(", "{\\move(", "{\\org(", "{\\an", "{\\fad(", "{\\fade(", "{\\clip(",
)


def has_tags(text: str) -> bool:
    """Cheap check before paying for full stripping."""
    return _ASS_TAG_RE.search(text) is not None


def strip_tags(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Return (clean_text, [(index_in_clean, tag), ...]).

    ``index_in_clean`` is the position the tag occupied relative to the
    *cleaned* text — i.e. how many visible characters precede it. Indices
    are stable under whitespace-preserving transforms like ``replace("\\n", " ")``
    because newline replacement does not change length.
    """
    tags: list[tuple[int, str]] = []
    parts: list[str] = []
    pos = 0
    clean_len = 0
    for m in _ASS_TAG_RE.finditer(text):
        chunk = text[pos:m.start()]
        parts.append(chunk)
        clean_len += len(chunk)
        tags.append((clean_len, m.group()))
        pos = m.end()
    parts.append(text[pos:])
    return "".join(parts), tags


def restore_tags(
    translated: str,
    tags: list[tuple[int, str]],
    original_clean_len: int,
) -> str:
    """Reinsert ASS tags into ``translated``.

    Inline tags are placed at offsets proportional to their original position
    (``orig_index / original_clean_len`` mapped onto the translated length).
    Positional/metadata tags are concatenated at the very start. Best-effort:
    we lose nothing, we just may shift italics by a few characters in the
    translated rendering."""
    if not tags:
        return translated

    positional: list[str] = []
    inline: list[tuple[int, str]] = []
    for idx, tag in tags:
        if _is_positional(tag):
            positional.append(tag)
        else:
            inline.append((idx, tag))

    body = translated
    if inline:
        if original_clean_len <= 0:
            # No visible text in the source; just stack tags at the start.
            body = "".join(t for _, t in inline) + body
        else:
            ratio = len(translated) / original_clean_len
            # Insert from the rightmost position so earlier insertions don't
            # shift later ones.
            inline_sorted = sorted(inline, key=lambda p: p[0], reverse=True)
            for orig_index, tag in inline_sorted:
                insert_at = int(round(orig_index * ratio))
                insert_at = max(0, min(len(body), insert_at))
                body = body[:insert_at] + tag + body[insert_at:]

    return "".join(positional) + body


def _is_positional(tag: str) -> bool:
    return any(tag.startswith(p) for p in _POSITIONAL_PREFIXES)
