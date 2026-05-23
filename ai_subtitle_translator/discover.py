"""Auto-extract a candidate glossary from the source subtitles and ask the
provider to propose target-language translations for those terms.

This is a v1 heuristic: capitalized tokens that appear N+ times across the
file are treated as proper-noun candidates. A small stoplist filters common
English words/titles that would otherwise dominate the output."""

from __future__ import annotations

import json
import logging
import re
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


# Capitalized tokens (Latin alphabet, allowing apostrophes/hyphens inside).
_CANDIDATE_RE = re.compile(r"\b[A-Z][a-zA-Z][a-zA-Z'\-]*\b")

# Common capitalized words that aren't proper nouns. Kept intentionally small —
# the goal is to drop the worst offenders, not to whitelist every English word.
_STOPLIST: frozenset[str] = frozenset({
    "I", "I'm", "I'll", "I've", "I'd",
    "Mr", "Mrs", "Ms", "Dr", "Sir", "Madam", "Miss",
    "OK", "Okay", "Yes", "No", "Maybe", "Hi", "Hello", "Hey", "Bye",
    "Mom", "Dad", "Mum", "Mama", "Papa",
    "God", "Jesus", "Christ", "Lord",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "English", "French", "Spanish", "German", "Italian", "Chinese",
    "American", "British",
    "Mr.", "Mrs.", "Ms.", "Dr.",
})

_SENTENCE_START_RE = re.compile(r"(?:^|[.!?؟]\s+)")


def extract_candidates(
    subtitles: list[Subtitle],
    min_occurrences: int = 3,
) -> list[str]:
    """Return the proper-noun-like terms that appear ``min_occurrences``+ times.

    Heuristic: a candidate must either appear in mid-sentence position at least
    once (where capitalization implies a proper noun) OR appear ``min_occurrences``
    times total even after stoplist filtering. Returns terms sorted by descending
    frequency, then alphabetically for stability.
    """
    if not subtitles:
        return []

    counts: dict[str, int] = {}
    mid_sentence: set[str] = set()

    for sub in subtitles:
        text = sub.text
        if not text:
            continue
        # Mark positions that start a new sentence so we can tell whether a
        # capitalized token appears mid-sentence (a stronger proper-noun signal).
        sentence_starts: set[int] = {0}
        for m in _SENTENCE_START_RE.finditer(text):
            sentence_starts.add(m.end())

        for m in _CANDIDATE_RE.finditer(text):
            term = m.group()
            if term in _STOPLIST:
                continue
            counts[term] = counts.get(term, 0) + 1
            if m.start() not in sentence_starts:
                mid_sentence.add(term)

    candidates = [
        term for term, n in counts.items()
        if n >= min_occurrences or term in mid_sentence
    ]
    # Re-filter against min_occurrences for the mid-sentence path too — a term
    # appearing only once still isn't useful glossary material.
    candidates = [t for t in candidates if counts[t] >= max(2, min_occurrences - 1)]

    candidates.sort(key=lambda t: (-counts[t], t))
    return candidates


_PROPOSAL_SYSTEM_PROMPT = (
    "You produce subtitle-glossary entries. Translate proper nouns and "
    "recurring terms accurately for the target language. Names of people are "
    "transliterated; place names use the standard local rendering when one "
    "exists. Output ONLY a JSON object mapping each input term to its "
    "translation. No prose, no markdown."
)


async def propose_translations(
    provider: _ChatProvider,
    model: str,
    target_language: str,
    candidates: list[str],
    temperature: float = 0.0,
) -> dict[str, str]:
    """Ask the provider to propose translations for the candidate terms.

    Returns an empty dict on failure — discovery is best-effort and must not
    block translation if the proposal call fails or returns malformed JSON.
    """
    if not candidates:
        return {}

    user_msg = (
        f"Target language: {target_language}\n\n"
        f"Terms (proper nouns and recurring vocabulary from a film):\n"
        f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
        "Return a JSON object like "
        '{"Term": "ترجمه", ...} covering every term above.'
    )
    try:
        raw = await provider.chat(
            system=_PROPOSAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            model=model,
            temperature=temperature,
        )
    except Exception as exc:
        logger.warning("Glossary proposal call failed (%s)", exc)
        return {}

    parsed = _parse_proposal(raw)
    if not parsed:
        logger.warning("Glossary proposal returned unparseable JSON: %s", raw[:200])
        return {}

    cleaned: dict[str, str] = {}
    for term, translation in parsed.items():
        if not isinstance(term, str) or not isinstance(translation, str):
            continue
        term_s = term.strip()
        trans_s = translation.strip()
        if term_s and trans_s and term_s != trans_s:
            cleaned[term_s] = trans_s
    return cleaned


def _parse_proposal(raw: str) -> dict[str, str] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if isinstance(data, dict):
        return data
    return None
