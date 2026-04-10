"""Persian (Farsi) post-processing pipeline for translated subtitles."""

from __future__ import annotations

import re


def postprocess_persian(text: str) -> str:
    """Run the full Persian normalization pipeline on a subtitle string."""
    text = fix_nim_fasele(text)
    text = convert_punctuation(text)
    text = simplify_formal(text)
    text = normalize_whitespace(text)
    return text


# -- Half-space (nim-fasele) fixes --

# Unicode range for Persian/Arabic characters (used as guard in patterns)
_PERSIAN_CHAR = "[\u0600-\u06FF]"

# Prefix patterns — only join when followed by a Persian letter (verb stem)
# بر+می compound must be matched first, before the standalone می pattern
_PREFIX_PATTERNS = [
    (r"بر\s+می\s+(?=" + _PERSIAN_CHAR + ")", "بر\u200cمی\u200c"),  # بر می گرده → بر‌می‌گرده
    (r"(?<!\S)می\s+(?=" + _PERSIAN_CHAR + ")", "می\u200c"),         # می خوام → می‌خوام
    (r"(?<!\S)نمی\s+(?=" + _PERSIAN_CHAR + ")", "نمی\u200c"),        # نمی دونم → نمی‌دونم
]

# Suffix patterns — only join when preceded by a Persian letter
# NOTE: ای, ام, ات are intentionally excluded (too ambiguous — can be standalone words)
_SUFFIX_PATTERNS = [
    (r"(?<=" + _PERSIAN_CHAR + r")\s+ها(?=\s|$|[؟،.!])", "\u200cها"),       # کتاب ها → کتاب‌ها
    (r"(?<=" + _PERSIAN_CHAR + r")\s+هایی(?=\s|$)", "\u200cهایی"),           # آدم هایی → آدم‌هایی
    (r"(?<=" + _PERSIAN_CHAR + r")\s+هایش(?=\s|$)", "\u200cهایش"),           # دست هایش → دست‌هایش
    (r"(?<=" + _PERSIAN_CHAR + r")\s+هایم(?=\s|$)", "\u200cهایم"),
    (r"(?<=" + _PERSIAN_CHAR + r")\s+هایت(?=\s|$)", "\u200cهایت"),
    (r"(?<=" + _PERSIAN_CHAR + r")\s+هایمان(?=\s|$)", "\u200cهایمان"),
    (r"(?<=" + _PERSIAN_CHAR + r")\s+هایشان(?=\s|$)", "\u200cهایشان"),
    (r"(?<=" + _PERSIAN_CHAR + r")\s+ترین(?=\s|$|[؟،.!])", "\u200cترین"),   # بزرگ ترین → بزرگ‌ترین
    (r"(?<=" + _PERSIAN_CHAR + r")\s+تر(?=\s|$|[؟،.!])", "\u200cتر"),       # بزرگ تر → بزرگ‌تر
]


def fix_nim_fasele(text: str) -> str:
    """Insert ZWNJ (half-space) for common Persian prefix/suffix patterns."""
    for pattern, replacement in _PREFIX_PATTERNS:
        text = re.sub(pattern, replacement, text)
    for pattern, replacement in _SUFFIX_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


# -- Punctuation conversion --

_PUNCTUATION_MAP = {
    "?": "\u061f",  # → ؟
    ",": "\u060c",  # → ،
    ";": "\u061b",  # → ؛
}


def convert_punctuation(text: str) -> str:
    """Replace Latin punctuation with Persian equivalents."""
    for latin, persian in _PUNCTUATION_MAP.items():
        text = text.replace(latin, persian)
    return text


# -- Formal → conversational replacements --

_FORMAL_REPLACEMENTS = [
    ("می\u200cباشد", "هست"),
    ("نمی\u200cباشد", "نیست"),
    ("می باشد", "هست"),
    ("نمی باشد", "نیست"),
    ("بنابراین", "پس"),
    ("همچنین", "هم"),
    ("به عنوان مثال", "مثلاً"),
    ("لیکن", "ولی"),
    ("اما", "ولی"),
    ("بدین ترتیب", "این‌طوری"),
    ("علی\u200cرغم", "با وجود"),
    ("به منظور", "برای"),
    ("مورد نیاز", "لازم"),
]


def simplify_formal(text: str) -> str:
    """Replace overly formal/literary phrases with conversational equivalents."""
    for formal, casual in _FORMAL_REPLACEMENTS:
        text = text.replace(formal, casual)
    return text


def normalize_whitespace(text: str) -> str:
    """Clean up stray whitespace while preserving newlines."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        # Collapse multiple spaces into one
        line = re.sub(r" {2,}", " ", line).strip()
        cleaned.append(line)
    return "\n".join(cleaned)
