# Translation Quality Improvements — Implementation Plan

This document is an implementation reference, not a finished spec. Each item lists files touched, concrete code/prompt sketches, edge cases, and a verification checklist. Use it as a checklist when opening PRs.

**Phasing summary:**

| Phase | Items | Goal |
|---|---|---|
| 1 | Speaker detection · CPS budget · Non-dialog cues | Same-chunk quality wins, no new state |
| 2 | Rolling summary · Register probe | Run-level memory & tone consistency |
| 3 | Auto-glossary · Glossary validation | Lifecycle around terminology |
| 4 | Better multi-line · Self-critique · ASS styling · Length re-translation | Polish |

Each phase is shippable on its own.

---

## Phase 1 — Same-chunk improvements

### 1.1 Speaker / dialog-turn detection

**Goal:** Tag each subtitle with a `speaker` identifier so the model can maintain consistent register (شما vs تو, formal vs informal) per character across the whole file.

**Files touched:**
- `ai_subtitle_translator/parser.py` — extract speaker from ASS `Name` field; detect `- `/`– `/`— ` dash prefixes in SRT
- `ai_subtitle_translator/translator.py` — include `speaker` in payload; update system prompt
- `ai_subtitle_translator/chunker.py` — avoid splitting a question/answer pair across chunks

**Data model change** (`parser.py`):

```python
@dataclass
class Subtitle:
    id: int
    start: str
    end: str
    text: str
    speaker: str | None = None          # NEW
    metadata: dict[str, Any] | None = None
```

**Extraction logic:**

```python
# ASS: data["name"] is already parsed in _parse_ass_dialogue_line — just keep it
speaker = (data.get("name") or "").strip() or None

# SRT: detect dash-prefixed turns inside a single subtitle text
_SRT_TURN_RE = re.compile(r"^\s*[-–—]\s*", re.MULTILINE)

def _detect_srt_speakers(text: str) -> tuple[str | None, bool]:
    """Returns (single_speaker_label, has_multiple_speakers)."""
    turn_count = len(_SRT_TURN_RE.findall(text))
    if turn_count >= 2:
        return None, True          # multi-speaker within one cue
    return None, False             # SRT has no per-line speaker labels
```

For SRT we can't recover *names*; we only know there's a speaker change. That's still useful — pass `speaker_change: true` so the model holds register per turn within the line.

**Payload update** (`translator.py`):

```python
payload = []
for s in chunk:
    item = {"id": s.id, "text": s.text.replace("\n", " ")}
    if s.speaker:
        item["speaker"] = s.speaker
    payload.append(item)
```

**System prompt addition:**

```
SPEAKER CONSISTENCY:
- Items may include a "speaker" field (character name) or be marked with
  speaker_change for multi-turn lines.
- Maintain consistent register (formal/informal) per speaker across the
  entire run. Pick register from relationship cues (boss/employee → formal,
  friends → informal, parent → child → mixed).
- Never switch register for the same speaker mid-conversation unless the
  source text clearly signals it.
```

**Chunker guard** (`chunker.py`):

In `_split_by_size`, before forcing a split, check: if the *current last item* ends with `?`/`؟` and the next item starts with a dash, treat them as one logical unit and don't split between them. Implement as a "would-split-mid-turn" override that allows exceeding `max_lines` by 1.

**Edge cases:**
- ASS files with `Name=""` for every line → falls back to no-speaker mode (current behavior). Don't error.
- SRT with stray dashes that aren't speaker indicators (e.g. `— well, you see —`) — the regex matches at line start only, so mid-line dashes are ignored.
- Speaker names with non-ASCII chars (e.g. `جان`) — preserve as-is.

**Verification:**
- [ ] ASS file with `Name` field round-trips speaker info into prompt
- [ ] SRT file with `- Hello\n- Hi` produces `speaker_change: true`
- [ ] Translation of a long dialog (>50 lines) shows consistent register per speaker — spot-check first/middle/end
- [ ] No regression for files without speaker info

---

### 1.2 Reading-speed (CPS) budget

**Goal:** Give the model a per-line character budget derived from each subtitle's display duration, so Persian translations stop overrunning on-screen time.

**Files touched:**
- `ai_subtitle_translator/config.py` — add `cps_target` setting (default 13 CPS for Persian, configurable per language)
- `ai_subtitle_translator/translator.py` — compute `max_chars` per line, inject into payload, add length-validation step
- `ai_subtitle_translator/parser.py` — already exposes `start_ms`/`end_ms`

**Config:**

```python
# config.py — TranslatorConfig
cps_target: float = _env_float("CPS_TARGET", 13.0)
cps_min_chars: int = _env_int("CPS_MIN_CHARS", 20)  # floor for very short cues
enforce_cps: bool = _env_bool("ENFORCE_CPS", True)
```

**Budget computation** (`translator.py`):

```python
def _char_budget(sub: Subtitle, cps: float, min_chars: int) -> int:
    duration_sec = max(0.5, (sub.end_ms - sub.start_ms) / 1000.0)
    return max(min_chars, int(duration_sec * cps))
```

**Payload update:**

```python
{"id": s.id, "max_chars": _char_budget(s, cfg.cps_target, cfg.cps_min_chars), "text": ...}
```

**System prompt addition:**

```
LENGTH BUDGET:
- Each item has "max_chars". The translation MUST NOT exceed it.
- Compress aggressively: drop fillers ("you know", "well", "I mean"),
  contract clauses, use shorter synonyms.
- If the source is verbose but the budget is tight, paraphrase to fit.
  Meaning preservation > literal preservation.
```

**Post-translation validation:**

```python
def _validate_budget(orig: Subtitle, translated: str, budget: int) -> bool:
    # Strip ZWNJ and whitespace from the count — they're free in rendering
    visible = translated.replace("‌", "").strip()
    return len(visible) <= budget
```

Collect overruns from the whole chunk and send ONE retry request:

```
The following translations exceed their max_chars budget. Re-translate them
within budget, preserving meaning. Output JSON array of {id, text}:
[{"id": 5, "max_chars": 42, "current": "...", "current_length": 58}, ...]
```

**Edge cases:**
- Very short cues (< 1 sec): floor at `cps_min_chars` so we don't demand 5-char translations.
- Multi-line source: budget applies to total chars (excluding newline). The line break itself is free.
- Lines that are 100% glossary terms (e.g. just a name): may legitimately exceed budget; allow `1.2x` overrun before flagging.
- CPS for non-Persian targets: 17 for English, 13 for Persian, 10 for languages with longer words (German, Finnish). Keep configurable.

**Verification:**
- [ ] Compute budget for a known subtitle and assert `len(translated) <= budget` for 95%+ of lines
- [ ] Re-translation pass reduces overruns to <5%
- [ ] `--no-enforce-cps` flag (or `ENFORCE_CPS=false`) reverts to old behavior

---

### 1.3 Non-dialog cue detection

**Goal:** Classify lines as dialog, sound cue, screen text, or lyrics so each gets translated (or preserved) appropriately.

**Files touched:**
- New file: `ai_subtitle_translator/classify.py`
- `ai_subtitle_translator/translator.py` — route by class
- `ai_subtitle_translator/config.py` — `translate_cues`, `translate_lyrics` flags

**Classifier** (`classify.py`):

```python
from enum import Enum
import re

class LineKind(str, Enum):
    DIALOG = "dialog"
    SOUND_CUE = "sound_cue"      # [music], [door slams]
    STAGE_DIR = "stage_dir"      # (whispering), (laughing)
    SCREEN_TEXT = "screen_text"  # ALL CAPS titles, signs
    LYRICS = "lyrics"            # ♪ ... ♪

_SOUND_RE  = re.compile(r"^\s*\[.+\]\s*$", re.DOTALL)
_STAGE_RE  = re.compile(r"^\s*\(.+\)\s*$", re.DOTALL)
_LYRICS_RE = re.compile(r"^\s*[♪#].*[♪#]?\s*$|^\s*♪")
_CAPS_RE   = re.compile(r"^[A-Z0-9\s\.\,\!\?\-']{4,}$")

def classify(text: str) -> LineKind:
    stripped = text.strip()
    if _LYRICS_RE.match(stripped):
        return LineKind.LYRICS
    if _SOUND_RE.match(stripped):
        return LineKind.SOUND_CUE
    if _STAGE_RE.match(stripped):
        return LineKind.STAGE_DIR
    if _CAPS_RE.match(stripped) and len(stripped) > 3:
        return LineKind.SCREEN_TEXT
    return LineKind.DIALOG
```

**Routing** (`translator.py`):

```python
# Group chunk items by kind
groups = {kind: [] for kind in LineKind}
for s in chunk:
    groups[classify(s.text)].append(s)

# Translate each non-empty group with its tailored prompt,
# preserving id alignment via the by-id lookup added earlier.
```

**Per-class prompts:**

| Class | Prompt addition |
|---|---|
| `DIALOG` | (default prompt) |
| `SOUND_CUE` | "Translate to Persian equivalent: `[music]` → `[موسیقی]`, `[door slams]` → `[صدای کوبیده شدن در]`. Keep brackets." |
| `STAGE_DIR` | "Translate the parenthetical, keep parentheses." |
| `SCREEN_TEXT` | "Treat as on-screen text (sign, title, caption). Translate naturally without conversational softening." |
| `LYRICS` | "Translate poetically; preserve rhythm and ♪ markers." |

**Config:**

```python
translate_cues: bool = _env_bool("TRANSLATE_CUES", True)
translate_lyrics: bool = _env_bool("TRANSLATE_LYRICS", True)
```

When disabled, lines of that class pass through unchanged.

**Edge cases:**
- Mixed line: `MARK: Hello there` — starts with caps but isn't pure screen text. The `_CAPS_RE` requires the *whole* string to be caps; this would be `DIALOG`. Good.
- `[Inaudible]` should translate.
- Subtitle that is both a cue AND dialog (`[Mark, sighing] I don't know`) — falls into DIALOG class (no leading bracket on whole line). Acceptable for v1.

**Verification:**
- [ ] Unit tests for `classify()` covering each class
- [ ] Sample file with mixed classes round-trips correctly
- [ ] `TRANSLATE_CUES=false` leaves cues untouched
- [ ] No regression on pure-dialog files

---

## Phase 2 — Run-level state

### 2.1 Rolling story summary

**Goal:** Carry context across the whole file via a short evolving summary, fixing the "model forgets who anyone is after 20 minutes" problem.

**Files touched:**
- New file: `ai_subtitle_translator/memory.py`
- `ai_subtitle_translator/translator.py` — inject summary in user message; update summary every N chunks
- `ai_subtitle_translator/config.py` — `enable_memory`, `memory_update_interval`

**Memory helper:**

```python
class StoryMemory:
    def __init__(self) -> None:
        self.summary: str = ""

    def context_block(self) -> str:
        if not self.summary:
            return ""
        return f"STORY SO FAR (for character/context continuity):\n{self.summary}\n"

    async def update(self, provider, model, new_chunk_text: str) -> None:
        prompt = (
            f"Current summary:\n{self.summary or '(none)'}\n\n"
            f"New scene:\n{new_chunk_text}\n\n"
            "Update the summary to 2–3 sentences capturing characters, "
            "relationships, and current situation. Output only the new summary."
        )
        self.summary = (await provider.chat(...)).strip()
```

**Integration:**
- Run `await memory.update(...)` every `memory_update_interval` chunks (default: 5)
- Inject `memory.context_block()` into each chunk's user message above the "Previous context" block

**Config:**

```python
enable_memory: bool = _env_bool("ENABLE_MEMORY", False)
memory_update_interval: int = _env_int("MEMORY_UPDATE_INTERVAL", 5)
```

**Cost note:** Adds ~50 tokens to every translation prompt + one summary-update call every N chunks. For a 1000-chunk film with N=5, that's ~200 extra calls. Make it opt-in.

**Verification:**
- [ ] Summary populates after first update
- [ ] Translated character references stay consistent across chunks (manual spot-check at minutes 10, 60, 110)
- [ ] Disabling via env reverts to baseline behavior

---

### 2.2 Tone / register probe

**Goal:** Detect the film's genre, register, and tone once at the start so all prompts can be tuned to it.

**Files touched:**
- `ai_subtitle_translator/translator.py` — `_probe_register()` method run once before `translate_chunks`
- `ai_subtitle_translator/config.py` — `auto_probe`, `register_override`

**Implementation:**

```python
async def _probe_register(self, sample: list[Subtitle]) -> str:
    sample_text = "\n".join(s.text for s in sample[:20])
    response = await self._provider.chat(
        system="You are a translation prep assistant.",
        messages=[{"role": "user", "content":
            f"Given these subtitle lines:\n{sample_text}\n\n"
            "In 2 lines, describe: (1) genre/setting, (2) register "
            "(formal/conversational/period/slangy/poetic). Be specific."
        }],
        model=self._model,
        temperature=0.0,
    )
    return response.strip()
```

The result becomes a new section in the system prompt:

```
FILM CONTEXT (detected automatically):
{register_description}
```

**Edge cases:**
- Files <20 lines: probe on whatever's available.
- `--register "modern conversational, gangster slang"` overrides auto-detect.

**Verification:**
- [ ] Probe runs once, returns non-empty description
- [ ] System prompt includes it in subsequent calls (log first call's prompt to confirm)
- [ ] Manual override via env/flag works

---

## Phase 3 — Glossary lifecycle

### 3.1 Auto-extracted glossary

**Goal:** Discover proper nouns and recurring terms automatically; have the LLM propose translations; persist for the run.

**Files touched:**
- New file: `ai_subtitle_translator/discover.py`
- `ai_subtitle_translator/glossary.py` — merge auto-extracted with user-provided
- `main.py` — `--auto-glossary` flag

**Discovery algorithm:**

1. Tokenize all source text, extract capitalized words (excluding sentence starters — use a simple heuristic: capitalized AND not at position 0 of a sentence, OR appears capitalized mid-sentence elsewhere).
2. Frequency count. Keep terms appearing ≥3 times.
3. Filter common English words (small stoplist: `I`, `OK`, `Mr`, `Mrs`, `Dr`, etc.)
4. Send candidate list to the model: "Propose Persian translations for these names/terms. Output JSON {term: translation}."
5. Merge with user glossary (user entries win on conflict).

```python
def extract_candidates(subtitles: list[Subtitle]) -> list[str]:
    counter: dict[str, int] = {}
    pattern = re.compile(r"\b[A-Z][a-zA-Z']+\b")
    for s in subtitles:
        for m in pattern.finditer(s.text):
            term = m.group()
            if term not in _STOPLIST:
                counter[term] = counter.get(term, 0) + 1
    return [t for t, n in counter.items() if n >= 3]
```

**Verification:**
- [ ] Sample film yields a non-empty candidate list
- [ ] LLM-proposed translations look reasonable for names
- [ ] User glossary entry for a discovered term overrides the auto value

---

### 3.2 Glossary compliance validation

**Goal:** After translation, verify every glossary term that appeared in source produced its mapped translation. Re-request mismatches.

**Implementation sketch:**

```python
def check_compliance(orig: str, translated: str, glossary: Glossary) -> list[tuple[str, str]]:
    violations = []
    for term, expected in glossary.entries.items():
        if term in orig and expected not in translated:
            violations.append((term, expected))
    return violations
```

If violations exist for a chunk, send a targeted re-request:

```
Your translation did not use the required glossary terms:
- "John" must always translate to "جان"
Re-translate the items below with these terms used correctly:
[...payload of failing items...]
```

**Verification:**
- [ ] Test file with a glossary entry that the model is likely to translate differently — compliance check catches it, re-request fixes it.

---

## Phase 4 — Polish

### 4.1 Punctuation-aware multi-line restoration

Replace `_restore_multiline` in `translator.py`:

```python
def _restore_multiline(translated: str, original: str) -> str:
    n = original.count("\n") + 1
    if n <= 1:
        return translated
    # Prefer splitting on Persian/Latin clause boundaries
    candidates = [m.end() for m in re.finditer(r"[،؛.,;:]\s", translated)]
    if len(candidates) < n - 1:
        # fall back to word-count split (current behavior)
        return _split_by_words(translated, n)
    # Pick split points closest to even spacing
    target_lengths = [len(translated) * i // n for i in range(1, n)]
    chosen = []
    for target in target_lengths:
        best = min(candidates, key=lambda c: abs(c - target))
        chosen.append(best)
        candidates.remove(best)
    chosen.sort()
    lines = []
    prev = 0
    for split in chosen:
        lines.append(translated[prev:split].strip())
        prev = split
    lines.append(translated[prev:].strip())
    return "\n".join(line for line in lines if line)
```

**Verification:**
- [ ] Multi-line subtitle with clear punctuation produces clean break
- [ ] Multi-line subtitle without internal punctuation falls back gracefully

---

### 4.2 Self-critique refinement

Replace `_refine()`:

```python
async def _refine(self, chunk_index, items):
    # Step 1: critique
    critique = await self._provider.chat(
        system="You are a Persian subtitle editor. Identify specific issues only.",
        messages=[{"role": "user", "content":
            f"Find issues in these translations (naturalness, register, length, "
            f"glossary, awkward phrasing). Output JSON array of "
            f'{{"id": int, "issues": [string]}} for items with problems. '
            f"Empty array if all are fine.\n\n{json.dumps(items)}"
        }],
        model=self._model,
        temperature=0.0,
    )
    # Step 2: revise based on critique
    if "[]" in critique.strip():
        return items  # nothing to fix
    revised = await self._provider.chat(
        system="You revise Persian subtitles given specific criticism.",
        messages=[{"role": "user", "content":
            f"Original translations:\n{json.dumps(items)}\n\n"
            f"Criticism:\n{critique}\n\n"
            f"Output the revised JSON array, keeping the same ids and fixing "
            f"every flagged issue."
        }],
        model=self._model,
        temperature=self._temperature,
    )
    return _parse_response(revised, len(items))
```

---

### 4.3 ASS inline-style preservation

ASS `text` can contain `{\i1}italic{\i0}`, `{\b1}bold{\b0}`, position tags `{\pos(x,y)}`, etc. Currently passed verbatim to the model — likely mangled.

**Strategy:**
1. Strip ASS override tags before translation, replace with sentinels: `{\i1}hello{\i0}` → `⟨0⟩hello⟨1⟩`
2. Translate the cleaned text
3. Restore tags at sentinel positions

Use a small `ass_tags.py` helper:

```python
_ASS_TAG_RE = re.compile(r"\{[^}]+\}")

def strip_tags(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Returns (clean_text, [(index, tag), ...]) for restoration."""
    tags = []
    out = []
    pos = 0
    for m in _ASS_TAG_RE.finditer(text):
        out.append(text[pos:m.start()])
        tags.append((sum(len(s) for s in out), m.group()))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), tags

def restore_tags(translated: str, tags: list[tuple[int, str]]) -> str:
    # Best-effort: insert tags at proportionally-mapped positions
    if not tags:
        return translated
    ratio = len(translated) / max(1, tags[-1][0])  # rough scale
    result = list(translated)
    for orig_index, tag in reversed(tags):
        insert_at = min(len(result), int(orig_index * ratio))
        result.insert(insert_at, tag)
    return "".join(result)
```

**Edge case:** Position tags `{\pos(...)}` should stay at line start. Detect and keep separately.

---

### 4.4 Length-budget targeted re-translation

When CPS validation (Phase 1.2) finds overruns *after* the main translation + refinement pass, send one final compression pass for just the offending lines. See Phase 1.2 prompt above; this is the same mechanism extended to fire after the refinement pass too.

---

## Cross-cutting concerns

### Cache compatibility

| Phase | Cache impact |
|---|---|
| 1.1 Speaker | Cache key stays `source_text` → translation. Speaker context is lost on cache hit. Acceptable for v1. To fix: extend key to `(speaker, source_text)`. |
| 1.2 CPS | Same as 1.1 — cache hit ignores duration. Risk: a cached translation may exceed a shorter cue's budget. Mitigation: validate cached translations against current budget; on overrun, treat as cache miss. |
| 1.3 Cues | Cache stays valid (key is text). |
| 2.1 Summary | Translations depend on rolling summary, so cached hits are stale. Either disable cache when memory is enabled, or extend key to include a summary hash. |
| 2.2 Register | Run-scoped; cache stays valid within a run, stale across runs with different probe results. |

Recommendation: make cache *invalidation-aware* in Phase 2:

```python
def cache_key(text: str, ctx: tuple[str, ...]) -> str:
    if not ctx:
        return text
    return hashlib.sha1(("\x00".join((*ctx, text))).encode()).hexdigest()
```

### Test infrastructure

Currently zero tests. Add `tests/` with:
- `tests/test_parser.py` — SRT/ASS parsing, dot-separator, speaker extraction
- `tests/test_chunker.py` — time-gap splits, adaptive sizing, turn-preservation
- `tests/test_postprocess.py` — ZWNJ, punctuation guards, formal replacements
- `tests/test_classify.py` — line-kind detection
- `tests/test_budget.py` — CPS budget computation
- `tests/test_translator_offline.py` — mock provider; test alignment, retry, validation

Use `pytest` + `pytest-asyncio`. Add to `requirements.txt`:

```
pytest>=8.0
pytest-asyncio>=0.23
```

### Logging

Each new step should log at INFO:
- Speaker detection: `"Detected speakers: {set} from {n} subtitles"`
- CPS validation: `"Chunk {i}: {n} lines over budget, re-requesting"`
- Memory update: `"Updated story summary after chunk {i}"`
- Probe: `"Detected register: {text}"`

### Config sprawl

Phase 1–4 add ~15 new env vars. Mitigation:
- Group related flags into a single dataclass section in `config.py`
- Document each in `README.md` env table
- Provide sensible defaults so users can ignore most

### Rollout checklist per phase

For each phase, before merging:
- [ ] All new features behind opt-in flags OR backwards-compatible defaults
- [ ] Tests pass (`pytest tests/`)
- [ ] Manual end-to-end run on `tests/fixtures/sample.srt`
- [ ] AGENTS.md updated to reflect new modules
- [ ] README.md env table + usage examples updated
- [ ] `.env.sample` updated

---

## Recommended starting PR (Phase 1.1 + 1.2)

Smallest unit that produces measurable quality improvement:

1. `parser.py` — add `Subtitle.speaker`, populate from ASS `Name` and SRT dash patterns
2. `translator.py` — extend payload with `speaker` and `max_chars`; update system prompt; add post-translation CPS validation + one retry
3. `config.py` — add `cps_target`, `cps_min_chars`, `enforce_cps`
4. `.env.sample`, `README.md` — document new vars
5. `tests/test_budget.py`, `tests/test_parser.py` — basic coverage

Estimated diff: ~250 LOC added, ~30 changed.

Verification before merge:
- Translate a 30-min sample with and without the feature flags
- Compare: % of lines over CPS budget, manual quality spot-check on dialog register
- Confirm baseline behavior unchanged when `ENFORCE_CPS=false` and ASS file has no `Name` field
