# AI Subtitle Translator

A production-grade async subtitle translation system that translates SRT and ASS subtitle files using the OpenAI API, with a focus on Persian (Farsi) translation quality.

## Features

- **Context-aware translation** -- each chunk receives the previous chunk's tail as read-only context for continuity
- **Adaptive smart chunking** -- hybrid strategy combining time-gap splitting, density-based size limits, and dialogue integrity
- **Glossary support** -- inject a term dictionary for consistent translation of names and terms
- **Persian post-processing** -- automatic nim-fasele (half-space) fixes, punctuation conversion, and formal-to-conversational normalization
- **Multi-line handling** -- joins multi-line subtitles before translation, restores structure after
- **Subtitle compression** -- prompt-level guidance to keep translations concise for reading speed
- **Refinement pass** -- optional second API call to improve fluency and naturalness
- **Correction retry** -- on invalid JSON, sends a correction prompt instead of blindly retrying
- **Persistent caching** -- avoids re-translating identical lines; optionally saves cache to disk
- **Resume / retry failed chunks** -- a per-file status file records failed chunks so re-running on the same file retries only those; it is deleted automatically once everything is translated
- **Multi-provider routing** -- define several providers/accounts in a JSON file and route across them (failover or round-robin), switching on rate-limit/quota errors or your own per-minute/per-day caps
- **Async & parallel** -- concurrent API calls with configurable semaphore limit
- **Flexible config** -- all settings via `.env`, CLI flags, or both (CLI takes priority)
- **Custom endpoints** -- works with any OpenAI-compatible API

## Project Structure

```
.
├── main.py                             # CLI entry point
├── web.py                              # Web UI launcher (uvicorn)
├── .env.sample                         # Environment config template
├── glossary.sample.json                # Example glossary file
├── providers.sample.json               # Example multi-provider routing file
├── requirements.txt
├── ai_subtitle_translator/
│   ├── __init__.py
│   ├── config.py                       # Dataclass configs, loads from .env
│   ├── parser.py                       # SRT/ASS parsing and subtitle document model
│   ├── chunker.py                      # Adaptive hybrid chunking + context windows
│   ├── translator.py                   # Async OpenAI translation with retry
│   ├── glossary.py                     # Glossary loading and prompt injection
│   ├── postprocess.py                  # Persian text normalization pipeline
│   ├── cache.py                        # Translation cache with persistence
│   ├── resume.py                       # Resume / retry-failed-chunks status files
│   ├── providers.py                    # Multi-provider routing (rate-limit/quota aware)
│   └── merger.py                       # Merge & write SRT/ASS output
└── webapp/                             # Web UI (FastAPI + SSE, reuses the pipeline)
    ├── server.py                       # Routes: page, translate, SSE, download
    ├── jobs.py                         # In-memory job registry + orchestration
    ├── models.py                       # API response models
    ├── templates/index.html            # Upload page (server-rendered)
    └── static/                         # app.js (fetch + EventSource), style.css
```

## Requirements

- Python 3.11+

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy the sample env file and fill in your values:

```bash
cp .env.sample .env
```

### `.env` variables

| Variable | Description | Default |
|---|---|---|
| `PROVIDER` | Provider backend: `copilot` or `codex` | `copilot` |
| `OPENAI_API_KEY` | Your OpenAI API key | -- |
| `OPENAI_BASE_URL` | API base URL (for proxies or compatible APIs) | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Model to use | `gpt-4o-mini` |
| `OPENAI_TEMPERATURE` | Sampling temperature | `0.3` |
| `OPENAI_API_MODE` | API surface for `copilot`: `auto`, `chat`, or `responses` (ignored for `codex`) | `auto` |
| `OPENAI_SEND_TEMPERATURE` | Send the temperature param (set `false` for models that reject it) | `true` |
| `TARGET_LANGUAGE` | Target translation language | `Persian (Farsi)` |
| `ENABLE_REFINEMENT` | Enable second-pass quality improvement | `false` |
| `ENABLE_POSTPROCESS` | Enable Persian post-processing | `true` |
| `GLOSSARY_PATH` | Path to glossary JSON file (empty to disable) | -- |
| `CACHE_PATH` | Path to cache JSON file (empty to disable) | -- |
| `MAX_CONCURRENCY` | Max parallel API calls | `5` |
| `MAX_RETRIES` | Retry attempts per chunk | `3` |
| `RETRY_BASE_DELAY` | Base delay for exponential backoff (seconds) | `1.0` |
| `CHUNK_MAX_LINES` | Max subtitle lines per chunk | `18` |
| `CHUNK_MAX_CHARS` | Max characters per chunk | `1500` |
| `CHUNK_TIME_GAP_MS` | Time gap to split dialogues (milliseconds) | `2500` |
| `CHUNK_CONTEXT_LINES` | Lines from previous chunk sent as context | `3` |
| `ENFORCE_CPS` | Enforce per-line reading-speed budget (compresses overruns) | `true` |
| `CPS_TARGET` | Target characters-per-second for the translated language | `13.0` |
| `CPS_MIN_CHARS` | Minimum char budget for very short cues | `20` |
| `CPS_TOLERANCE` | Multiplier before flagging a line as over budget | `1.2` |
| `TRANSLATE_CUES` | Translate `[sound]` and `(action)` cues | `true` |
| `TRANSLATE_LYRICS` | Translate `♪ lyrics ♪` lines | `true` |
| `ENABLE_MEMORY` | Maintain a rolling story summary across chunks (bypasses cache) | `false` |
| `MEMORY_UPDATE_INTERVAL` | Update the rolling summary every N chunks | `5` |
| `AUTO_PROBE` | Run a one-shot register/tone probe before translation | `false` |
| `REGISTER_OVERRIDE` | Manual film-context description (skips `AUTO_PROBE`) | -- |
| `AUTO_GLOSSARY` | Auto-discover proper nouns + recurring terms and propose translations | `false` |
| `AUTO_GLOSSARY_MIN_OCCURRENCES` | Minimum occurrences in source for a term to be a candidate | `3` |
| `ENFORCE_GLOSSARY` | Re-translate lines that omit a required glossary term | `true` |

### Providers

Two OpenAI-compatible backends are supported, selected via `PROVIDER` / `--provider`:

- **`copilot`** (default) — uses `chat.completions`, falling back to the non-streaming Responses API per `OPENAI_API_MODE`. Works with the real OpenAI API and most compatible proxies.
- **`codex`** — uses the **streaming-only** Responses API (`stream:true`, list-shaped input). For proxies that expose models this way. `OPENAI_API_MODE` is ignored; set `OPENAI_SEND_TEMPERATURE=false` for models that reject an explicit temperature.

```bash
# codex example
PROVIDER=codex OPENAI_BASE_URL=http://localhost:3001/v1 \
  python main.py movie.srt -m codex/gpt-5.5
```

## Usage

```bash
# Basic SRT -- output writes to input.fa.srt
python main.py movie.srt

# Basic ASS -- output writes to input.fa.ass
python main.py episode.ass

# Specify output path
python main.py movie.srt movie_fa.srt

# Use a different model and language
python main.py movie.srt -m gpt-4o -l "Turkish"

# Model that supports chat.completions with temperature (e.g. gpt-5-mini)
python main.py movie.srt -m gpt-5-mini --api-mode chat

# Model that only works via the Responses API and rejects temperature (e.g. gpt-5.4-mini)
python main.py movie.srt -m gpt-5.4-mini --api-mode responses --no-temperature

# Use a glossary for consistent names
python main.py movie.srt --glossary glossary.json

# Enable refinement pass (higher quality, 2x API calls)
python main.py movie.srt --refine

# Persistent cache (reuse across runs)
python main.py movie.srt --cache .translation_cache.json

# Resume: re-run the same file to retry only the chunks that failed last time
python main.py movie.srt            # auto: resumes if a status file exists
python main.py movie.srt --mode retry_failed   # only retry failed chunks (status file required)
python main.py movie.srt --mode fresh          # ignore any status file, translate everything

# Multiple providers / accounts (see "Multiple providers" below)
python main.py movie.srt --providers providers.json

# Custom API endpoint
python main.py movie.srt --base-url https://my-proxy.example.com/v1

# Increase parallelism
python main.py movie.srt --concurrency 10

# Disable Persian post-processing
python main.py movie.srt --no-postprocess

# Full example
python main.py movie.srt output.srt \
  -m gpt-4o \
  --glossary glossary.json \
  --refine \
  --cache .cache.json \
  --concurrency 8
```

### CLI Options

| Flag | Description |
|---|---|
| `input` | Path to the input subtitle file (`.srt` or `.ass`) |
| `output` | (Optional) Output path, defaults to `<input>.fa.<ext>` |
| `--provider` | Provider backend: `copilot` or `codex` |
| `--providers` | Path to a JSON providers file for multi-provider routing (overrides `--provider`/`-m`) |
| `-l`, `--language` | Target language |
| `-m`, `--model` | Model name |
| `--api-key` | OpenAI API key |
| `--base-url` | OpenAI-compatible API base URL |
| `--api-mode` | API surface: `auto`, `chat`, or `responses` |
| `--no-temperature` | Don't send the temperature parameter |
| `--glossary` | Path to glossary JSON file |
| `--refine` | Enable refinement pass |
| `--no-postprocess` | Disable Persian post-processing |
| `--concurrency` | Max concurrent API calls |
| `--cache` | Path to persistent cache JSON file |
| `--mode` | Resume mode: `auto` (default), `fresh`, or `retry_failed` |
| `--status-dir` | Directory for status files (default: `.translation-status/` next to the source) |
| `--max-lines` | Max subtitle lines per chunk |
| `--max-chars` | Max characters per chunk |
| `--context-lines` | Context lines from previous chunk |

CLI flags override `.env` values when provided.

## Web UI

A browser front-end is included for translating files without the terminal. It
reuses the same translation pipeline and streams live per-chunk progress over
Server-Sent Events.

```bash
pip install -r requirements.txt   # includes the optional web deps
python web.py                     # serve on http://127.0.0.1:8000
python web.py --port 9000         # custom port
python web.py --host 0.0.0.0      # bind all interfaces (also via $HOST/$PORT)
python web.py --reload            # auto-reload on code changes (development)
```

Open the page, upload a `.srt` or `.ass` file, choose the target language,
model, and provider (the form is pre-filled from your `.env`), then click
**Translate**. A progress bar advances as chunks complete; when finished, a
download button serves the translated `*.fa.*` file. API keys default to your
`.env` but can be overridden per-run in the "Advanced" panel.

Endpoints (all under the same app):

| Route | Purpose |
|---|---|
| `GET /` | Upload page |
| `POST /api/translate` | Start a job (multipart: file + options) → `{job_id}` |
| `GET /api/jobs/{id}/events` | SSE stream of live progress |
| `GET /api/jobs/{id}` | JSON status snapshot |
| `GET /api/jobs/{id}/download` | Translated subtitle file |

Job state is kept in memory for the lifetime of the process, which suits a local
single-user tool. The MVP exposes the core flow (upload → translate → download);
advanced CLI features (glossary editor, multi-provider routing, resume UI,
chunking/CPS knobs) use sensible defaults and are easy follow-ups.

## How It Works

1. **Parse** -- reads the SRT or ASS file into structured subtitle objects (id, timestamps, text)
2. **Chunk** -- groups subtitles using an adaptive hybrid strategy:
   - Splits on dialogue boundaries (time gaps > 2.5s)
   - Adapts size limits based on text density (short lines get bigger chunks, dense lines get smaller)
   - Enforces max lines and characters per chunk
3. **Context** -- builds a read-only context window (last N subtitles of previous chunk) for each chunk
4. **Translate** -- sends chunks to the OpenAI API in parallel with:
   - Previous context injected in the prompt (not re-translated)
   - Glossary terms injected for consistency
   - Per-line speaker tag (from ASS `Name` field) so register stays consistent per character
   - Per-line `max_chars` budget derived from on-screen duration × CPS target
   - Line kind tag (dialog / sound_cue / stage_dir / screen_text / lyrics) with kind-specific instructions
   - Smart retry: on invalid JSON, sends a correction prompt to the model
   - CPS retry: if any line exceeds its char budget, sends one compression request for the offenders
   - Optional **two-step refinement** (`--refine` / `ENABLE_REFINEMENT=true`): the model first critiques each translation, then revises only the items it flagged. Empty critique short-circuits the revise call.
   - ASS inline-style preservation: `{\i1}`, `{\b1}`, `{\pos(...)}` etc. are stripped before translation and restored onto the translated text (inline tags positioned proportionally, positional tags reattached at line start)
   - Optional one-shot **register probe** (`AUTO_PROBE=true`) detects genre + tone from a sample and injects it into every system prompt; `REGISTER_OVERRIDE` skips the probe
   - Optional **rolling story summary** (`ENABLE_MEMORY=true`) keeps character/context continuity across long files — chunks process in batches of `MEMORY_UPDATE_INTERVAL`, summary updates between batches, translation cache is bypassed while active
   - Optional **auto-glossary** (`AUTO_GLOSSARY=true` or `--auto-glossary`) discovers recurring proper nouns / terms in the source, asks the provider to propose translations, and merges them with any `--glossary` entries (user entries win)
   - **Glossary compliance** (`ENFORCE_GLOSSARY=true`, default on): after the main translation, any line whose source contains a glossary term but whose translation omits the mapped value triggers one targeted retry
5. **Post-process** -- Persian-specific normalization:
   - Half-space (nim-fasele) insertion for prefixes/suffixes
   - Latin-to-Persian punctuation conversion
   - Formal-to-conversational phrase simplification
6. **Multi-line** -- joins multi-line text before translation, restores line structure after
   - Restore prefers Persian/Latin clause boundaries (`،؛.,;:`) for split points; falls back to even word-count distribution
   - SRT line breaks are preserved as real newlines
   - ASS line breaks are decoded from `\N` / `\n` and restored on write
7. **Cache** -- caches source→translated pairs; optionally persists between runs
8. **Merge** -- deduplicates, reassembles, and writes a valid SRT or ASS file with original timing preserved
9. **Resume** -- only the chunks that aren't already translated are sent to the API. A status file records each chunk's status, source/translated text, attempts, and last error, and is written atomically after every chunk so progress survives a crash. When all chunks succeed the status file is deleted; if any fail it remains for a later resume. See **Resume & retrying failed chunks** below.

### Resume & retrying failed chunks

When a run leaves one or more failed chunks, the failed chunk keeps its original (source-language) text in the output and the run saves a status file next to the source at `.translation-status/<source-hash>.<lang>.<fmt>.json`. Re-running on the same file:

- Matches the status file by a normalized source-content **SHA-256 hash** (line endings normalized to `\n`, BOM stripped) plus target language, format, and chunking version. If the file content changed, the hash differs and it is treated as a brand-new job.
- **Resumes** only the failed/pending chunks — already-translated chunks are never sent to the API again.
- Rebuilds the full output (translated text where available, source text otherwise) so the file is always usable.
- **Deletes** the status file automatically once every chunk is translated.

Modes (`--mode` / `RESUME_MODE`):

| Mode | Behavior |
|---|---|
| `auto` (default) | Resume a matching status file if present, otherwise translate fresh |
| `fresh` | Ignore (and clear) any status file and translate everything |
| `retry_failed` | Require an existing status file and retry only failed/pending chunks |

### Multiple providers

If your accounts have rate limits or daily caps, list several providers in a JSON file and pass `--providers providers.json` (or set `PROVIDERS_PATH`). This **overrides** the single `--provider`/`-m`/`OPENAI_*` settings. Each request picks a provider and switches to the next when the current one returns a rate-limit/quota error **or** reaches a cap you configured. Usage is tracked in memory for the run and reset on the next run (matching daily-quota rollover). Copy `providers.sample.json` to start.

```json
{
  "strategy": "failover",
  "providers": [
    { "name": "codex-acct1", "provider": "codex", "model": "m1",
      "base_url": "http://localhost:3001/v1", "api_key": "sk-acct1",
      "send_temperature": false,
      "limits": { "requests_per_minute": 60, "requests_per_day": 1000, "concurrency": 3, "cooldown_seconds": 60 } },
    { "name": "copilot-acct2", "provider": "copilot", "model": "m2", "api_mode": "chat",
      "api_key": "sk-acct2",
      "limits": { "requests_per_minute": 30, "requests_per_day": 500, "concurrency": 2, "cooldown_seconds": 120 } }
  ]
}
```

- **`strategy`** — `failover` (use provider #1 until it's limited, then #2, …) or `round_robin` (rotate to spread load and maximize combined throughput).
- **Per-provider fields** — `name` (unique), `provider` (`copilot`|`codex`), `model` (required), `api_key`/`base_url` (fall back to `OPENAI_API_KEY`/`OPENAI_BASE_URL`), `api_mode` (copilot only), `send_temperature` (defaults: copilot `true`, codex `false`).
- **`limits`** — `requests_per_minute` / `requests_per_day` (`0` = unlimited), `concurrency` (per-provider in-flight cap), `cooldown_seconds` (how long to sideline a provider after a rate-limit error; a `Retry-After` header is honored when present).
- **Throughput note** — with `round_robin`, set `MAX_CONCURRENCY` ≥ the sum of the providers' `concurrency` values, otherwise the global concurrency cap throttles you below both accounts' combined capacity.

## Architecture Notes

### End-to-end flow

`main.py` coordinates a linear pipeline:

1. `parse_subtitle_file()` converts the input file into `Subtitle` dataclass objects
2. `chunk_subtitles()` groups subtitles into translation-sized chunks
3. `build_context_window()` attaches previous-chunk context for continuity
4. `find_existing_translation_status()` (from `resume.py`) looks for a resumable status file; only the unfinished chunks become translation targets
5. `Translator.translate_chunks_detailed()` calls the OpenAI API concurrently for the target chunks, persisting status after each via a progress callback
6. `build_final_subtitle_from_status()` rebuilds chunks (translated where available, else source); `merge_chunks()` flattens them back into subtitle order
7. `write_subtitle_file()` writes the final subtitle file in the original format; the status file is deleted if complete, kept otherwise

### Module responsibilities

- `main.py`: CLI parsing, config overrides, pipeline orchestration, logging
- `config.py`: `.env` loading and runtime dataclasses for chunking + translation settings
- `parser.py`: format-aware parsing, `Subtitle` / `SubtitleDocument` models, ASS dialogue extraction
- `chunker.py`: time-gap splitting, adaptive size limits, previous-chunk context windows
- `translator.py`: prompt construction, OpenAI API calls (chat.completions with Responses fallback), retries, correction prompts, refinement pass, multiline restoration, cache usage
- `postprocess.py`: Persian-specific normalization after translation
- `cache.py`: in-memory translation cache with optional JSON persistence
- `glossary.py`: glossary loading and prompt injection
- `merger.py`: dedupe, ordering, and SRT/ASS formatting/writing
- `resume.py`: source/chunk hashing, the translation-status model, status-file path/load/atomic-save/delete, and final-output rebuild for resume
- `providers.py`: multi-provider routing — providers-file loader/validator, per-provider in-memory limit/cooldown state, rate-limit error classification, and the `RoutingProvider` that fails over / round-robins across backends

### Best places to change behavior

- Change prompt wording or output rules: `ai_subtitle_translator/translator.py`
- Alter the API request shape (chat.completions / Responses): `ai_subtitle_translator/translator.py`
- Tune chunk sizes or boundary logic: `ai_subtitle_translator/chunker.py`
- Adjust Persian cleanup rules: `ai_subtitle_translator/postprocess.py`
- Add new config/env settings: `ai_subtitle_translator/config.py` and `main.py`
- Change SRT/ASS parsing or output formatting: `parser.py` and `merger.py`

### Current implementation constraints

- Multi-line subtitles are flattened before translation and rebuilt heuristically afterward
- Cache keys are raw source subtitle text, so identical source lines reuse the same translation
- Post-processing currently runs whenever target language contains `persian` or `farsi`
- If a chunk fails permanently, the original source text is kept for that chunk instead of aborting the whole run, and the failure is recorded in a status file for later resume (removed once everything is translated)
- A chunk is counted as failed (and recorded for resume) when it errors after retries, comes back empty, changes the subtitle count, or returns text identical to the source across all translatable lines
- JSON response validation checks item shape and warns on item-count mismatch, but still uses the model output when possible
- For ASS files, only dialogue text is translated; sections, styles, and non-dialogue event lines are preserved

## Glossary

Create a JSON file mapping source terms to their required translations:

```json
{
  "John": "جان",
  "Netflix": "نتفلیکس",
  "OK": "باشه"
}
```

Pass it via `--glossary glossary.json` or set `GLOSSARY_PATH=glossary.json` in `.env`.

## License

MIT
