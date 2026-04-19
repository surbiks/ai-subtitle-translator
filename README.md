# AI Subtitle Translator

A production-grade async subtitle translation system that translates SRT and ASS subtitle files using the OpenAI or Anthropic API, with a focus on Persian (Farsi) translation quality.

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
- **Async & parallel** -- concurrent API calls with configurable semaphore limit
- **Flexible config** -- all settings via `.env`, CLI flags, or both (CLI takes priority)
- **Multi-provider** -- supports OpenAI-compatible APIs and Anthropic Claude models
- **Custom endpoints** -- works with any OpenAI-compatible API

## Project Structure

```
.
├── main.py                             # CLI entry point
├── .env.sample                         # Environment config template
├── glossary.sample.json                # Example glossary file
├── requirements.txt
└── ai_subtitle_translator/
    ├── __init__.py
    ├── config.py                       # Dataclass configs, loads from .env
    ├── parser.py                       # SRT/ASS parsing and subtitle document model
    ├── chunker.py                      # Adaptive hybrid chunking + context windows
    ├── translator.py                   # Async translation with provider abstraction, retry
    ├── glossary.py                     # Glossary loading and prompt injection
    ├── postprocess.py                  # Persian text normalization pipeline
    ├── cache.py                        # Translation cache with persistence
    └── merger.py                       # Merge & write SRT/ASS output
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
| `PROVIDER` | API provider (`openai` or `anthropic`) | `openai` |
| `OPENAI_API_KEY` | Your OpenAI API key | -- |
| `OPENAI_BASE_URL` | API base URL (for proxies or compatible APIs) | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Model to use | `gpt-4o-mini` |
| `OPENAI_TEMPERATURE` | Sampling temperature | `0.3` |
| `ANTHROPIC_API_KEY` | Your Anthropic API key (when `PROVIDER=anthropic`) | -- |
| `ANTHROPIC_BASE_URL` | Anthropic API base URL (for proxies) | -- |
| `ANTHROPIC_MODEL` | Anthropic model to use | `claude-sonnet-4-20250514` |
| `ANTHROPIC_TEMPERATURE` | Anthropic sampling temperature | `0.3` |
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

# Use Anthropic Claude
python main.py movie.srt -p anthropic -m claude-sonnet-4-20250514

# Use Anthropic with explicit API key
python main.py movie.srt -p anthropic -m claude-sonnet-4-20250514 --anthropic-api-key sk-ant-...

# Use a glossary for consistent names
python main.py movie.srt --glossary glossary.json

# Enable refinement pass (higher quality, 2x API calls)
python main.py movie.srt --refine

# Persistent cache (reuse across runs)
python main.py movie.srt --cache .translation_cache.json

# Custom API endpoint
python main.py movie.srt --base-url https://my-proxy.example.com/v1

# Increase parallelism
python main.py movie.srt --concurrency 10

# Disable Persian post-processing
python main.py movie.srt --no-postprocess

# Full example (OpenAI)
python main.py movie.srt output.srt \
  -m gpt-4o \
  --glossary glossary.json \
  --refine \
  --cache .cache.json \
  --concurrency 8

# Full example (Anthropic)
python main.py movie.srt output.srt \
  -p anthropic \
  -m claude-sonnet-4-20250514 \
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
| `-p`, `--provider` | API provider (`openai` or `anthropic`) |
| `-l`, `--language` | Target language |
| `-m`, `--model` | Model name |
| `--api-key` | OpenAI API key |
| `--base-url` | OpenAI-compatible API base URL |
| `--anthropic-api-key` | Anthropic API key |
| `--anthropic-base-url` | Anthropic API base URL |
| `--anthropic-model` | Anthropic model |
| `--anthropic-temperature` | Anthropic sampling temperature |
| `--glossary` | Path to glossary JSON file |
| `--refine` | Enable refinement pass |
| `--no-postprocess` | Disable Persian post-processing |
| `--concurrency` | Max concurrent API calls |
| `--cache` | Path to persistent cache JSON file |
| `--max-lines` | Max subtitle lines per chunk |
| `--max-chars` | Max characters per chunk |
| `--context-lines` | Context lines from previous chunk |

CLI flags override `.env` values when provided.

## How It Works

1. **Parse** -- reads the SRT or ASS file into structured subtitle objects (id, timestamps, text)
2. **Chunk** -- groups subtitles using an adaptive hybrid strategy:
   - Splits on dialogue boundaries (time gaps > 2.5s)
   - Adapts size limits based on text density (short lines get bigger chunks, dense lines get smaller)
   - Enforces max lines and characters per chunk
3. **Context** -- builds a read-only context window (last N subtitles of previous chunk) for each chunk
4. **Translate** -- sends chunks to the selected provider (OpenAI or Anthropic) in parallel with:
   - Previous context injected in the prompt (not re-translated)
   - Glossary terms injected for consistency
   - Concise-translation guidance for subtitle readability
   - Smart retry: on invalid JSON, sends a correction prompt to the model
   - Optional refinement pass for fluency improvement
5. **Post-process** -- Persian-specific normalization:
   - Half-space (nim-fasele) insertion for prefixes/suffixes
   - Latin-to-Persian punctuation conversion
   - Formal-to-conversational phrase simplification
6. **Multi-line** -- joins multi-line text before translation, restores line structure after
   - SRT line breaks are preserved as real newlines
   - ASS line breaks are decoded from `\N` / `\n` and restored on write
7. **Cache** -- caches source→translated pairs; optionally persists between runs
8. **Merge** -- deduplicates, reassembles, and writes a valid SRT or ASS file with original timing preserved

## Architecture Notes

### End-to-end flow

`main.py` coordinates a linear pipeline:

1. `parse_subtitle_file()` converts the input file into `Subtitle` dataclass objects
2. `chunk_subtitles()` groups subtitles into translation-sized chunks
3. `build_context_window()` attaches previous-chunk context for continuity
4. `Translator.translate_chunks()` calls the configured LLM provider concurrently
5. `merge_chunks()` flattens translated chunks back into subtitle order
6. `write_subtitle_file()` writes the final subtitle file in the original format

### Module responsibilities

- `main.py`: CLI parsing, config overrides, pipeline orchestration, logging
- `config.py`: `.env` loading and runtime dataclasses for chunking + translation settings
- `parser.py`: format-aware parsing, `Subtitle` / `SubtitleDocument` models, ASS dialogue extraction
- `chunker.py`: time-gap splitting, adaptive size limits, previous-chunk context windows
- `translator.py`: prompt construction, provider abstraction, retries, correction prompts, refinement pass, multiline restoration, cache usage
- `postprocess.py`: Persian-specific normalization after translation
- `cache.py`: in-memory translation cache with optional JSON persistence
- `glossary.py`: glossary loading and prompt injection
- `merger.py`: dedupe, ordering, and SRT/ASS formatting/writing

### Best places to change behavior

- Change prompt wording or output rules: `ai_subtitle_translator/translator.py`
- Add a new provider or alter provider-specific request shape: `ai_subtitle_translator/translator.py`
- Tune chunk sizes or boundary logic: `ai_subtitle_translator/chunker.py`
- Adjust Persian cleanup rules: `ai_subtitle_translator/postprocess.py`
- Add new config/env settings: `ai_subtitle_translator/config.py` and `main.py`
- Change SRT/ASS parsing or output formatting: `parser.py` and `merger.py`

### Current implementation constraints

- Multi-line subtitles are flattened before translation and rebuilt heuristically afterward
- Cache keys are raw source subtitle text, so identical source lines reuse the same translation
- Post-processing currently runs whenever target language contains `persian` or `farsi`
- If a chunk fails permanently, the original source text is kept for that chunk instead of aborting the whole run
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
