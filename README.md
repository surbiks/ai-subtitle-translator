# AI Subtitle Translator

A high-performance async tool that translates SRT subtitle files into Persian (Farsi) using the OpenAI API.

## Features

- **Smart chunking** -- groups subtitles by dialogue timing, size limits, and context overlap for higher translation quality
- **Async & parallel** -- concurrent API calls with configurable semaphore limit
- **Retry with backoff** -- automatic exponential backoff on API failures
- **In-memory caching** -- skips API calls for repeated subtitle lines
- **Flexible config** -- all settings configurable via `.env` file, CLI flags, or both (CLI takes priority)
- **Custom endpoints** -- works with any OpenAI-compatible API (base URL + API key)

## Project Structure

```
.
├── main.py                             # CLI entry point
├── .env.sample                         # Environment config template
├── requirements.txt
└── ai_subtitle_translator/
    ├── __init__.py
    ├── config.py                       # Dataclass configs, loads from .env
    ├── parser.py                       # SRT file parsing
    ├── chunker.py                      # Hybrid chunking strategy
    ├── translator.py                   # Async OpenAI translation
    └── merger.py                       # Merge & write SRT output
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
| `OPENAI_API_KEY` | Your OpenAI API key | -- |
| `OPENAI_BASE_URL` | API base URL (for proxies or compatible APIs) | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Model to use | `gpt-4o-mini` |
| `OPENAI_TEMPERATURE` | Sampling temperature | `0.3` |
| `MAX_CONCURRENCY` | Max parallel API calls | `5` |
| `MAX_RETRIES` | Retry attempts per chunk | `3` |
| `RETRY_BASE_DELAY` | Base delay for exponential backoff (seconds) | `1.0` |
| `CHUNK_MAX_LINES` | Max subtitle lines per chunk | `18` |
| `CHUNK_MAX_CHARS` | Max characters per chunk | `1500` |
| `CHUNK_TIME_GAP_MS` | Time gap to split dialogues (milliseconds) | `2500` |
| `CHUNK_OVERLAP_LINES` | Overlap lines between chunks for context | `2` |

## Usage

```bash
# Basic -- output writes to input.fa.srt
python main.py movie.srt

# Specify output path
python main.py movie.srt movie_fa.srt

# Use a different model
python main.py movie.srt -m gpt-4o

# Custom API endpoint
python main.py movie.srt --base-url https://my-proxy.example.com/v1

# Increase parallelism
python main.py movie.srt --concurrency 10

# Adjust chunking
python main.py movie.srt --max-lines 25 --max-chars 2000

# Disable overlap between chunks
python main.py movie.srt --no-overlap
```

### CLI Options

| Flag | Description |
|---|---|
| `input` | Path to the input SRT file |
| `output` | (Optional) Output path, defaults to `<input>.fa.srt` |
| `-m`, `--model` | OpenAI model |
| `--api-key` | OpenAI API key |
| `--base-url` | OpenAI-compatible API base URL |
| `--concurrency` | Max concurrent API calls |
| `--max-lines` | Max subtitle lines per chunk |
| `--max-chars` | Max characters per chunk |
| `--overlap` | Overlap lines between chunks (0 to disable) |
| `--no-overlap` | Disable chunk overlap |

CLI flags override `.env` values when provided.

## How It Works

1. **Parse** -- reads the SRT file into structured subtitle objects (id, timestamps, text)
2. **Chunk** -- groups subtitles using a hybrid strategy:
   - Splits on dialogue boundaries (time gaps > 2.5s)
   - Enforces size limits (max lines and characters per chunk)
   - Adds overlap between chunks so the model has context across boundaries
3. **Translate** -- sends chunks to OpenAI in parallel with concurrency control, retries, and caching
4. **Merge** -- deduplicates overlap, reassembles translated subtitles, and writes a valid SRT file with original timestamps preserved

## License

MIT
