#!/usr/bin/env python3
"""
Subtitle Translator — translates SRT subtitles to Persian using OpenAI.

Usage:
    python main.py input.srt output.srt
    python main.py input.srt                     # writes to input.fa.srt
    python main.py input.srt -m gpt-4o           # use a specific model
    python main.py input.srt --concurrency 10    # increase parallelism
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from ai_subtitle_translator.chunker import chunk_subtitles
from ai_subtitle_translator.config import AppConfig, ChunkConfig, TranslatorConfig
from ai_subtitle_translator.merger import merge_chunks, write_srt
from ai_subtitle_translator.parser import parse_srt
from ai_subtitle_translator.translator import Translator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai_subtitle_translator")


async def translate_file(
    input_path: str,
    output_path: str,
    config: AppConfig,
) -> None:
    """Full pipeline: parse → chunk → translate → merge → write."""
    t0 = time.perf_counter()

    # 1. Parse
    logger.info("Parsing %s", input_path)
    subtitles = parse_srt(input_path)
    logger.info("Parsed %d subtitles", len(subtitles))

    if not subtitles:
        logger.warning("No subtitles found — nothing to translate")
        return

    # 2. Chunk
    chunks = chunk_subtitles(subtitles, config.chunk)
    total_chars = sum(len(s.text) for s in subtitles)
    logger.info(
        "Created %d chunks (total %d chars across %d subtitles)",
        len(chunks), total_chars, len(subtitles),
    )

    # 3. Translate
    translator = Translator(config.translator)
    translated_chunks = await translator.translate_chunks(
        chunks, overlap=config.chunk.overlap_lines
    )

    # 4. Merge & write
    merged = merge_chunks(translated_chunks)
    write_srt(merged, output_path)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done — %d subtitles translated in %.1fs → %s",
        len(merged), elapsed, output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Translate SRT subtitles to Persian (Farsi) using OpenAI",
    )
    p.add_argument("input", help="Path to the input SRT file")
    p.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path for the translated SRT file (default: <input>.fa.srt)",
    )
    p.add_argument(
        "-m", "--model",
        default=None,
        help="OpenAI model (default: from .env or gpt-4o-mini)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (default: from .env or OPENAI_API_KEY env var)",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL (default: from .env)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max concurrent API calls (default: from .env or 5)",
    )
    p.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Max subtitle lines per chunk (default: from .env or 18)",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Max characters per chunk (default: from .env or 1500)",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Overlap lines between chunks (default: from .env or 2, 0 to disable)",
    )
    p.add_argument(
        "--no-overlap",
        action="store_true",
        help="Disable chunk overlap",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_suffix(".fa.srt"))

    overlap = 0 if args.no_overlap else args.overlap

    # Start from .env defaults, then override with any explicit CLI args
    chunk_cfg = ChunkConfig()
    if args.max_lines is not None:
        chunk_cfg.max_lines = args.max_lines
    if args.max_chars is not None:
        chunk_cfg.max_chars = args.max_chars
    if overlap is not None:
        chunk_cfg.overlap_lines = overlap

    translator_cfg = TranslatorConfig()
    if args.model is not None:
        translator_cfg.model = args.model
    if args.api_key is not None:
        translator_cfg.api_key = args.api_key
    if args.base_url is not None:
        translator_cfg.base_url = args.base_url
    if args.concurrency is not None:
        translator_cfg.max_concurrency = args.concurrency

    config = AppConfig(chunk=chunk_cfg, translator=translator_cfg)

    asyncio.run(translate_file(str(input_path), output_path, config))


if __name__ == "__main__":
    main()
