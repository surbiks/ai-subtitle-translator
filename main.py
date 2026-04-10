#!/usr/bin/env python3
"""
AI Subtitle Translator — translates SRT subtitles using OpenAI.

Usage:
    python main.py input.srt output.srt
    python main.py input.srt                          # writes to input.fa.srt
    python main.py input.srt -m gpt-4o                # use a specific model
    python main.py input.srt --glossary glossary.json  # use a glossary
    python main.py input.srt --refine                  # enable refinement pass
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from ai_subtitle_translator.cache import TranslationCache
from ai_subtitle_translator.chunker import build_context_window, chunk_subtitles
from ai_subtitle_translator.config import AppConfig, ChunkConfig, TranslatorConfig
from ai_subtitle_translator.glossary import Glossary
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
    """Full pipeline: parse → chunk → context → translate → merge → write."""
    t0 = time.perf_counter()

    # 1. Parse
    logger.info("Parsing %s", input_path)
    subtitles = parse_srt(input_path)
    logger.info("Parsed %d subtitles", len(subtitles))

    if not subtitles:
        logger.warning("No subtitles found — nothing to translate")
        return

    # 2. Chunk (adaptive)
    chunks = chunk_subtitles(subtitles, config.chunk)
    total_chars = sum(len(s.text) for s in subtitles)
    logger.info(
        "Created %d chunks (total %d chars across %d subtitles)",
        len(chunks), total_chars, len(subtitles),
    )

    # 3. Build context windows (previous chunk tail for each chunk)
    contexts = build_context_window(chunks, config.chunk.context_lines)

    # 4. Load glossary
    glossary = None
    if config.translator.glossary_path:
        glossary = Glossary.from_file(config.translator.glossary_path)

    # 5. Load / create cache
    cache = TranslationCache()
    if config.translator.cache_path:
        cache.load_from_file(config.translator.cache_path)

    # 6. Translate
    translator = Translator(config.translator, glossary=glossary, cache=cache)
    translated_chunks = await translator.translate_chunks(chunks, contexts)

    # 7. Merge & write
    merged = merge_chunks(translated_chunks)
    write_srt(merged, output_path)

    # 8. Save cache
    if config.translator.cache_path:
        translator.cache.save_to_file(config.translator.cache_path)

    elapsed = time.perf_counter() - t0
    cache_stats = translator.cache.stats
    logger.info(
        "Done — %d subtitles translated in %.1fs "
        "(cache: %d hits, %d misses, %d stored) → %s",
        len(merged), elapsed,
        cache_stats["hits"], cache_stats["misses"], cache_stats["size"],
        output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Translate SRT subtitles using OpenAI",
    )
    p.add_argument("input", help="Path to the input SRT file")
    p.add_argument(
        "output", nargs="?", default=None,
        help="Path for the translated SRT file (default: <input>.fa.srt)",
    )

    # Translation
    p.add_argument(
        "-l", "--language", default=None,
        help="Target language (default: from .env or 'Persian (Farsi)')",
    )
    p.add_argument(
        "-m", "--model", default=None,
        help="OpenAI model (default: from .env or gpt-4o-mini)",
    )
    p.add_argument(
        "--api-key", default=None,
        help="OpenAI API key (default: from .env)",
    )
    p.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible API base URL (default: from .env)",
    )

    # Quality
    p.add_argument(
        "--glossary", default=None,
        help="Path to glossary JSON file",
    )
    p.add_argument(
        "--refine", action="store_true",
        help="Enable refinement pass for higher quality",
    )
    p.add_argument(
        "--no-postprocess", action="store_true",
        help="Disable Persian post-processing",
    )

    # Performance
    p.add_argument(
        "--concurrency", type=int, default=None,
        help="Max concurrent API calls (default: from .env or 5)",
    )
    p.add_argument(
        "--cache", default=None,
        help="Path to cache JSON file for persistent caching",
    )

    # Chunking
    p.add_argument(
        "--max-lines", type=int, default=None,
        help="Max subtitle lines per chunk (default: from .env or 18)",
    )
    p.add_argument(
        "--max-chars", type=int, default=None,
        help="Max characters per chunk (default: from .env or 1500)",
    )
    p.add_argument(
        "--context-lines", type=int, default=None,
        help="Context lines from previous chunk (default: from .env or 3)",
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

    # Start from .env defaults, then override with any explicit CLI args
    chunk_cfg = ChunkConfig()
    if args.max_lines is not None:
        chunk_cfg.max_lines = args.max_lines
    if args.max_chars is not None:
        chunk_cfg.max_chars = args.max_chars
    if args.context_lines is not None:
        chunk_cfg.context_lines = args.context_lines

    translator_cfg = TranslatorConfig()
    if args.language is not None:
        translator_cfg.target_language = args.language
    if args.model is not None:
        translator_cfg.model = args.model
    if args.api_key is not None:
        translator_cfg.api_key = args.api_key
    if args.base_url is not None:
        translator_cfg.base_url = args.base_url
    if args.concurrency is not None:
        translator_cfg.max_concurrency = args.concurrency
    if args.glossary is not None:
        translator_cfg.glossary_path = args.glossary
    if args.cache is not None:
        translator_cfg.cache_path = args.cache
    if args.refine:
        translator_cfg.enable_refinement = True
    if args.no_postprocess:
        translator_cfg.enable_postprocess = False

    config = AppConfig(chunk=chunk_cfg, translator=translator_cfg)

    asyncio.run(translate_file(str(input_path), output_path, config))


if __name__ == "__main__":
    main()
