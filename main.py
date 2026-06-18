#!/usr/bin/env python3
"""
AI Subtitle Translator — translates SRT or ASS subtitles using OpenAI.

Usage:
    python main.py input.srt output.srt
    python main.py input.srt                          # writes to input.fa.srt
    python main.py input.ass                          # writes to input.fa.ass
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
from ai_subtitle_translator.merger import merge_chunks, write_subtitle_file
from ai_subtitle_translator.parser import parse_subtitle_file
from ai_subtitle_translator.resume import (
    STATUS_FAILED,
    STATUS_TRANSLATED,
    TranslationStatus,
    build_final_subtitle_from_status,
    calculate_source_file_hash,
    create_initial_translation_status,
    delete_translation_status,
    find_existing_translation_status,
    get_status_file_path,
    is_translation_complete,
    now_iso,
    save_translation_status,
    subtitles_to_items,
)
from ai_subtitle_translator.translator import ChunkOutcome, Translator

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
    """Full pipeline with resume support.

    parse → chunk → (find resumable status) → translate only the unfinished
    chunks → rebuild output → write. A per-file status file records failed
    chunks so a later run retries only those; it is deleted automatically once
    every chunk is translated.
    """
    t0 = time.perf_counter()
    tcfg = config.translator
    mode = tcfg.resume_mode

    # 1. Parse
    logger.info("Parsing %s", input_path)
    document = parse_subtitle_file(input_path)
    subtitles = document.subtitles
    logger.info("Parsed %d subtitles from %s", len(subtitles), document.format.upper())

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

    # 4. Resume: locate a status file matching this exact job (unless fresh).
    source_hash = calculate_source_file_hash(input_path)
    status_path = get_status_file_path(
        input_path, source_hash, tcfg.target_language, document.format, tcfg.status_dir,
    )

    existing: TranslationStatus | None = None
    if mode != "fresh":
        found = find_existing_translation_status(
            input_path, source_hash, tcfg.target_language,
            document.format, chunks, tcfg.status_dir,
        )
        if found is not None:
            existing, status_path = found

    if mode == "retry_failed" and existing is None:
        logger.error(
            "retry_failed mode requires an existing translation status file, but "
            "none was found for %s — nothing to do",
            input_path,
        )
        return

    if mode == "fresh":
        delete_translation_status(status_path)  # clean restart

    if existing is not None:
        status = existing
        logger.info(
            "Resuming translation: %d/%d chunks already done, retrying %d",
            status.translated_count(), status.total_chunks,
            len(status.pending_chunk_ids()),
        )
    else:
        status = create_initial_translation_status(
            input_path, source_hash, tcfg.target_language,
            document.format, tcfg.model, chunks,
        )

    is_resume = existing is not None
    previously_translated = status.translated_count()
    targets = status.pending_chunk_ids()

    # 5. Load glossary & cache
    glossary = None
    if tcfg.glossary_path:
        glossary = Glossary.from_file(tcfg.glossary_path)
    cache = TranslationCache()
    if tcfg.cache_path:
        cache.load_from_file(tcfg.cache_path)

    # 6. Translate only the unfinished chunks, persisting after each one so
    #    progress survives a crash mid-run.
    record_by_id = {c.chunk_id: c for c in status.chunks}

    def on_chunk_done(outcome: ChunkOutcome) -> None:
        record = record_by_id.get(outcome.index)
        if record is None:
            return
        record.attempts += 1
        record.last_attempt_at = now_iso()
        if outcome.ok:
            record.status = STATUS_TRANSLATED
            record.translated_items = subtitles_to_items(outcome.subtitles)
            record.translated_text = "\n".join(s.text for s in outcome.subtitles)
            record.last_error = None
        else:
            record.status = STATUS_FAILED
            record.translated_items = None
            record.translated_text = None
            record.last_error = outcome.error
        save_translation_status(status, status_path)

    newly_translated = 0
    cache_stats = cache.stats
    if targets:
        translator = Translator(tcfg, glossary=glossary, cache=cache)
        outcomes = await translator.translate_chunks_detailed(
            chunks, contexts,
            targets=targets,
            progress_callback=on_chunk_done,
            retry_mode=is_resume,
        )
        newly_translated = sum(1 for o in outcomes if o.ok)
        cache_stats = translator.cache.stats
        if tcfg.cache_path:
            translator.cache.save_to_file(tcfg.cache_path)
    else:
        logger.info("All chunks already translated — nothing to send to the API")

    # 7. Rebuild final subtitle: translated text where available, else source.
    final_chunks = build_final_subtitle_from_status(chunks, status)
    merged = merge_chunks(final_chunks)
    write_subtitle_file(document, merged, output_path)

    # 8. Delete the status file once complete; keep it for resume otherwise.
    complete = is_translation_complete(status)
    if complete:
        delete_translation_status(status_path)
    else:
        save_translation_status(status, status_path)

    elapsed = time.perf_counter() - t0
    _report_summary(
        status=status,
        is_resume=is_resume,
        previously_translated=previously_translated,
        retried=len(targets),
        newly_translated=newly_translated,
        complete=complete,
        status_path=status_path,
        elapsed=elapsed,
        output_path=output_path,
        cache_stats=cache_stats,
    )


def _report_summary(
    *,
    status: TranslationStatus,
    is_resume: bool,
    previously_translated: int,
    retried: int,
    newly_translated: int,
    complete: bool,
    status_path,
    elapsed: float,
    output_path: str,
    cache_stats: dict,
) -> None:
    """Print a clear, user-facing run summary and log a one-line result."""
    total = status.total_chunks
    translated = status.translated_count()
    failed = total - translated

    lines = ["", "=" * 52, "Translation summary"]
    if complete:
        lines += [
            f"  Total chunks:      {total}",
            f"  Translated chunks: {translated}",
            "  Failed chunks:     0",
            "  Status:            complete",
            "  Status file deleted: yes",
        ]
    elif is_resume:
        lines += [
            f"  Previous translated chunks: {previously_translated}",
            f"  Retried chunks:             {retried}",
            f"  Newly translated chunks:    {newly_translated}",
            f"  Still failed chunks:        {failed}",
            f"  Total translated chunks:    {translated}",
            "  Status:                     partial",
            "  Resume available:           yes",
            f"  Status file:                {status_path}",
        ]
    else:
        lines += [
            f"  Total chunks:      {total}",
            f"  Translated chunks: {translated}",
            f"  Failed chunks:     {failed}",
            "  Status:            partial",
            "  Resume available:  yes",
            f"  Status file:       {status_path}",
        ]
    lines.append("=" * 52)
    print("\n".join(lines))

    if complete:
        logger.info(
            "Translation complete — all %d chunks translated in %.1fs "
            "(cache: %d hits, %d misses) → %s",
            total, elapsed, cache_stats["hits"], cache_stats["misses"], output_path,
        )
    else:
        logger.info(
            "Translation partial — %d/%d chunks translated in %.1fs → %s "
            "(status saved to %s)",
            translated, total, elapsed, output_path, status_path,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Translate SRT or ASS subtitles using OpenAI",
    )
    p.add_argument("input", help="Path to the input subtitle file (.srt or .ass)")
    p.add_argument(
        "output", nargs="?", default=None,
        help="Path for the translated subtitle file (default: <input>.fa.<ext>)",
    )

    # Translation
    p.add_argument(
        "--provider", default=None, choices=["copilot", "codex"],
        help="Provider backend: 'copilot' (chat + non-stream Responses) or "
             "'codex' (streaming-only Responses API) (default: from .env or 'copilot')",
    )
    p.add_argument(
        "-l", "--language", default=None,
        help="Target language (default: from .env or 'Persian (Farsi)')",
    )
    p.add_argument(
        "-m", "--model", default=None,
        help="Model name (default: from .env or gpt-4o-mini)",
    )
    p.add_argument(
        "--api-key", default=None,
        help="OpenAI API key (default: from .env)",
    )
    p.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible API base URL (default: from .env)",
    )
    p.add_argument(
        "--api-mode", default=None, choices=["auto", "chat", "responses"],
        help="API surface: 'auto' (try chat.completions, fall back to responses), "
             "'chat' (force chat.completions), or 'responses' (force Responses API) "
             "(default: from .env or 'auto')",
    )
    p.add_argument(
        "--no-temperature", action="store_true",
        help="Don't send the temperature parameter (for models that reject it)",
    )

    # Quality
    p.add_argument(
        "--glossary", default=None,
        help="Path to glossary JSON file",
    )
    p.add_argument(
        "--auto-glossary", action="store_true",
        help="Auto-discover proper nouns and recurring terms, then ask the "
             "provider for translations (merged with --glossary; user wins)",
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

    # Resume / retry failed chunks
    p.add_argument(
        "--mode", default=None, choices=["auto", "fresh", "retry_failed"],
        help="Resume mode: 'auto' (resume a matching status file if present, "
             "else fresh), 'fresh' (ignore any status file), or 'retry_failed' "
             "(require a status file and retry only failed chunks) "
             "(default: from .env or 'auto')",
    )
    p.add_argument(
        "--status-dir", default=None,
        help="Directory for translation status files "
             "(default: a .translation-status folder next to the source file)",
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
        output_path = str(input_path.with_name(f"{input_path.stem}.fa{input_path.suffix}"))

    # Start from .env defaults, then override with any explicit CLI args
    chunk_cfg = ChunkConfig()
    if args.max_lines is not None:
        chunk_cfg.max_lines = args.max_lines
    if args.max_chars is not None:
        chunk_cfg.max_chars = args.max_chars
    if args.context_lines is not None:
        chunk_cfg.context_lines = args.context_lines

    translator_cfg = TranslatorConfig()
    if args.provider is not None:
        translator_cfg.provider = args.provider
    if args.language is not None:
        translator_cfg.target_language = args.language
    if args.model is not None:
        translator_cfg.model = args.model
    if args.api_key is not None:
        translator_cfg.api_key = args.api_key
    if args.base_url is not None:
        translator_cfg.base_url = args.base_url
    if args.api_mode is not None:
        translator_cfg.api_mode = args.api_mode
    if args.no_temperature:
        translator_cfg.send_temperature = False
    if args.concurrency is not None:
        translator_cfg.max_concurrency = args.concurrency
    if args.glossary is not None:
        translator_cfg.glossary_path = args.glossary
    if args.auto_glossary:
        translator_cfg.auto_glossary = True
    if args.cache is not None:
        translator_cfg.cache_path = args.cache
    if args.mode is not None:
        translator_cfg.resume_mode = args.mode
    if args.status_dir is not None:
        translator_cfg.status_dir = args.status_dir
    if args.refine:
        translator_cfg.enable_refinement = True
    if args.no_postprocess:
        translator_cfg.enable_postprocess = False

    config = AppConfig(chunk=chunk_cfg, translator=translator_cfg)

    asyncio.run(translate_file(str(input_path), output_path, config))


if __name__ == "__main__":
    main()
