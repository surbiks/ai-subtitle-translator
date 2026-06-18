"""Tests for resume / retry-failed-chunks support (no network required)."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_subtitle_translator import translator as translator_mod
from ai_subtitle_translator.config import AppConfig, ChunkConfig, TranslatorConfig
from ai_subtitle_translator.parser import Subtitle
from ai_subtitle_translator.translator import _validate_translation
from ai_subtitle_translator import resume
from ai_subtitle_translator.resume import (
    ChunkRecord,
    TranslationStatus,
    build_final_subtitle_from_status,
    calculate_chunk_hash,
    calculate_source_file_hash,
    create_initial_translation_status,
    delete_translation_status,
    find_existing_translation_status,
    get_status_file_path,
    is_translation_complete,
    load_translation_status,
    normalize_source_content,
    save_translation_status,
)

import main as main_mod


def _sub(i, text, start="00:00:01,000", end="00:00:02,000"):
    return Subtitle(id=i, start=start, end=end, text=text)


# Three time-separated dialogue groups → three chunks of two subtitles each.
SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:02,000
Hello there.

2
00:00:02,500 --> 00:00:03,500
General Kenobi.

3
00:00:10,000 --> 00:00:11,000
You are a bold one.

4
00:00:11,500 --> 00:00:12,500
Indeed I am.

5
00:00:20,000 --> 00:00:21,000
Back away.

6
00:00:21,500 --> 00:00:22,500
Never.
"""


class HashingTest(unittest.TestCase):
    def test_normalize_strips_bom_and_crlf(self):
        self.assertEqual(normalize_source_content("﻿a\r\nb\rc"), "a\nb\nc")

    def test_source_hash_stable_across_line_endings_and_bom(self):
        with tempfile.TemporaryDirectory() as d:
            unix = Path(d) / "a.srt"
            unix.write_text("1\nhi\n", encoding="utf-8")
            win = Path(d) / "b.srt"
            win.write_bytes("﻿1\r\nhi\r\n".encode("utf-8"))
            self.assertEqual(
                calculate_source_file_hash(unix), calculate_source_file_hash(win)
            )

    def test_chunk_hash_changes_with_text(self):
        a = [_sub(1, "hello")]
        b = [_sub(1, "world")]
        self.assertNotEqual(calculate_chunk_hash(a), calculate_chunk_hash(b))
        self.assertEqual(calculate_chunk_hash(a), calculate_chunk_hash([_sub(1, "hello")]))


class StatusModelTest(unittest.TestCase):
    def _status(self):
        chunks = [[_sub(1, "a")], [_sub(2, "b")]]
        return create_initial_translation_status(
            "movie.srt", "deadbeef", "Persian (Farsi)", "srt", "gpt-4o-mini", chunks
        )

    def test_initial_status_all_pending(self):
        status = self._status()
        self.assertEqual(status.total_chunks, 2)
        self.assertEqual(status.pending_chunk_ids(), [0, 1])
        self.assertFalse(is_translation_complete(status))

    def test_roundtrip_serialization(self):
        status = self._status()
        status.chunks[0].status = resume.STATUS_TRANSLATED
        status.chunks[0].translated_items = [{"id": 1, "text": "آ"}]
        restored = TranslationStatus.from_dict(json.loads(json.dumps(status.to_dict())))
        self.assertEqual(restored.chunks[0].translated_items, [{"id": 1, "text": "آ"}])
        self.assertEqual(restored.pending_chunk_ids(), [1])

    def test_complete_when_all_translated(self):
        status = self._status()
        for rec in status.chunks:
            rec.status = resume.STATUS_TRANSLATED
            rec.translated_items = [{"id": rec.chunk_id, "text": "x"}]
        self.assertTrue(is_translation_complete(status))
        self.assertEqual(status.pending_chunk_ids(), [])

    def test_translated_status_without_items_is_not_complete(self):
        # A "translated" marker with no actual text must not count as done.
        rec = ChunkRecord(chunk_id=0, source_hash="h", source_text="a",
                          status=resume.STATUS_TRANSLATED, translated_items=None)
        self.assertFalse(rec.is_translated)


class PersistenceTest(unittest.TestCase):
    def test_save_load_delete_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            chunks = [[_sub(1, "a")]]
            status = create_initial_translation_status(
                "m.srt", "hash123", "Spanish", "srt", "m", chunks
            )
            path = Path(d) / ".translation-status" / "x.json"
            save_translation_status(status, path)
            self.assertTrue(path.exists())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

            loaded = load_translation_status(path)
            self.assertEqual(loaded.source_file_hash, "hash123")

            delete_translation_status(path)
            self.assertFalse(path.exists())
            # The .translation-status dir is removed when it becomes empty.
            self.assertFalse(path.parent.exists())

    def test_load_missing_returns_none(self):
        self.assertIsNone(load_translation_status("/nonexistent/x.json"))


class FindExistingTest(unittest.TestCase):
    def _setup(self, d):
        src = Path(d) / "movie.srt"
        src.write_text(SAMPLE_SRT, encoding="utf-8")
        chunks = [[_sub(1, "a"), _sub(2, "b")], [_sub(3, "c")]]
        h = calculate_source_file_hash(src)
        status = create_initial_translation_status(
            src, h, "Spanish", "srt", "m", chunks
        )
        path = get_status_file_path(src, h, "Spanish", "srt")
        save_translation_status(status, path)
        return src, chunks, h

    def test_finds_matching_status(self):
        with tempfile.TemporaryDirectory() as d:
            src, chunks, h = self._setup(d)
            found = find_existing_translation_status(src, h, "Spanish", "srt", chunks)
            self.assertIsNotNone(found)

    def test_rejects_on_language_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            src, chunks, h = self._setup(d)
            # Different language → different path → nothing found.
            self.assertIsNone(
                find_existing_translation_status(src, h, "French", "srt", chunks)
            )

    def test_rejects_on_chunk_count_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            src, chunks, h = self._setup(d)
            wrong = chunks + [[_sub(4, "d")]]
            self.assertIsNone(
                find_existing_translation_status(src, h, "Spanish", "srt", wrong)
            )

    def test_rejects_on_chunk_content_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            src, chunks, h = self._setup(d)
            changed = [[_sub(1, "a"), _sub(2, "CHANGED")], [_sub(3, "c")]]
            self.assertIsNone(
                find_existing_translation_status(src, h, "Spanish", "srt", changed)
            )


class BuildFinalTest(unittest.TestCase):
    def test_uses_translation_where_available_else_source(self):
        chunks = [[_sub(1, "hello"), _sub(2, "world")], [_sub(3, "bye")]]
        status = create_initial_translation_status(
            "m.srt", "h", "Spanish", "srt", "m", chunks
        )
        status.chunks[0].status = resume.STATUS_TRANSLATED
        status.chunks[0].translated_items = [
            {"id": 1, "text": "hola"}, {"id": 2, "text": "mundo"}
        ]
        # chunk 1 left failed/pending → original text preserved
        rebuilt = build_final_subtitle_from_status(chunks, status)
        self.assertEqual([s.text for s in rebuilt[0]], ["hola", "mundo"])
        self.assertEqual([s.text for s in rebuilt[1]], ["bye"])
        # timing is preserved from the source chunk
        self.assertEqual(rebuilt[0][0].start, chunks[0][0].start)


class ValidateTranslationTest(unittest.TestCase):
    def test_ok(self):
        orig = [_sub(1, "hello")]
        trans = [_sub(1, "hola")]
        self.assertIsNone(_validate_translation(orig, trans, lambda k: True))

    def test_empty(self):
        self.assertIsNotNone(_validate_translation([_sub(1, "x")], [], lambda k: True))

    def test_count_change(self):
        orig = [_sub(1, "a"), _sub(2, "b")]
        trans = [_sub(1, "a")]
        self.assertIsNotNone(_validate_translation(orig, trans, lambda k: True))

    def test_identical_prose_fails(self):
        orig = [_sub(1, "Hello there."), _sub(2, "General Kenobi.")]
        trans = [_sub(1, "Hello there."), _sub(2, "General Kenobi.")]
        self.assertIsNotNone(_validate_translation(orig, trans, lambda k: True))

    def test_name_only_chunk_passes(self):
        # A chunk of just a proper noun stays identical and must NOT be flagged.
        orig = [_sub(482, "Carla!"), _sub(483, "Carla!")]
        trans = [_sub(482, "Carla!"), _sub(483, "Carla!")]
        self.assertIsNone(_validate_translation(orig, trans, lambda k: True))

    def test_symbol_only_chunk_passes(self):
        orig = [_sub(1, "♪"), _sub(2, "123")]
        trans = [_sub(1, "♪"), _sub(2, "123")]
        self.assertIsNone(_validate_translation(orig, trans, lambda k: True))

    def test_partial_translation_passes(self):
        # Name kept verbatim but the prose line was translated → not a failure.
        orig = [_sub(1, "Carla!"), _sub(2, "I love you.")]
        trans = [_sub(1, "Carla!"), _sub(2, "دوستت دارم")]
        self.assertIsNone(_validate_translation(orig, trans, lambda k: True))


class FakeProvider:
    """Translates by prefixing 'ES:'; raises for chunks touching fail_ids."""

    def __init__(self, fail_ids=None):
        self.fail_ids = set(fail_ids or [])
        self.calls = 0

    async def chat(self, system, messages, model, temperature):
        self.calls += 1
        content = messages[-1]["content"]
        payload = json.loads(content.split("Translate the following:\n")[-1])
        ids = [int(item["id"]) for item in payload]
        if any(i in self.fail_ids for i in ids):
            raise RuntimeError("simulated chunk failure")
        return json.dumps(
            [{"id": int(item["id"]), "text": "ES:" + item["text"]} for item in payload],
            ensure_ascii=False,
        )


def _make_config(status_dir):
    chunk = ChunkConfig(max_lines=18, max_chars=1500, time_gap_threshold_ms=2500,
                        context_lines=0)
    tcfg = TranslatorConfig()
    tcfg.target_language = "Spanish"   # avoid Persian post-processing in tests
    tcfg.provider = "copilot"
    tcfg.api_key = "dummy"
    tcfg.max_retries = 1
    tcfg.retry_base_delay = 0.0
    tcfg.max_concurrency = 4
    tcfg.enforce_cps = False
    tcfg.enforce_glossary = False
    tcfg.enable_refinement = False
    tcfg.enable_postprocess = False
    tcfg.auto_probe = False
    tcfg.auto_glossary = False
    tcfg.enable_memory = False
    tcfg.glossary_path = None
    tcfg.cache_path = None
    tcfg.status_dir = status_dir
    return AppConfig(chunk=chunk, translator=tcfg)


class EndToEndResumeTest(unittest.TestCase):
    def _run(self, src, out, config, fail_ids):
        provider = FakeProvider(fail_ids)
        orig_build = translator_mod._build_provider
        translator_mod._build_provider = lambda cfg: provider
        try:
            asyncio.run(main_mod.translate_file(str(src), str(out), config))
        finally:
            translator_mod._build_provider = orig_build
        return provider

    def test_partial_then_resume_completes_and_deletes_status(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "movie.srt"
            src.write_text(SAMPLE_SRT, encoding="utf-8")
            out = Path(d) / "movie.es.srt"
            config = _make_config(str(Path(d) / "status"))

            h = calculate_source_file_hash(src)
            status_path = get_status_file_path(src, h, "Spanish", "srt",
                                               str(Path(d) / "status"))

            # First run: chunk containing ids 3,4 fails.
            self._run(src, out, config, fail_ids={3, 4})

            self.assertTrue(status_path.exists(), "status file should persist failures")
            status = load_translation_status(status_path)
            self.assertEqual(status.translated_count(), 2)
            self.assertEqual(status.pending_chunk_ids(), [1])
            self.assertEqual(status.chunks[1].status, resume.STATUS_FAILED)

            # Failed chunk keeps original text in the output.
            out_text = out.read_text(encoding="utf-8")
            self.assertIn("ES:Hello there.", out_text)
            self.assertIn("You are a bold one.", out_text)        # original, untranslated
            self.assertNotIn("ES:You are a bold one.", out_text)

            # Second run (auto mode): only the failed chunk is retried.
            provider = self._run(src, out, config, fail_ids=set())
            self.assertEqual(provider.calls, 1, "resume must retry only the 1 failed chunk")

            self.assertFalse(status_path.exists(), "status removed once complete")
            out_text = out.read_text(encoding="utf-8")
            self.assertIn("ES:You are a bold one.", out_text)
            self.assertIn("ES:Never.", out_text)

    def test_clean_run_leaves_no_status_file(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "movie.srt"
            src.write_text(SAMPLE_SRT, encoding="utf-8")
            out = Path(d) / "movie.es.srt"
            config = _make_config(str(Path(d) / "status"))
            h = calculate_source_file_hash(src)
            status_path = get_status_file_path(src, h, "Spanish", "srt",
                                               str(Path(d) / "status"))

            self._run(src, out, config, fail_ids=set())
            self.assertFalse(status_path.exists())
            self.assertIn("ES:Never.", out.read_text(encoding="utf-8"))

    def test_fresh_mode_ignores_existing_status(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "movie.srt"
            src.write_text(SAMPLE_SRT, encoding="utf-8")
            out = Path(d) / "movie.es.srt"
            config = _make_config(str(Path(d) / "status"))

            # Seed a partial status (chunk 1 failed).
            self._run(src, out, config, fail_ids={3, 4})

            # Fresh mode retranslates everything regardless of the status file.
            config.translator.resume_mode = "fresh"
            provider = self._run(src, out, config, fail_ids=set())
            self.assertEqual(provider.calls, 3, "fresh mode retranslates all 3 chunks")


if __name__ == "__main__":
    unittest.main()
