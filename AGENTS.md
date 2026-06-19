# Core Workflow Rules (Always Follow)

- When implementing a **new feature**, **major change**, **refactor**, or **architectural update**:
  1. First understand the current project structure, tech stack, and coding standards from this AGENTS.md and relevant files.
  2. After completing the changes:
     - Review what was added/modified.
     - **Update this AGENTS.md file** to reflect the new feature, changed architecture, updated standards, new commands, or important lessons learned.
     - Keep updates concise, clear, and actionable. Do not make the file excessively long.
  3. Suggest a brief summary of the update for the user to review.

- Treat AGENTS.md as a **living project memory**. It must remain accurate and up-to-date so future sessions start with the latest context.
# Behavioral guidelines
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Project Snapshot

- Stack: Python 3.11+, async CLI app
- Entry point: `main.py`
- Core package: `ai_subtitle_translator/`
- External deps: `openai`, `python-dotenv`
- Providers: two OpenAI-compatible backends selected via `PROVIDER` / `--provider` (`copilot` | `codex`, default `copilot`). `copilot` is the original engine — API surface is settings-driven via `OPENAI_API_MODE` / `--api-mode` (`auto` | `chat` | `responses`); `auto` tries chat.completions and falls back to the non-streaming Responses API. `codex` (`_CodexProvider`) speaks the streaming-only Responses API via raw httpx SSE (`stream:true`, list-shaped `input`), tolerates non-standard events like `codex.rate_limits`, and ignores `OPENAI_API_MODE`. Both share the `OPENAI_*` connection settings and `OPENAI_SEND_TEMPERATURE` / `--no-temperature`
- Supported subtitle formats: `.srt`, `.ass`

## Current Architecture

- `main.py` orchestrates parse → chunk → context → translate → merge → write
- `config.py` owns env-backed dataclasses: `ChunkConfig`, `TranslatorConfig`, `AppConfig`
- `parser.py` parses SRT and ASS into `SubtitleDocument`; ASS parsing preserves original file lines and dialogue metadata for reconstruction
- `chunker.py` groups subtitles by time gap, then adaptive line/char limits
- `translator.py` owns prompt construction, the provider backends (`_OpenAIProvider` for copilot = chat.completions with Responses fallback; `_CodexProvider` for codex = streaming Responses via httpx SSE), `_build_provider` selection, retries, correction prompts, optional refinement, cache integration, multiline restoration, and metadata preservation across translations
- `postprocess.py` applies Persian-specific cleanup after translation
- `cache.py` persists source-text → translated-text mappings
- `merger.py` flattens translated chunks and writes the final SRT or ASS file
- `resume.py` owns resume / retry-failed-chunks: source/chunk SHA-256 hashing, the `TranslationStatus`/`ChunkRecord` model, status-file path/load/atomic-save/delete, and `build_final_subtitle_from_status`. `translator.py` exposes `translate_chunks_detailed(targets, progress_callback, retry_mode)` returning `ChunkOutcome` per chunk (with validation via `_validate_translation`); `translate_chunks` stays as a backward-compatible wrapper
- `providers.py` owns multi-provider routing. `RoutingProvider` implements the same `_ChatProvider.chat(...)` interface as the single backends, so it's a drop-in: `_build_provider` returns it when `TranslatorConfig.providers_path` is set, else the single provider unchanged. It loads/validates a JSON providers file, keeps per-provider in-memory limit state (RPM window, daily count, cooldown, per-provider semaphore), classifies rate-limit/quota errors across both backend shapes (`_is_rate_limit_error`), and routes `failover` or `round_robin`. It ignores the passed `model` (uses each backend's own `spec.model`) and forwards `temperature` (each backend's `send_temperature` gates the wire)

## Change Guide

- Prompt/provider behavior changes usually belong in `ai_subtitle_translator/translator.py`
- Resume / status-file behavior (hashing, status schema, path naming, rebuild) belongs in `ai_subtitle_translator/resume.py`; the orchestration (find/resume/save-per-chunk/delete/report) lives in `main.translate_file`
- Multi-provider routing (providers-file schema, selection strategy, rate-limit detection, cooldown/cap state) belongs in `ai_subtitle_translator/providers.py`; new backend *types* still go in `translator.py` and must be wired into `providers._build_backend`
- New CLI/env options usually require matching edits in both `main.py` and `ai_subtitle_translator/config.py`
- Chunking changes belong in `ai_subtitle_translator/chunker.py`
- Persian wording or punctuation cleanup changes belong in `ai_subtitle_translator/postprocess.py`
- Output formatting changes belong in `ai_subtitle_translator/merger.py`
- Input format detection and ASS dialogue extraction belong in `ai_subtitle_translator/parser.py`

## Important Constraints

- Keep subtitle count and IDs aligned with source subtitles
- Multi-line subtitles are translated as single-line text, then split back heuristically
- Cache keys are raw source subtitle text
- If a chunk translation fails after retries, the original text is preserved for that chunk. The failure is also recorded in a `.translation-status/<hash>.<lang>.<fmt>.json` file (matched by normalized source-content hash + language + format + chunking version) so a later run resumes only the failed/pending chunks; the status file is written atomically after each chunk and deleted once every chunk is translated. `chunking_version` (`resume.CHUNKING_VERSION`) must be bumped when chunk boundaries change
- Avoid speculative abstractions; this codebase is intentionally small and linear
- For ASS files, only `Dialogue:` text is translated; styles, timing, and non-dialogue sections must remain untouched
- ASS line breaks are normalized to real newlines during translation and restored to `\N`/`\n` when writing
- The system prompt instructs the model to keep markup (`<i>`, ASS `{\...}` tags, `♪`) verbatim and to preserve one-to-one id mapping; a Persian-specific style block (spoken register, تو/شما, ZWNJ, Persian punctuation) is injected only when the target is Persian/Farsi (`_is_persian`). The refinement prompt is parameterized by target language
