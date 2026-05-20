# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A single-purpose Python tool that listens to a live singer and auto-advances slides in **ProPresenter 7** by detecting when the singer reaches the end of the current slide's lyrics. All speech recognition runs locally on CPU via `sherpa-onnx` SenseVoice.

## Run / Develop

The project is two standalone scripts (no build, no tests, no package). All deps come from `requirements.txt`.

```bash
# Activate the existing venv (do not create a new one — optVenv is already set up)
source optVenv/bin/activate

# Standard version
python PP7_SenseVoice_Only.py

# Optimized version (adds VAD silence-skip + dynamic short/long buffer)
python PP7_SenseVoice_Optimized.py
```

Both scripts prompt for a microphone index at startup (press Enter for default). They require ProPresenter 7 running with **Network enabled on port 1025, no password** (`Settings > Network`).

On macOS, `pynput` global hotkeys require **Accessibility permission** for the terminal/IDE running the script — silent hotkey failures usually mean this is missing.

Performance logs are written to `performance_<date>.log` / `performance_optimized_<date>.log` in the repo root each run; these are gitignored implicitly via being session artifacts (not in `.gitignore`, but treat as disposable).

## Architecture

`PP7_SenseVoice_Only.py` and `PP7_SenseVoice_Optimized.py` are **near-duplicates** sharing the same core architecture. Treat the optimized version as the iteration target; the "Only" version is the reference baseline. When changing logic, **expect to mirror non-trivial edits across both files** unless the user explicitly says otherwise.

### The three concurrent loops

1. **`PP7SmartPoller.update_loop`** (background thread, ~150ms cadence) — polls `/v1/presentation/slide_index`, detects slide/presentation changes via slide index + presentation UUID, and on a UUID change re-fetches the whole presentation into `slide_cache` (a list of `{text, group}` dicts). The cache is what enables section-jump hotkeys without re-querying PP7.
2. **`pynput` keyboard listener** (background thread) — handles arrow keys (slide/playlist nav), `/` pause toggle, `,` / `.` fast/slow mode toggle, `;` clear cued jump, and the `HOTKEYS` dict for section jumps. Section jumps use a **double-press-within-0.4s** pattern: single press cues a jump for end-of-section, double press jumps immediately.
3. **`main` audio loop** (main thread) — reads 2048-sample chunks from PyAudio into a rolling float32 buffer, every `PROCESS_INTERVAL` (0.4s standard / 0.6s optimized) decodes the buffer through whichever SenseVoice engine matches the current slide's language, fuzzy-matches against the slide's target phrase, and triggers the next slide on a threshold hit.

### Target phrase extraction (`PP7SmartPoller.get_target`)

The "target" is the **tail** of the current slide's text — what the singer says last before the slide should advance. Language detection is character-based: any CJK char in `一-鿿` flips `is_chinese_slide=True` and the target becomes the **last 10 Chinese characters only** (non-Chinese chars stripped). Otherwise the target is the **last 35 characters** of the full text. This is also what determines which of the two preloaded SenseVoice engines (`rec_sv_en` for `"en"`, `rec_sv_cn` for `"yue"` Cantonese) decodes the audio.

### Trigger decision (`fast_smart_score`)

`fuzz.partial_ratio` against the target, then **two penalties** applied to the raw score:

- **Repetition enforcer**: if the target ends in a repeated anchor (last 4 chars for CN, last word for EN) and the heard text hasn't repeated it the same number of times, subtract 40. This prevents premature triggers on choruses like "hallelujah hallelujah hallelujah".
- **Missing-content penalty**: −6 per missing Chinese char, −3 per missing English letter.

Heard Chinese is normalized through `opencc` Simplified→Traditional before comparison (PP7 slide text is typically Traditional).

Thresholds: `EN_THRESHOLD=65`, `CN_THRESHOLD=55` (both scripts; tuned for plugged-in mic). Note that `is_slow_mode` currently does **not** change thresholds — only delay caps — despite the variable name suggesting otherwise.

### Dynamic pre-trigger delay

When score crosses threshold but `len(clean_heard) < len(target)`, the script sleeps before triggering: `missing_chars * 0.3s` (CN) or `* 0.15s` (EN), capped at 2.5s (or 3.5/4.0s in slow mode). Slow mode multipliers are `0.5s`/`0.25s`. After triggering, an unconditional 0.8s sleep + audio buffer wipe prevents the previous line's trailing audio ("ghost audio") from matching the new slide.

### Cued jumps vs. immediate jumps

`handle_lyric_trigger` checks whether the next slide belongs to a different group than the current slide. If yes (end-of-section) AND a `cued_slide_index` is set, it jumps there instead of just calling `/next/trigger`. This is how the single-press section hotkeys work — they queue a destination that fires only when the current section naturally finishes.

### Optimized script differences

Only three meaningful additions over the baseline:

1. **VAD silence skip**: RMS energy of the latest chunk < `SILENCE_THRESHOLD=0.005` → skip decode entirely (tracked in `perf_metrics['vad_skips']`).
2. **Dynamic decode buffer**: `check_target_repetitions` checks if the target ends in a repeated anchor. If yes, decode the full 8s buffer (need history to count repeats); if no, decode only the last 4s (`SHORT_BUFFER_SEC`) for faster inference.
3. **Longer process interval**: 0.6s vs 0.4s, fewer decode cycles.

The `MAX_BUFFER_SIZE` rolling buffer is still maintained at 8s in both — the difference is what slice gets sent to the decoder.

## Conventions specific to this codebase

- **Silent exception swallowing is intentional** in the polling loop, API trigger functions, and main audio loop (`except: pass` / `except Exception: pass`). PP7's network API drops requests when slides are mid-transition, and PyAudio occasionally throws on overflow — the loops must keep running. Do not "fix" these by adding logging or re-raising without understanding the live-performance failure mode.
- **`os._exit(0)` in `force_quit`** is deliberate. The script has multiple background threads holding sockets and audio streams; a clean shutdown would hang. SIGINT prints the performance summary and hard-exits.
- The `MODEL_DIR_SV` folder (`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/`) is **gitignored** (>150MB). A `.tar.bz2` of it is checked in at the repo root; extract before first run.
- `Meeting_Transcriber.py`, `Archive/`, `Qwen_Local/`, and the `optVenv*` dirs are gitignored — treat as out-of-scope unless explicitly asked.
- Console UX relies on `\r` carriage-return updates and ANSI color codes in log messages — don't replace `sys.stdout.write` with `print` in the hot path; it'll spam the terminal.
