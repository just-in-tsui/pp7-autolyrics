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

# Full-auto version (tracks position across the whole song, not just the current slide)
python PP7_SenseVoice_FullAuto.py
python PP7_SenseVoice_FullAuto.py --shadow    # log decisions without driving PP7
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

### Full-auto script (`PP7_SenseVoice_FullAuto.py`)

A **third** standalone script, not a near-duplicate of the other two. It keeps the
tail-match advance path byte-for-byte identical to the baseline (same thresholds,
repetition enforcer, and dynamic pre-trigger delay) and adds one independent
subsystem on top: the **song-wide localizer**. Do not "sync" tuning changes from
the other two scripts into the localizer, and do not let the advance path here
drift from the baseline — a field test comparing the scripts is only meaningful
while that half stays identical.

**What the localizer does.** Every decode cycle it scores the heard transcript
against *every slide in the presentation* using `rapidfuzz.partial_ratio_alignment`
(not `thefuzz`, which cannot return an alignment). The alignment gives both a
similarity score and the character span of the slide the transcript landed on, so
each candidate carries a `progress` (0..1) saying how far through that slide the
singer is. Raw `partial_ratio` is corrected by two penalties before ranking:
`COVERAGE_PENALTY` (the slide explains only part of what we heard — stops a
two-word slide out-ranking the long line that accounts for the whole buffer) and
`RECENCY_PENALTY` (the slide matches the *start* of the buffer, so the singer has
already sung past it).

**Three-legged confidence gate**, all required before anything moves: an absolute
score floor (`LOCATE_THRESHOLD_*`), a `LOCATE_MARGIN` over the best rival, and
`VOTE_REQUIRED` **consecutive** agreeing cycles. The votes are consecutive rather
than a count over the window on purpose — a plain count lets a flapping recogniser
(A, B, A) reach quorum for A without ever settling. There is a test for this.

**Verbatim-repeated slides** (a chorus printed twice) are found at index-build
time and grouped in `dup_groups`. They are acoustically indistinguishable, so they
are excluded from the margin calculation (otherwise a repeated chorus could never
clear the margin at all) and resolved by position instead — nearest copy, forward
preferred.

**Actions.** `delta > 1` → absolute jump. `delta == 1` → routed through
`handle_lyric_trigger(origin="rescue")` rather than calling `/next` directly, so a
rescue still honours a cued section jump; it is gated on `NEXT_MIN_PROGRESS` so it
only fires once the singer is audibly *into* the next slide, leaving normal
end-of-line advances to the tuned tail matcher.

**Operator overrides win.** With hold-at-section-end (`;`) on, full auto degrades
to auto-*cueing* — it sets `cued_slide_index` and waits for `→` instead of jumping.

**Bilingual decks.** Language is per-slide as elsewhere, but the localizer doesn't
know where it is, so `plan_languages` decodes the current slide's language every
cycle and *probes* the other engine every `PROBE_EVERY` cycles (and every cycle
when lost). A single-language deck never pays for a second decode.

**Field-test instrumentation** is the point of the script, not a nicety: it writes
`fullauto_events_<date>.jsonl` (one record per decode cycle and decision) alongside
the usual perf log, and measures **lead time** — how long it had already been sure
of a slide before the operator moved there by hand. `analyze_fullauto_log.py`
turns that into the comparison numbers. See `FULLAUTO_FIELD_TEST.md`.

Heavy imports stay at module level to match the other scripts, but the SenseVoice
engines load inside `main()` rather than at import, so the module can be imported
for tests. `tests/_fullauto_stubs.py` stubs only genuinely-absent hardware modules,
so on a real dev machine the real ones are still used.

## Conventions specific to this codebase

- **Silent exception swallowing is intentional** in the polling loop, API trigger functions, and main audio loop (`except: pass` / `except Exception: pass`). PP7's network API drops requests when slides are mid-transition, and PyAudio occasionally throws on overflow — the loops must keep running. Do not "fix" these by adding logging or re-raising without understanding the live-performance failure mode.
- **`os._exit(0)` in `force_quit`** is deliberate. The script has multiple background threads holding sockets and audio streams; a clean shutdown would hang. SIGINT prints the performance summary and hard-exits.
- The `MODEL_DIR_SV` folder (`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/`) is **gitignored** (>150MB). A `.tar.bz2` of it is checked in at the repo root; extract before first run.
- `Meeting_Transcriber.py`, `Archive/`, `Qwen_Local/`, and the `optVenv*` dirs are gitignored — treat as out-of-scope unless explicitly asked.
- Console UX relies on `\r` carriage-return updates and ANSI color codes in log messages — don't replace `sys.stdout.write` with `print` in the hot path; it'll spam the terminal.
