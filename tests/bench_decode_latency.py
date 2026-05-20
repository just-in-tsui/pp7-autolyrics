"""SenseVoice decode-latency benchmark across realistic buffer sizes.

The live script's rolling buffer grows from 0.4s (first decode after
PROCESS_INTERVAL) up to MAX_BUFFER_SIZE=8s during continuous singing. Each
decode processes the ENTIRE buffer, not just the latest 0.4s chunk — so the
latency the system actually sees scales with how full the buffer is. The cap
is the worst case: every decode is on 8s of audio until the next trigger
clears the buffer.

This benchmark prints min / median / p95 decode latency for EN and yue
engines at several buffer fill levels, and flags any row whose decode time
exceeds the PROCESS_INTERVAL budget (400ms) — that's when the live script
falls behind real-time.

Run with:
    python tests/bench_decode_latency.py
    # or, after `cd <repo_root>`:
    python -m tests.bench_decode_latency
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
TEST_WAVS_DIR = MODEL_DIR / "test_wavs"

SAMPLE_RATE = 16000
# Buffer fill levels the live script actually sees:
#   0.4s = first decode after PROCESS_INTERVAL
#   8.0s = MAX_BUFFER_SIZE steady state during continuous singing
BUFFER_SECONDS = (0.4, 1.0, 2.0, 4.0, 6.0, 8.0)
N_RUNS_PER_BUFFER = 20
BUDGET_MS = 400  # PROCESS_INTERVAL=0.4s; decode must stay under this to keep up
ENGINES = (
    ("en", "en.wav"),
    ("yue", "yue.wav"),
)


def _load_engine(language: str):
    import sherpa_onnx
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(MODEL_DIR / "model.int8.onnx"),
        tokens=str(MODEL_DIR / "tokens.txt"),
        num_threads=2,
        use_itn=False,
        language=language,
        provider="cpu",
    )


def _load_wav(path: Path) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"{path.name}: sample rate {sr}, expected {SAMPLE_RATE}")
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32)
    return data


def _sliding_windows(samples: np.ndarray, window_len: int, n_windows: int):
    """Yield n_windows fixed-step windows of length window_len.

    If the WAV is too short for n distinct windows, falls back to repeating
    the available windows (latency is the property under test, content
    variability is secondary).
    """
    if len(samples) < window_len:
        # Repeat the WAV until it's long enough
        reps = (window_len // len(samples)) + 2
        samples = np.tile(samples, reps)

    max_start = len(samples) - window_len
    if max_start <= 0:
        for _ in range(n_windows):
            yield samples[:window_len]
        return

    step = max(1, max_start // max(1, n_windows - 1))
    starts = [(i * step) % (max_start + 1) for i in range(n_windows)]
    for s in starts:
        yield samples[s : s + window_len]


def _bench_one(engine, samples: np.ndarray, buffer_sec: float) -> list[float]:
    window_len = int(buffer_sec * SAMPLE_RATE)
    timings_ms: list[float] = []

    # Warm-up: first decode is slow (cold caches, lazy init)
    warm_stream = engine.create_stream()
    warm_stream.accept_waveform(SAMPLE_RATE, samples[:window_len])
    engine.decode_stream(warm_stream)

    for window in _sliding_windows(samples, window_len, N_RUNS_PER_BUFFER):
        stream = engine.create_stream()
        stream.accept_waveform(SAMPLE_RATE, window)
        t0 = time.perf_counter()
        engine.decode_stream(stream)
        timings_ms.append((time.perf_counter() - t0) * 1000)

    return timings_ms


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"ERROR: SenseVoice model not found at {MODEL_DIR}", file=sys.stderr)
        return 1
    if not TEST_WAVS_DIR.exists():
        print(f"ERROR: test_wavs not found at {TEST_WAVS_DIR}", file=sys.stderr)
        return 1

    rows = []
    for lang, wav_name in ENGINES:
        wav_path = TEST_WAVS_DIR / wav_name
        if not wav_path.exists():
            print(f"WARNING: skipping {lang}, missing {wav_path}", file=sys.stderr)
            continue
        print(f"Loading {lang} engine and {wav_name}…", file=sys.stderr)
        engine = _load_engine(lang)
        samples = _load_wav(wav_path)

        for buffer_sec in BUFFER_SECONDS:
            print(f"  benchmarking {lang} @ {buffer_sec}s buffer × {N_RUNS_PER_BUFFER} runs…", file=sys.stderr)
            timings = _bench_one(engine, samples, buffer_sec)
            timings_sorted = sorted(timings)
            p95_idx = int(0.95 * (len(timings_sorted) - 1))
            row = {
                "engine": lang,
                "buffer_s": buffer_sec,
                "min_ms": min(timings_sorted),
                "median_ms": statistics.median(timings_sorted),
                "p95_ms": timings_sorted[p95_idx],
                "n": len(timings),
            }
            row["median_over"] = row["median_ms"] >= BUDGET_MS
            row["p95_over"] = row["p95_ms"] >= BUDGET_MS
            rows.append(row)

    # Markdown table
    print()
    print("# SenseVoice decode-latency benchmark")
    print()
    print(f"Machine: {sys.platform}, CPU provider, num_threads=2")
    print(f"Samples per row: {N_RUNS_PER_BUFFER} sliding windows from the matching test_wav")
    print(f"Budget: {BUDGET_MS}ms — the PROCESS_INTERVAL in the live script. Decodes")
    print("slower than this mean the audio loop falls behind real-time.")
    print()
    print("Buffer length is how much audio is in the rolling window AT decode time.")
    print("0.4s = first decode after PROCESS_INTERVAL fires.")
    print("8.0s = MAX_BUFFER_SIZE; steady-state during continuous singing.")
    print()
    print("| Engine | Buffer | Min     | Median  | P95     | N    | vs budget         |")
    print("|--------|--------|---------|---------|---------|------|-------------------|")
    for r in rows:
        if r["median_over"]:
            flag = "⚠️ median OVER"
        elif r["p95_over"]:
            flag = "⚠️ p95 OVER"
        else:
            flag = "ok"
        print(
            f"| {r['engine']:<6} | {r['buffer_s']:.1f}s   | "
            f"{r['min_ms']:6.1f}ms | {r['median_ms']:6.1f}ms | "
            f"{r['p95_ms']:6.1f}ms | {r['n']:>4} | {flag:<17} |"
        )
    print()
    if any(r["median_over"] for r in rows):
        print(
            "❌ Some median decodes exceed the budget — the live script will fall behind\n"
            "   real-time once the buffer reaches that fill level. Mitigations: reduce\n"
            "   MAX_BUFFER_SIZE, raise PROCESS_INTERVAL, use a smaller model, or move to\n"
            "   a faster ONNX provider."
        )
    elif any(r["p95_over"] for r in rows):
        print(
            "⚠️  Median decodes fit the budget, but p95 occasionally exceeds it. The live\n"
            "   script will hiccup under load but should recover."
        )

    return 1 if any(r["median_over"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
