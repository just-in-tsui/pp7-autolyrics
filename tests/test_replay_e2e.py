"""WAV-replay end-to-end harness.

Replays a singing WAV through the full process_audio_chunk pipeline against a
fixture slide_cache and asserts the algorithm triggers at the expected sample
index. This is the YouTube+PP7 manual-workflow replacement.

Skips gracefully when no `*.expected.json` fixtures exist in
tests/fixtures/audio/. See tests/README.md for how to add them.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from audio_pipeline import AudioLoopState, process_audio_chunk
from pp7_poller import PP7SmartPoller

CHUNK_SIZE = 2048              # matches PyAudio frames_per_buffer in main()
SAMPLE_RATE = 16000


def _discover_fixtures():
    audio_dir = Path(__file__).resolve().parent / "fixtures" / "audio"
    return sorted(audio_dir.glob("*.expected.json"))


@pytest.mark.parametrize(
    "expected_json_path",
    _discover_fixtures() or [None],
    ids=lambda p: p.stem.replace(".expected", "") if p else "no-fixtures-present",
)
def test_replay_triggers_at_expected_sample(
    expected_json_path, load_wav, sv_en_engine, sv_yue_engine
):
    if expected_json_path is None:
        pytest.skip(
            "No singing fixtures in tests/fixtures/audio/. "
            "See tests/README.md to add some."
        )

    spec = json.loads(expected_json_path.read_text())
    wav_path = expected_json_path.parent / (
        expected_json_path.name.replace(".expected.json", ".wav")
    )
    if not wav_path.exists():
        pytest.fail(f"Fixture {expected_json_path.name} has no matching {wav_path.name}")

    # Accept either seconds (preferred — read straight off any audio player) or
    # raw samples. Seconds win if both are present.
    def _spec_samples(sec_key, sample_key, required=True):
        if sec_key in spec:
            return int(round(spec[sec_key] * SAMPLE_RATE))
        if sample_key in spec:
            return spec[sample_key]
        if required:
            pytest.fail(
                f"{expected_json_path.name} must define '{sec_key}' (seconds) "
                f"or '{sample_key}' (samples)."
            )
        return None

    samples = load_wav(wav_path)
    assert samples.dtype == np.float32

    # Drive the poller directly: skip HTTP, just put it in the state the live
    # script would be in after PP7 sent the slide_cache.
    poller = PP7SmartPoller()
    poller.slide_cache = spec["slide_cache"]
    poller.current_index = spec["starting_slide_index"]
    poller.current_full_text = poller.slide_cache[poller.current_index]["text"]
    target, idx = poller.get_target()
    is_chinese = poller.is_chinese_slide

    state = AudioLoopState()
    state.current_slide = idx

    triggers = []  # list of (sample_index_where_trigger_decision_was_made, pre_trigger_delay)
    for offset in range(0, len(samples), CHUNK_SIZE):
        chunk = samples[offset : offset + CHUNK_SIZE]
        if len(chunk) == 0:
            break
        result = process_audio_chunk(
            chunk, state, target, is_chinese,
            sv_en_engine, sv_yue_engine,
        )
        if result.should_trigger:
            decision_sample = offset + len(chunk)
            effective_sample = decision_sample + int(result.pre_trigger_delay * SAMPLE_RATE)
            triggers.append((decision_sample, result.pre_trigger_delay, effective_sample))
            break  # live script would advance to next slide and reset the buffer

    assert triggers, (
        f"Algorithm never triggered for {expected_json_path.stem}. "
        f"Target was {target!r}; consider lowering threshold or extending the WAV."
    )

    decision_sample, delay, effective_sample = triggers[0]

    fp_limit = _spec_samples(
        "no_false_positive_before_sec", "no_false_positive_before_sample", required=False
    )
    if fp_limit is not None:
        assert decision_sample >= fp_limit, (
            f"False-positive trigger at sample {decision_sample} "
            f"({decision_sample / SAMPLE_RATE:.2f}s), before the no-false-positive "
            f"limit of {fp_limit} ({fp_limit / SAMPLE_RATE:.2f}s). Target was {target!r}."
        )

    expected_sample = _spec_samples("expected_trigger_sec", "expected_trigger_sample")
    tolerance = _spec_samples("tolerance_sec", "tolerance_samples")
    drift = effective_sample - expected_sample
    assert abs(drift) <= tolerance, (
        f"Triggered at {effective_sample / SAMPLE_RATE:.2f}s "
        f"(decision {decision_sample / SAMPLE_RATE:.2f}s + {delay:.2f}s delay), "
        f"expected {expected_sample / SAMPLE_RATE:.2f}s "
        f"± {tolerance / SAMPLE_RATE:.2f}s. "
        f"Drift: {drift / SAMPLE_RATE:+.2f}s. Target was {target!r}."
    )
