"""Tests for process_audio_chunk: rolling buffer, decode gating, VAD silence
skip, dynamic decode window, engine routing, and trigger wiring.

Uses a mock SenseVoice engine so they run in milliseconds with no model on disk.

The buffer/decode-logic tests disable VAD and dynamic windowing (via `_run`) so
they exercise one behavior at a time; VAD and the dynamic window have their own
dedicated tests.
"""

from dataclasses import dataclass

import numpy as np

from audio_pipeline import AudioLoopState, process_audio_chunk, target_has_repeat


@dataclass
class _FakeResult:
    text: str


class _FakeStream:
    def __init__(self):
        self.received_samples = None
        self.result = _FakeResult("")

    def accept_waveform(self, sample_rate, samples):
        self.received_samples = np.asarray(samples).copy()


class FakeEngine:
    def __init__(self, fake_text: str = ""):
        self.fake_text = fake_text
        self.decode_calls = 0
        self.last_stream = None

    def create_stream(self):
        return _FakeStream()

    def decode_stream(self, stream):
        self.decode_calls += 1
        self.last_stream = stream
        stream.result = _FakeResult(self.fake_text)


PI = 6400      # process interval (0.4s)
MAXB = 16000   # max buffer (1s)


def _run(samples, state, target, is_chinese, en, cn, **kw):
    """Call process_audio_chunk with VAD and dynamic windowing OFF by default,
    so buffer/decode-logic tests aren't affected by them."""
    kw.setdefault("silence_threshold", 0.0)
    kw.setdefault("short_buffer_samples", 10_000_000)
    return process_audio_chunk(samples, state, target, is_chinese, en, cn, **kw)


# ----- accumulation --------------------------------------------------------

def test_single_chunk_appends_without_decoding():
    state = AudioLoopState()
    en, cn = FakeEngine(), FakeEngine()
    chunk = np.full(2048, 0.1, dtype=np.float32)
    result = _run(chunk, state, "target", False, en, cn,
                  process_interval_samples=PI, max_buffer_samples=MAXB)
    assert len(state.audio_buffer) == 2048
    assert state.samples_since_last_process == 2048
    assert result.decoded is False
    assert en.decode_calls == 0


def test_chunks_accumulate_across_iterations():
    state = AudioLoopState()
    en, cn = FakeEngine(), FakeEngine()
    chunk = np.full(2048, 0.1, dtype=np.float32)
    for _ in range(3):
        _run(chunk, state, "target", False, en, cn,
             process_interval_samples=PI, max_buffer_samples=MAXB)
    assert len(state.audio_buffer) == 3 * 2048
    assert state.samples_since_last_process == 3 * 2048


# ----- decode gating ------------------------------------------------------

def test_decode_does_not_fire_below_process_interval():
    state = AudioLoopState()
    en, cn = FakeEngine(), FakeEngine()
    for _ in range(3):  # 3 * 2048 = 6144 < PI=6400
        result = _run(np.zeros(2048, dtype=np.float32), state, "target", False, en, cn,
                      process_interval_samples=PI, max_buffer_samples=MAXB)
        assert result.decoded is False
    assert en.decode_calls == 0


def test_decode_fires_once_process_interval_reached_and_counter_resets():
    state = AudioLoopState()
    en, cn = FakeEngine(), FakeEngine()
    for _ in range(3):
        _run(np.zeros(2048, dtype=np.float32), state, "target", False, en, cn,
             process_interval_samples=PI, max_buffer_samples=MAXB)
    result = _run(np.zeros(2048, dtype=np.float32), state, "target", False, en, cn,
                  process_interval_samples=PI, max_buffer_samples=MAXB)
    assert result.decoded is True
    assert en.decode_calls == 1
    assert state.samples_since_last_process == 0


def test_decode_receives_full_rolling_buffer_not_just_latest_chunk():
    state = AudioLoopState()
    en, cn = FakeEngine(), FakeEngine()
    chunks = [
        np.full(2048, 0.1, dtype=np.float32),
        np.full(2048, 0.2, dtype=np.float32),
        np.full(2048, 0.3, dtype=np.float32),
        np.full(2048, 0.4, dtype=np.float32),
    ]
    for c in chunks:
        _run(c, state, "target", False, en, cn,
             process_interval_samples=PI, max_buffer_samples=MAXB)
    received = en.last_stream.received_samples
    assert len(received) == 4 * 2048
    assert np.isclose(received[0], 0.1)
    assert np.isclose(received[2048], 0.2)
    assert np.isclose(received[-1], 0.4)


# ----- buffer trim ---------------------------------------------------------

def test_buffer_trims_to_max_buffer_samples():
    state = AudioLoopState()
    en, cn = FakeEngine(), FakeEngine()
    never, max_buf = 10_000_000, 5000
    _run(np.full(2048, 0.1, dtype=np.float32), state, "target", False, en, cn,
         process_interval_samples=never, max_buffer_samples=max_buf)
    _run(np.full(2048, 0.2, dtype=np.float32), state, "target", False, en, cn,
         process_interval_samples=never, max_buffer_samples=max_buf)
    assert len(state.audio_buffer) == 4096
    _run(np.full(2048, 0.9, dtype=np.float32), state, "target", False, en, cn,
         process_interval_samples=never, max_buffer_samples=max_buf)
    assert len(state.audio_buffer) == max_buf
    assert np.isclose(state.audio_buffer[-1], 0.9)


def test_buffer_never_exceeds_max_under_sustained_load():
    state = AudioLoopState()
    en, cn = FakeEngine(), FakeEngine()
    max_buf = 10_000
    for _ in range(100):
        _run(np.zeros(2048, dtype=np.float32), state, "target", False, en, cn,
             process_interval_samples=10_000_000, max_buffer_samples=max_buf)
    assert len(state.audio_buffer) == max_buf


# ----- state resets --------------------------------------------------------

def test_reset_for_new_slide_clears_buffer_counter_and_updates_slide():
    state = AudioLoopState()
    state.audio_buffer = np.full(5000, 0.5, dtype=np.float32)
    state.samples_since_last_process = 5000
    state.current_slide = 3
    state.reset_for_new_slide(7)
    assert state.audio_buffer.size == 0
    assert state.samples_since_last_process == 0
    assert state.current_slide == 7


def test_reset_after_trigger_clears_buffer_but_leaves_slide_index():
    state = AudioLoopState()
    state.audio_buffer = np.full(5000, 0.5, dtype=np.float32)
    state.samples_since_last_process = 5000
    state.current_slide = 3
    state.reset_after_trigger()
    assert state.audio_buffer.size == 0
    assert state.samples_since_last_process == 0
    assert state.current_slide == 3


# ----- engine routing ------------------------------------------------------

def test_english_target_routes_to_en_engine():
    state = AudioLoopState()
    en, cn = FakeEngine(), FakeEngine()
    _run(np.zeros(PI + 10, dtype=np.float32), state, "amazing grace", False, en, cn,
         process_interval_samples=PI, max_buffer_samples=MAXB)
    assert en.decode_calls == 1 and cn.decode_calls == 0


def test_chinese_target_routes_to_cn_engine():
    state = AudioLoopState()
    en, cn = FakeEngine(), FakeEngine()
    _run(np.zeros(PI + 10, dtype=np.float32), state, "奇異恩典", True, en, cn,
         process_interval_samples=PI, max_buffer_samples=MAXB)
    assert cn.decode_calls == 1 and en.decode_calls == 0


# ----- trigger decision ----------------------------------------------------

def test_above_threshold_decode_fires_trigger():
    state = AudioLoopState()
    en = FakeEngine(fake_text="amazing grace how sweet the sound")
    cn = FakeEngine()
    result = _run(np.zeros(PI + 10, dtype=np.float32), state,
                  "amazing grace how sweet the sound", False, en, cn,
                  process_interval_samples=PI, max_buffer_samples=MAXB, en_threshold=65)
    assert result.should_trigger is True
    assert result.score == 100
    assert result.pre_trigger_delay == 0.0


def test_below_threshold_decode_does_not_trigger():
    state = AudioLoopState()
    en = FakeEngine(fake_text="something totally unrelated to the lyric")
    cn = FakeEngine()
    result = _run(np.zeros(PI + 10, dtype=np.float32), state, "amazing grace", False, en, cn,
                  process_interval_samples=PI, max_buffer_samples=MAXB, en_threshold=65)
    assert result.should_trigger is False


def test_empty_decode_result_does_not_trigger():
    state = AudioLoopState()
    en = FakeEngine(fake_text="")
    cn = FakeEngine()
    result = _run(np.zeros(PI + 10, dtype=np.float32), state, "amazing grace", False, en, cn,
                  process_interval_samples=PI, max_buffer_samples=MAXB)
    assert result.decoded is True
    assert result.should_trigger is False
    assert result.heard_text is None
    assert result.score is None


# ----- VAD silence skip ----------------------------------------------------

def test_vad_skips_silent_decode():
    state = AudioLoopState()
    en = FakeEngine(fake_text="should never be decoded")
    cn = FakeEngine()
    # Pure silence past the interval, VAD on -> decode skipped
    result = process_audio_chunk(
        np.zeros(PI + 10, dtype=np.float32), state, "amazing grace", False, en, cn,
        process_interval_samples=PI, max_buffer_samples=MAXB, silence_threshold=0.005)
    assert result.vad_skipped is True
    assert result.decoded is False
    assert en.decode_calls == 0
    assert state.samples_since_last_process == 0  # counter reset so we don't spin


def test_vad_allows_loud_decode():
    state = AudioLoopState()
    en = FakeEngine(fake_text="hello")
    cn = FakeEngine()
    loud = np.full(PI + 10, 0.2, dtype=np.float32)  # RMS 0.2 >> 0.005
    result = process_audio_chunk(
        loud, state, "amazing grace", False, en, cn,
        process_interval_samples=PI, max_buffer_samples=MAXB, silence_threshold=0.005)
    assert result.vad_skipped is False
    assert result.decoded is True
    assert en.decode_calls == 1


# ----- dynamic decode window ----------------------------------------------

def test_dynamic_short_window_for_non_repeated_target():
    state = AudioLoopState()
    en, cn = FakeEngine(), FakeEngine()
    buf = np.full(10000, 0.2, dtype=np.float32)
    process_audio_chunk(buf, state, "amazing grace", False, en, cn,
                        process_interval_samples=PI, max_buffer_samples=16000,
                        short_buffer_samples=4000, silence_threshold=0.0)
    # non-repeated -> decode only the last 4000 samples
    assert len(en.last_stream.received_samples) == 4000


def test_full_window_for_repeated_target():
    state = AudioLoopState()
    en, cn = FakeEngine(), FakeEngine()
    buf = np.full(10000, 0.2, dtype=np.float32)
    process_audio_chunk(buf, state, "hey hey", False, en, cn,  # anchor "hey" x2
                        process_interval_samples=PI, max_buffer_samples=16000,
                        short_buffer_samples=4000, silence_threshold=0.0)
    # repeated -> use the full (max) window; buffer 10000 < 16000 so all 10000
    assert len(en.last_stream.received_samples) == 10000


def test_target_has_repeat_detection():
    assert target_has_repeat("走祢道路走祢道路", True) is True
    assert target_has_repeat("奇異恩典何等甘甜", True) is False
    assert target_has_repeat("hey hey", False) is True
    assert target_has_repeat("how sweet the sound", False) is False
