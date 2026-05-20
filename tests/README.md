# Tests

## Running

```bash
source ../optVenv/bin/activate                  # if not already
pip install pytest responses soundfile          # one-time

cd ..
pytest tests/ -v                                # all unit + mock + replay tests
pytest tests/test_fuzzy_score.py -v             # just unit tests (fast, no model)
pytest tests/test_replay_e2e.py -v              # replay against singing fixtures
python tests/bench_decode_latency.py            # SenseVoice timing report
```

The unit, mock, and benchmark tests run immediately. The **replay E2E** tests skip until you drop in singing fixtures (see below).

## Adding singing fixtures (TODO)

For each clip you want covered by the replay E2E suite, drop two files into `tests/fixtures/audio/` with matching basenames:

- `<name>.wav` — 16kHz mono int16, ~10–20 seconds of singing covering the end of one slide's lyrics
- `<name>.expected.json` — slide cache + expected trigger sample index

### 1. Capture and convert the WAV

```bash
ffmpeg -i source.m4a -ac 1 -ar 16000 -sample_fmt s16 tests/fixtures/audio/<name>.wav
```

Keep clips short (≤20s). Each replay decodes ~50 times for a 20s clip; longer clips just slow the test without adding coverage.

### 2. Find the expected trigger time

Open the WAV in any audio player. Note the **time in seconds** when the singer
finishes the slide's last word — that's when the slide should advance.

### 3. Write the `<name>.expected.json`

Use **seconds** (`*_sec`) — they're sample-rate-independent, so you read them
straight off any player with no math:

```jsonc
{
  "description": "Amazing Grace, end of verse 1 line 1",
  "slide_cache": [
    {"text": "Amazing grace how sweet the sound", "group": "Verse 1"},
    {"text": "That saved a wretch like me", "group": "Verse 1"}
  ],
  "starting_slide_index": 0,
  "expected_trigger_sec": 6.0,
  "tolerance_sec": 0.5,
  "no_false_positive_before_sec": 2.0
}
```

Field reference:
- `slide_cache` — copy-paste from a real `/v1/presentation/active` response, or hand-write
- `starting_slide_index` — which slide the singer is on at the start of the WAV
- `expected_trigger_sec` — when the singer finishes the slide's last word (slide should advance here)
- `tolerance_sec` — pass window (start with 0.5)
- `no_false_positive_before_sec` — assert no trigger fires before this time (optional; set well before `expected_trigger_sec` to detect over-eager matching)

> Sample-based equivalents (`expected_trigger_sample`, `tolerance_samples`,
> `no_false_positive_before_sample`) still work as a fallback, but seconds are
> preferred. If both are present, seconds win.

### 4. Bootstrapping the expected time

Don't know the trigger time yet? Put a rough guess in `expected_trigger_sec` and
run the test. On failure it prints exactly where the algorithm fired, e.g.:

```
Triggered at 12.89s (decision 12.29s + 0.60s delay), expected 6.00s ± 0.50s.
```

**Important:** set `expected_trigger_sec` to the time *you* hear the line end —
not to whatever the algorithm printed. If you just copy the algorithm's number,
the test becomes circular and proves nothing. The point is to catch the algorithm
firing at the *wrong* time, so the expected value must be measured independently.

## Suggested fixture set (priority order)

1. `english_line.wav` — EN happy path, one line ending
2. `cantonese_line.wav` — exercises the yue engine + opencc Simplified→Traditional
3. `english_chorus_repetition.wav` — line ending in repeated phrase ("hallelujah hallelujah hallelujah") — validates the repetition enforcer
4. *(optional)* `slow_ballad.wav` — slow tempo, exercises dynamic-delay branch
5. *(optional)* `near_miss.wav` — singer drops the last word or two — validates missing-char penalty

## Committing fixtures

WAVs are not gitignored by default. Decide before committing: small generic clips are fine to commit, but recordings of yourself / your team / copyrighted material should stay local — add `tests/fixtures/audio/*.wav` to `.gitignore` first.
