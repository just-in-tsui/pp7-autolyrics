# Audio Signal Detection & ProPresenter 7 Integration

This document explains, end to end, how PP7_AutoLyrics turns a live singer's voice
into a slide advance in **ProPresenter 7 (PP7)**. It covers the audio capture path,
the speech-to-text decode, the fuzzy match that decides *when* the current slide is
"finished," and how that decision is sent back to PP7 over its network API.

Everything runs **locally on CPU** — no audio leaves the machine. Speech
recognition is done by `sherpa-onnx` SenseVoice (int8 quantized).

---

## 1. The big picture

```
  ┌─────────────┐   16 kHz PCM    ┌────────────────┐   transcript   ┌──────────────┐
  │  Microphone │ ───────────────▶│  SenseVoice    │ ──────────────▶│ Fuzzy matcher│
  │  (PyAudio)  │   rolling buf   │  (sherpa-onnx) │   "heard text" │ (thefuzz)    │
  └─────────────┘                 └────────────────┘                └──────┬───────┘
                                                                            │ score ≥ threshold?
                                                                            ▼
  ┌─────────────┐   slide text    ┌────────────────┐   "advance!"   ┌──────────────┐
  │ ProPresenter│ ◀───────────────│  PP7 Poller    │◀───────────────│  Trigger     │
  │  7 (HTTP)   │   /next/trigger │  (target tail) │                │  decision    │
  └─────────────┘                 └────────────────┘                └──────────────┘
```

Three concurrent loops cooperate:

| Loop | Thread | Cadence | Job |
|------|--------|---------|-----|
| **PP7 poller** | background | ~150 ms | Track which slide is showing, cache the song text, expose the current slide's *target phrase*. |
| **Audio loop** | main / background | per chunk, decode every ~0.4 s | Capture mic audio, decode to text, score against the target, fire the advance. |
| **Keyboard listener** | background | event-driven | Manual nav, pause, fast/slow toggle, section-jump hotkeys. |

> **Code layout:** the original monolith is `PP7_SenseVoice_Only.py` (the runnable
> CLI reference). The same logic is now also factored into reusable modules:
> `pp7_poller.py` (PP7 state), `audio_pipeline.py` (pure per-chunk algorithm),
> `controller.py` (orchestration shared by the CLI and the Tkinter UI). The pure
> pipeline functions in `audio_pipeline.py` touch no hardware and no network, which
> makes the detection algorithm replayable from a WAV file in tests.

---

## 2. Audio capture

### 2.1 Stream parameters

The microphone is opened with PyAudio in a fixed format:

```python
stream = p.open(
    format=pyaudio.paInt16,   # 16-bit signed PCM
    channels=1,               # mono
    rate=16000,               # 16 kHz — SenseVoice's native rate
    input=True,
    input_device_index=mic_idx,
    frames_per_buffer=2048,
)
```

- **16 kHz mono** is what SenseVoice expects; no resampling is needed downstream.
- The operator picks a mic index at startup (Enter = system default). The resolved
  device name is shown so it's obvious if macOS silently rerouted to the built-in mic.

### 2.2 Backlog draining (staying real-time)

Each iteration reads **all available frames**, not a fixed 2048:

```python
avail = stream.get_read_available()
nframes = avail if avail > 2048 else 2048
data = stream.read(nframes, exception_on_overflow=False)
```

If a decode runs long and audio piles up inside PortAudio, reading a fixed chunk
would never catch up and latency would grow without bound. Consuming the whole
backlog guarantees we always analyze the **freshest** audio.

### 2.3 Int16 → float32 normalization

Raw bytes become a normalized float array in `[-1, 1]`:

```python
samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
```

### 2.4 The rolling buffer

Samples are appended to a rolling NumPy buffer capped at **8 seconds**
(`MAX_BUFFER_SIZE = 16000 * 8`):

```python
audio_buffer = np.concatenate((audio_buffer, samples))
if len(audio_buffer) > MAX_BUFFER_SIZE:
    audio_buffer = audio_buffer[-MAX_BUFFER_SIZE:]   # keep only the last 8 s
```

8 seconds is enough context for SenseVoice to produce a stable transcript and for
the repetition logic (below) to count repeated phrases in a chorus.

---

## 3. Speech-to-text (SenseVoice)

### 3.1 Two preloaded engines

Two SenseVoice recognizers are booted once at startup — one per language — so no
model loading happens in the hot path:

```python
rec_sv_en = load_sensevoice_engine("en")    # English
rec_sv_cn = load_sensevoice_engine("yue")   # Cantonese / Chinese
```

Both load the same int8 model (`model.int8.onnx`) with `num_threads=2`,
`use_itn=False`, `provider="cpu"`. The model folder
(`sherpa-onnx-sense-voice-...`) is gitignored (>150 MB) and shipped as a
`.tar.bz2` that must be extracted before first run.

Which engine decodes a given chunk depends on the **current slide's language**
(see §4.1) — not on auto-detection of the audio.

### 3.2 Decode cadence and window

A decode does **not** run on every chunk. It runs every `PROCESS_INTERVAL`
(`16000 * 0.4` samples ≈ 0.4 s of new audio):

```python
if samples_since_last_process >= PROCESS_INTERVAL:
    ...
    samples_since_last_process = 0
```

**Dynamic decode window.** A normal (non-repeating) line only needs the last few
seconds, which decodes roughly twice as fast as the full buffer. Lines that end in
a repeated phrase need the full 8 s buffer so the repetition can be counted:

```python
has_repeat = target_has_repeat(target, is_chinese)
decode_max = MAX_BUFFER_SIZE if has_repeat else SHORT_BUFFER_SIZE   # 8 s vs 4 s
decode_buffer = audio_buffer[-decode_max:] if len(audio_buffer) > decode_max else audio_buffer
```

> In the modular `audio_pipeline.py`, the short window is **opt-in** (defaults to
> the full buffer) because for dense Chinese text a mid-phrase fragment can become a
> perfect substring match and trigger early. Tune deliberately.

### 3.3 Running the decode

```python
s_stream = active_engine.create_stream()
s_stream.accept_waveform(16000, decode_buffer)
active_engine.decode_stream(s_stream)
heard_raw = s_stream.result.text
```

Decode latency is timed (`time.perf_counter()`) and logged for the per-session
performance summary. SenseVoice tags its output with markers like
`<|en|><|EMO|>…`; these are stripped before matching.

### 3.4 Optional VAD silence skip

A voice-activity gate can skip decoding when the recent audio is essentially
silence, saving CPU:

```python
rms = sqrt(mean(recent**2))
if rms < silence_threshold:      # default 0.0 = OFF
    return   # skipped, no decode this cycle
```

It is **off by default** because the gate is an *absolute* RMS threshold that
depends entirely on mic/interface gain — set too high and quiet inputs get fully
skipped ("hears nothing"). Opt in with `silence_threshold ≈ 0.005` for a hot mic.

---

## 4. From slide text to a "target phrase"

The matcher never compares against the whole slide — it compares against the
**tail** of the slide, i.e. what the singer says *last* before the slide should
advance. This is computed by the poller's `get_target()`.

### 4.1 Language detection (per slide, not per audio)

Detection is **character-based on the slide text**:

```python
chinese_chars = "".join(re.findall(r"[一-鿿]", current_full_text))
if len(chinese_chars) > 0:
    is_chinese_slide = True
    target = chinese_chars[-10:]      # last 10 Chinese chars, others stripped
else:
    is_chinese_slide = False
    target = current_full_text[-35:]  # last 35 characters
```

- **Any** CJK character flips the slide to Chinese mode and selects `rec_sv_cn`.
- The target is the **last 10 Chinese characters** (CN) or **last 35 characters**
  (EN). Punctuation/whitespace is normalized away earlier when the song is cached.

### 4.2 Target tightening (non-repeating lines)

For display and scoring, plain lines are tightened further to the *final clinch*
so `partial_ratio` scores the real endpoint instead of a long tail diluted by
mid-line content (`short_target()`): last ~6 chars (CN) or ~25 chars snapped to a
word boundary (EN). Repeated endings keep the full target so loops can be counted.

---

## 5. The trigger decision (`fast_smart_score`)

This is the heart of the system: given the SenseVoice transcript ("heard") and the
slide's target tail, produce a 0–100 score. The base is
`fuzz.partial_ratio(target, heard)` (a substring-aware fuzzy ratio), then **two
penalties** are subtracted.

### 5.1 Chinese path

```python
heard_cn   = "".join(re.findall(r"[一-鿿]", heard_clean))   # keep only Chinese
heard_trad = cc.convert(heard_cn)        # OpenCC Simplified → Traditional
score      = fuzz.partial_ratio(target, heard_trad)
```

- **Simplified → Traditional normalization.** PP7 slide text is typically
  Traditional; SenseVoice often emits Simplified. OpenCC (`s2t.json`) bridges them.
- **Reverent-pronoun normalization** (modular version): worship lyrics use 祢/祂/衪,
  which SenseVoice transcribes as the homophones 你/他 — the target is normalized to
  match before scoring.

### 5.2 English path

```python
clean_target = re.sub(r"[^\w\s]", "", target.lower())
clean_heard  = re.sub(r"[^\w\s]", "", heard_clean.lower())
score        = fuzz.partial_ratio(clean_target, clean_heard)
```

### 5.3 Penalty 1 — repetition enforcer

`partial_ratio` scores a *single* repeat of a looping phrase as a perfect match, so
it can't tell "sung once" from "sung the whole thing." For choruses like
"hallelujah hallelujah hallelujah" this would fire on the first word.

The fix: find the repeated anchor (last 4 chars for CN, last word for EN), count
how many times it appears in the target vs. the heard text, and penalize if the
singer hasn't reached the same count:

```python
if target_reps > 1 and heard_reps < target_reps:
    score -= 40
```

> The modular Chinese path replaces the flat −40 cliff with a **graduated** slope:
> a steeper per-missing-char penalty (`CN_MISSING_PENALTY_REPEATED = 25` vs the
> normal `6`) so a singer mid-loop is blocked while a *near-complete* final repeat
> still clears the threshold.

### 5.4 Penalty 2 — missing-content penalty

Discourages firing before the line is actually complete:

- **−6 per missing Chinese character** (`len(target) − len(heard)`)
- **−3 per missing English letter**

### 5.5 Thresholds

```python
EN_THRESHOLD = 65
CN_THRESHOLD = 55
```

Tuned for a plugged-in mic. In slow-song mode the **display** threshold is shown as
`+10`, but historically slow mode changed only the *delay caps*, not the score
threshold. (The modular controller surfaces the base thresholds in its status.)

A trigger is requested when `score >= threshold`.

---

## 6. Timing the advance

Crossing the threshold doesn't always fire immediately.

### 6.1 Dynamic pre-trigger delay

If the score is already over threshold but the heard text is **shorter** than the
target, the singer matched *early* (e.g., partial-ratio caught the phrase before
the last syllables). The script waits proportionally to how much is left:

```python
missing = max(0, len(target) - len(clean_heard))
# Chinese:  missing * 0.3 s  (cap 2.5 s)  | slow: * 0.5 s  (cap 3.5 s)
# English:  missing * 0.15 s (cap 2.5 s)  | slow: * 0.25 s (cap 4.0 s)
```

This lets the singer actually reach the end of the line before the slide flips.

### 6.2 Post-trigger settle + ghost-audio wipe

After firing, the loop sleeps **0.8 s** and then **wipes the audio buffer** and
drains the stream:

```python
handle_lyric_trigger()
time.sleep(0.8)                       # let the line's tail finish
audio_buffer = np.array([], ...)      # destroy "ghost audio"
samples_since_last_process = 0
stream.read(stream.get_read_available(), exception_on_overflow=False)
```

Without this, the trailing audio of the line just sung would still be in the buffer
and could instantly match the *new* slide, double-advancing. The modular controller
also sets a `fast_first` flag so the **first** decode after a reset uses a shorter
0.2 s window — detection resumes quickly on the new slide.

### 6.3 Stale-result guard (modular)

Because a decode takes time, the operator may change slides mid-decode. The
controller checks the slide index before and after and **discards** a result whose
slide is no longer showing, so a stale transcript can't advance the wrong slide.

---

## 7. Talking to ProPresenter 7

### 7.1 Requirements

PP7 must be running with its network server enabled:
**Settings → Network → on, port `1025`, no password.** All calls go to
`http://127.0.0.1:1025/v1/...`.

### 7.2 Reading state — the poller

The poller's `update_loop` runs ~every 150 ms:

1. `GET /v1/presentation/slide_index` → current slide index + presentation UUID.
2. **UUID change** ⇒ the active presentation changed ⇒ re-fetch the whole song via
   `GET /v1/presentation/active` and rebuild `slide_cache`
   (a list of `{text, group}` dicts).
3. **Index change** ⇒ update `current_full_text` to that slide's cached text.

Caching the whole song up front is what makes instant section-jump hotkeys possible
without re-querying PP7 each time.

```python
resp = session.get(f"{base_url}/presentation/slide_index", timeout=0.2)
idx, new_uuid = get_slide_info_smart(resp.json())
if new_uuid != current_uuid:        # song changed
    fetch_full_song()               # GET /presentation/active → slide_cache
if idx != last_index:               # slide changed
    current_full_text = slide_cache[idx]["text"]
```

### 7.3 Writing state — triggers

All advances are plain HTTP GETs:

| Action | Endpoint |
|--------|----------|
| Next slide | `GET /v1/presentation/active/next/trigger` |
| Previous slide | `GET /v1/presentation/active/previous/trigger` |
| Jump to slide *n* | `GET /v1/presentation/active/{n}/trigger` |
| Next / prev song | `GET /v1/playlist/active/next|previous/trigger` |

### 7.4 Plain advance vs. cued section jump

`handle_lyric_trigger()` decides *where* to go when a line finishes:

- **Mid-section** (next slide is the same group): just call `…/next/trigger`.
- **End-of-section** (next slide is a different group) **and** a `cued_slide_index`
  is set: jump to the cued destination instead.

This is how single-press section hotkeys work — pressing a hotkey once *queues* a
destination that fires only when the current section naturally finishes; a
double-press (within 0.4 s) jumps immediately. A "hold at section end" toggle can
also park the show on the last slide of a section until the operator presses →.

---

## 8. Why the loops swallow exceptions

The polling loop, the HTTP trigger functions, and the main audio loop deliberately
use `except: pass`. **This is intentional, not sloppy:**

- PP7's network API drops requests when a slide is mid-transition.
- PyAudio occasionally throws on buffer overflow.

In a live performance the loops **must keep running** through these transient
failures. Do not "fix" them by re-raising or adding logging in the hot path without
understanding this failure mode. Similarly, `force_quit` uses `os._exit(0)` because
a clean shutdown would hang on the background threads holding sockets and the audio
stream.

---

## 9. Tuning cheat-sheet

| Symptom | Likely knob |
|---------|-------------|
| Fires too early | Raise `EN/CN_THRESHOLD`; increase missing-content penalty; check target tightening. |
| Fires too late / misses | Lower threshold; shorten target; reduce pre-trigger delay caps. |
| Choruses fire on first line | Repetition enforcer — confirm the anchor (last 4 chars / last word) actually repeats in the slide. |
| Double-advances | Increase post-trigger settle (0.8 s) / confirm buffer wipe. |
| "NO AUDIO" / hears nothing | Wrong mic index; or VAD `silence_threshold` set too high for the mic gain. |
| Wrong language engine | Slide text language detection (`一-鿿`) — a stray CJK char flips the whole slide to Chinese. |
| Decode too slow (>600 ms) | Use the short decode window for non-repeating lines; confirm `num_threads`. |
| No advances at all | PP7 Network not enabled / wrong port / password set. |

---

## 10. Glossary

- **Target / target phrase** — the tail of the current slide's text the matcher
  listens for (last 10 CN chars or last 35 EN chars, optionally tightened).
- **Heard text** — SenseVoice's transcript of the rolling audio buffer.
- **Score** — `fuzz.partial_ratio` minus the repetition and missing-content
  penalties; compared against the language threshold.
- **Cued jump** — a section destination queued by a hotkey that fires when the
  current section ends.
- **Ghost audio** — trailing audio of the just-sung line that would falsely match
  the next slide if not wiped after a trigger.
