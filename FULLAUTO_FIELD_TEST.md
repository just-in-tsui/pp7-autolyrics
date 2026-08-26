# Full Auto — how to field test it

`PP7_SenseVoice_FullAuto.py` adds one thing to the baseline script: it works out
**where in the whole song the singer is**, instead of only listening for the end
of the slide currently on screen. That means it can jump to the right section by
itself, and it can catch an advance the tail-match missed.

Everything else — the trigger thresholds, the repetition enforcer, the dynamic
pre-trigger delay — is unchanged from `PP7_SenseVoice_Only.py` on purpose, so a
side-by-side test measures the new part and nothing else.

## Run it

```bash
source optVenv/bin/activate

python PP7_SenseVoice_FullAuto.py                 # full auto, driving PP7
python PP7_SenseVoice_FullAuto.py --shadow        # observe + log only, never touches PP7
python PP7_SenseVoice_FullAuto.py --no-auto       # baseline behaviour, localizer off
python PP7_SenseVoice_FullAuto.py --mic 2         # skip the mic prompt
```

Same prerequisites as the other scripts: ProPresenter 7 running with Network
enabled on port 1025, no password; the SenseVoice model folder extracted; and on
macOS, Accessibility permission for the terminal so the hotkeys work.

## New keys

| Key | Effect |
|-----|--------|
| `a` | Full auto on/off. Off is baseline behaviour — current slide only. |
| `j` | Shadow mode on/off. Logs what it *would* do without touching PP7. |
| `k` | Early-advance guard on/off (see below). |
| `l` | Re-locate now — forget accumulated confidence and re-read the song. |

All the existing keys are unchanged: `/` pause, `;` hold at section end, `'`
clear cued jump, `,` / `.` fast/slow, arrows for slide and song, and the section
hotkeys.

## Do the first test in shadow mode

**Run the first service with `--shadow` while an operator drives manually.** The
script listens and logs its conclusions but never moves the screen, so there is
no risk, and it produces the exact comparison you want:

> for every slide the operator moved to by hand, how long had the script already
> been sure of that slide?

That is the "can it beat a human scanning the monitor" number. It prints on exit
(Ctrl-C) and is recorded in the event log.

## Reading the results

Each run writes two files:

- `performance_fullauto_<date>.log` — the human-readable log
- `fullauto_events_<date>.jsonl` — one JSON object per decode cycle and decision

```bash
python analyze_fullauto_log.py fullauto_events_2026-08-26.jsonl
```

What to look at:

- **Lead time vs operator** — the headline. Positive means the script identified
  the slide before the operator got there. Look at the *median* and at "script
  was ahead on N%", not the best case.
- **Confidently located %** — how often it knew where it was at all. A low number
  usually means mic level or a deck whose slide text doesn't match what is sung.
- **Script moves the operator corrected** — the false-positive proxy. An operator
  moving the screen within a few seconds of a script move is them undoing it.
  This is the number that decides whether you can trust it live.

## Then run it live

Once shadow mode looks good, run it for real. Two settings worth knowing:

- **Hold at section end (`;`) still works, and full auto respects it.** With hold
  on, the script *cues* the section it hears instead of jumping to it — the
  screen doesn't move until the operator presses `→`. This is the recommended
  first live configuration: the operator keeps control, but no longer has to
  work out which section the band went to.
- **The early-advance guard (`k`, on by default)** lets the localizer veto a
  tail-match advance when it can tell the singer is only part-way through the
  line. If you see advances being held back that shouldn't be, turn it off with
  `k` and check `Advances vetoed as early` in the analysis.

## Tuning

The constants that decide how eager it is are grouped under `LOCALIZER TUNING`
at the top of the script. The two that matter most:

- `LOCATE_THRESHOLD_CN` / `LOCATE_THRESHOLD_EN` — how good a match has to be
  before it counts. Raise if it jumps to wrong sections; lower if it spends the
  service `SEARCHING`.
- `LOCATE_MARGIN` — how far the best slide must beat the runner-up. This is what
  keeps a shared phrase from pulling it into the wrong verse. Raise it if you see
  wrong-section jumps between lyrically similar sections.

`VOTE_REQUIRED` consecutive agreeing reads are needed before anything moves, so
the script costs roughly one extra decode cycle (~0.5 s) of latency in exchange
for not reacting to a single mis-transcription. Lowering it to 1 makes it faster
and considerably twitchier.

## Tests

```bash
pytest tests/test_fullauto_localizer.py tests/test_fullauto_runtime.py -v
```

These need no model, microphone or ProPresenter. They cover the decisions that
can move the screen on their own: ambiguity between repeated choruses, the
confidence gates, the operator-override behaviour, and the lead-time measurement.
