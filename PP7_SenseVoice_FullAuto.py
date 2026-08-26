"""PP7 auto-lyrics — FULL AUTO mode.

Difference from `PP7_SenseVoice_Only.py`: the baseline script only ever listens
for the *tail of the slide currently on screen*, so it can advance a slide but it
can never tell the operator **where in the song the singer actually is**. If the
band jumps from Verse 2 straight to the Bridge, the baseline sits on Verse 2
scoring near-zero forever and the operator has to spot it by eye and cue the
section by hand.

This script adds a second, independent subsystem — the **song-wide localizer**.
Every decode cycle it scores the heard audio against *every slide in the
presentation*, not just the current one, and works out which slide is being sung
and how far into it the singer is. Two things fall out of that:

  * **Auto section jumps.** Confident that the singer is somewhere other than the
    displayed slide → move PP7 there, no hotkey needed.
  * **A missed-advance safety net.** If the tuned tail-match misses the end of a
    line, the localizer notices the singer is already into the next slide and
    catches up.

The tail-match advance logic is deliberately *unchanged* from the baseline
(same thresholds, same repetition enforcer, same dynamic pre-trigger delay), so a
field test comparing the two scripts measures the localizer and nothing else.

Shadow mode (`--shadow`, or the `j` key) runs the localizer read-only: it logs
every decision it *would* have made while the operator drives manually. That is
the safe way to answer "does the tool spot the section faster than a human
scanning the monitor?" — see the lead-time stats in the exit summary.
"""

import argparse
import json
import logging
import os
import re
import signal
import sys
import termios
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import opencc
import pyaudio
import requests
import sherpa_onnx
from pynput import keyboard
from rapidfuzz import fuzz as rfuzz
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from thefuzz import fuzz

# ================= PERFORMANCE LOGGING SETUP =================
_today = datetime.now().strftime("%Y-%m-%d")
log_filename = f"performance_fullauto_{_today}.log"
events_filename = f"fullauto_events_{_today}.jsonl"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

perf_metrics = {
    'audio_latency_sum': 0.0,
    'audio_latency_count': 0,
    'fuzzy_latency_sum': 0.0,
    'fuzzy_latency_count': 0,
    'api_latency_sum': 0.0,
    'api_latency_count': 0,
    'decode_cycles': 0,
    'auto_jumps': 0,          # localizer moved PP7 to a non-adjacent slide
    'auto_rescues': 0,        # localizer caught a missed advance (delta == 1)
    'tail_advances': 0,       # normal end-of-line advance (same as baseline)
    'vetoed_advances': 0,     # localizer blocked a tail advance as too early
    'shadow_jumps': 0,        # jumps suppressed because shadow mode was on
    'lost_cycles': 0,         # decodes where no slide was confidently matched
    'lead_times': [],         # seconds the localizer beat the operator by
}

# Structured event stream for post-service analysis. One JSON object per line.
_events_lock = threading.Lock()
_events_fh = None


def log_event(kind, **payload):
    """Append one structured event. Field-test analysis reads this file, not the
    human-readable log. Never allowed to break the audio loop."""
    if _events_fh is None:
        return
    payload['t'] = round(time.time(), 3)
    payload['kind'] = kind
    try:
        with _events_lock:
            _events_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            _events_fh.flush()
    except Exception:
        pass


# ================= CONFIGURATION =================
PP_HOST = "127.0.0.1"
PP_PORT = 1025
EN_THRESHOLD = 65  # 65 when plugged in
CN_THRESHOLD = 55  # 55 when plugged in
MODEL_DIR_SV = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"

# ---------------- LOCALIZER TUNING ----------------
# A candidate slide's *adjusted* score must clear this to count as a real read.
LOCATE_THRESHOLD_CN = 62
LOCATE_THRESHOLD_EN = 68
# ...and must beat the best non-duplicate rival by this much. This is what stops
# a shared phrase ("praise the Lord") from yanking us to the wrong verse.
LOCATE_MARGIN = 8
VOTE_WINDOW = 4            # decode cycles retained in the rolling vote
VOTE_REQUIRED = 2          # agreeing reads needed before we touch PP7
LOCATE_COOLDOWN = 1.5      # sec of localizer silence after any slide movement
LOST_CYCLES = 6            # unconfident decodes before we declare SEARCHING

# How much of the transcript tail the localizer looks at. Sized to roughly one
# sung line so "coverage" and "recency" below stay meaningful.
HEARD_WINDOW_CN = 16
HEARD_WINDOW_EN = 55
MIN_HEARD_CN = 4           # below this the transcript can match anything
MIN_HEARD_EN = 10

# Penalties applied to the raw partial_ratio (see score_slide for the reasoning).
COVERAGE_PENALTY = 25      # slide explains only part of what we heard
RECENCY_PENALTY = 15       # slide matches the *start* of what we heard, not the end

# Positional prior — staying put and stepping forward are overwhelmingly the
# common cases, so anything else has to earn it.
STICKY_BONUS = 8
NEXT_BONUS = 4
FORWARD_SKIP_PENALTY = 3
BACKWARD_PENALTY = 6

NEXT_MIN_PROGRESS = 0.35   # delta==1 rescue: singer must be this far into the next slide
ADVANCE_MIN_PROGRESS = 0.45  # below this the localizer vetoes a tail advance
PROBE_EVERY = 4            # mixed-language songs: cycles between other-engine probes
MAX_LEAD_SEC = 30.0        # older sightings aren't a lead over the operator, just noise
TOP_N_DISPLAY = 3

# ---------------- HOTKEY CONFIGURATION ----------------
HOTKEYS = {
    'u': 'English Verse 1', 'i': 'English Verse 2', 'o': 'English Verse 3', 'p': 'English Verse 4',
    't': 'English Pre-Chorus 1', 'g': 'English Pre-Chorus 2',
    'y': 'English Chorus 1', 'h': 'English Chorus 2',
    'r': 'English Bridge', 'w': 'English Ending',
    'v': 'Chinese Verse 1', 'b': 'Chinese Verse 2', 'n': 'Chinese Verse 3', 'm': 'Chinese Verse 4',
    'x': 'Chinese Pre-Chorus 1', 'd': 'Chinese Pre-Chorus 2',
    'c': 'Chinese Chorus 1', 'f': 'Chinese Chorus 2',
    's': 'Chinese Bridge', 'z': 'Chinese Ending'
}
# =================================================

cc = opencc.OpenCC('s2t.json')

CJK_RE = re.compile(r'[\u4e00-\u9FFF]')
TAG_RE = re.compile(r'<\|.*?\|>')
PUNCT_RE = re.compile(r'[^\w\s]')


# ================= NORMALIZATION =================
def normalize_slide_text(text):
    """Reduce a slide to the comparable form for its language.

    Returns (normalized, is_chinese). Language detection matches the baseline
    script: any CJK character at all makes the slide Chinese, and the Latin
    characters around it are dropped (PP7 slides routinely carry a romanised
    subtitle the singer never actually sings)."""
    cjk = "".join(CJK_RE.findall(text or ""))
    if cjk:
        return cjk, True
    clean = " ".join(PUNCT_RE.sub('', (text or "").lower()).split())
    return clean, False


def normalize_heard(raw, is_chinese):
    """Put a SenseVoice transcript into the same space as normalize_slide_text.
    Chinese goes through opencc Simplified->Traditional because PP7 slide text
    is typically Traditional while the recogniser emits Simplified."""
    heard = TAG_RE.sub('', raw or "").strip()
    if is_chinese:
        cjk = "".join(CJK_RE.findall(heard))
        return cc.convert(cjk) if cjk else ""
    return " ".join(PUNCT_RE.sub('', heard.lower()).split())


# ================= SONG-WIDE LOCALIZER =================
@dataclass
class SlideEntry:
    index: int
    group: str
    raw: str
    norm: str
    is_chinese: bool


@dataclass
class Candidate:
    index: int
    group: str
    score: float        # adjusted — priors and penalties applied
    raw_score: float    # bare partial_ratio, for the log
    progress: float     # 0..1, how far through this slide the match ends
    coverage: float     # fraction of the heard tail this slide explains


@dataclass
class LocateDecision:
    """What the localizer saw this cycle and what it wants done about it."""
    candidates: list = field(default_factory=list)   # top N, best first
    winner: Candidate | None = None                  # best candidate, confident or not
    confident: bool = False
    margin: float = 0.0
    action: str = "none"        # none | jump | next
    target_index: int | None = None
    reason: str = ""


def score_slide(entry, heard):
    """Score one slide against the heard tail, and say where in the slide it landed.

    `partial_ratio_alignment` gives us both halves of the answer at once: the
    similarity, and the span of the *slide* the transcript aligned to. That span
    is what tells us whether the singer is at the top of the line or the end of
    it — which the baseline's tail-only match can never know.

    Two corrections are applied to the raw ratio:

    * **coverage** — partial_ratio happily returns 100 when a two-word slide
      appears inside a long transcript. Penalising unexplained transcript stops
      short slides from out-ranking the longer slide that accounts for all of it.
    * **recency** — if a slide matches the *beginning* of the transcript, the
      singer has already sung past it. We want where they are now, not where
      they were a line ago.
    """
    if not heard or not entry.norm:
        return None
    if len(heard) < (MIN_HEARD_CN if entry.is_chinese else MIN_HEARD_EN):
        return None

    al = rfuzz.partial_ratio_alignment(heard, entry.norm)
    if al is None:
        return None

    span = max(1, al.src_end - al.src_start)
    coverage = min(1.0, span / len(heard))
    recency = al.src_end / len(heard) if len(heard) else 1.0
    progress = min(1.0, al.dest_end / len(entry.norm)) if entry.norm else 0.0

    adjusted = (al.score
                - COVERAGE_PENALTY * (1.0 - coverage)
                - RECENCY_PENALTY * (1.0 - recency))

    return Candidate(
        index=entry.index,
        group=entry.group,
        score=adjusted,
        raw_score=al.score,
        progress=progress,
        coverage=coverage,
    )


class SongLocalizer:
    """Tracks which slide of the whole song is being sung right now.

    Deliberately holds no PP7 or audio state — it is fed transcripts and the
    current slide index, and returns a decision. That keeps it unit-testable
    without a model, a microphone, or ProPresenter running."""

    def __init__(self):
        self.entries = []
        self.dup_groups = {}     # normalized text -> [slide indices sharing it]
        self.languages = set()   # {'cn'}, {'en'} or both
        self.votes = deque(maxlen=VOTE_WINDOW)
        self.unconfident_streak = 0
        self.cooldown_until = 0.0
        self.first_confident = {}   # slide index -> when we first became sure of it
        self.last_decision = LocateDecision()

    # ----- index building -------------------------------------------------
    def build(self, slide_cache):
        """(Re)build the search index. Called whenever the presentation changes."""
        entries, dups, langs = [], {}, set()
        for idx, slide in enumerate(slide_cache):
            norm, is_cn = normalize_slide_text(slide.get('text', ''))
            entries.append(SlideEntry(
                index=idx,
                group=slide.get('group', 'Default'),
                raw=slide.get('text', ''),
                norm=norm,
                is_chinese=is_cn,
            ))
            if norm:
                dups.setdefault(norm, []).append(idx)
                langs.add('cn' if is_cn else 'en')
        self.entries = entries
        # Only genuinely indistinguishable slides matter — a chorus repeated
        # verbatim three times can never be told apart by audio alone, so we
        # resolve those by position instead of pretending one of them "won".
        self.dup_groups = {k: v for k, v in dups.items() if len(v) > 1}
        self.languages = langs
        self.reset()

    def reset(self):
        """Forget accumulated confidence — after a song change or a manual jump."""
        self.votes.clear()
        self.unconfident_streak = 0
        self.first_confident.clear()
        self.last_decision = LocateDecision()

    @property
    def is_lost(self):
        return self.unconfident_streak >= LOST_CYCLES

    def dup_peers(self, index):
        """Slide indices whose text is identical to this one (including itself)."""
        for idx_list in self.dup_groups.values():
            if index in idx_list:
                return idx_list
        return [index]

    # ----- scoring --------------------------------------------------------
    def rank(self, heard_cn, heard_en, current_index):
        """Score every slide and return candidates best-first.

        Each slide is compared using the transcript from the engine that matches
        *its own* language, so a bilingual deck ranks its English and Chinese
        slides on the same scale."""
        heard_cn = (heard_cn or "")[-HEARD_WINDOW_CN:]
        heard_en = (heard_en or "")[-HEARD_WINDOW_EN:]

        scored = []
        for entry in self.entries:
            heard = heard_cn if entry.is_chinese else heard_en
            cand = score_slide(entry, heard)
            if cand is None:
                continue
            cand.score += self._positional_prior(cand.index, current_index)
            scored.append(cand)

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored

    @staticmethod
    def _positional_prior(index, current_index):
        if current_index is None or current_index < 0:
            return 0.0
        delta = index - current_index
        if delta == 0:
            return STICKY_BONUS
        if delta == 1:
            return NEXT_BONUS
        if delta > 1:
            return -FORWARD_SKIP_PENALTY
        return -BACKWARD_PENALTY

    def _resolve_duplicate(self, winner, current_index):
        """Pick which copy of an identical-text slide the singer is most likely on.

        Verbatim-repeated choruses are acoustically identical, so position is the
        only evidence there is: prefer the nearest copy, and prefer forward over
        backward, because songs move forward."""
        peers = self.dup_peers(winner.index)
        if len(peers) < 2 or current_index is None or current_index < 0:
            return winner.index

        def cost(idx):
            delta = idx - current_index
            if delta == 0:
                return -1                 # already there — always cheapest
            return delta if delta > 0 else (-delta) * 2

        return min(peers, key=cost)

    # ----- decision -------------------------------------------------------
    def observe(self, heard_cn, heard_en, current_index, now=None):
        """Fold one decode cycle into the position estimate and decide what to do.

        Confidence is deliberately three-legged — a single loud transcript is
        never enough to move a live service:
          1. the winner clears an absolute score floor,
          2. it beats the best *distinguishable* rival by LOCATE_MARGIN,
          3. it wins VOTE_REQUIRED of the last VOTE_WINDOW cycles.
        """
        now = time.time() if now is None else now
        decision = LocateDecision()

        candidates = self.rank(heard_cn, heard_en, current_index)
        decision.candidates = candidates[:TOP_N_DISPLAY]
        if not candidates:
            self.votes.append(None)
            self.unconfident_streak += 1
            self.last_decision = decision
            return decision

        winner = candidates[0]
        decision.winner = winner

        entry = self.entries[winner.index]
        floor = LOCATE_THRESHOLD_CN if entry.is_chinese else LOCATE_THRESHOLD_EN

        # Rivals that share the winner's exact text aren't rivals — they're the
        # same lyric printed twice, and position (not audio) decides between them.
        peers = set(self.dup_peers(winner.index))
        rival = next((c for c in candidates[1:] if c.index not in peers), None)
        decision.margin = winner.score - rival.score if rival else 100.0

        decision.confident = winner.score >= floor and decision.margin >= LOCATE_MARGIN

        if decision.confident:
            target = self._resolve_duplicate(winner, current_index)
            decision.target_index = target
            self.votes.append(target)
            self.unconfident_streak = 0
            self.first_confident.setdefault(target, now)
        else:
            self.votes.append(None)
            self.unconfident_streak += 1
            decision.reason = (
                f"score {winner.score:.0f} < {floor}" if winner.score < floor
                else f"margin {decision.margin:.0f} < {LOCATE_MARGIN}"
            )
            self.last_decision = decision
            return decision

        target = decision.target_index
        # Votes must be *consecutive*. A simple count over the window lets a
        # flapping recogniser (A, B, A) reach quorum for A even though it never
        # settled — exactly the reading that would yank the screen mid-line.
        streak = 0
        for vote in reversed(self.votes):
            if vote != target:
                break
            streak += 1
        if streak < VOTE_REQUIRED:
            decision.reason = f"{streak}/{VOTE_REQUIRED} consecutive votes"
            self.last_decision = decision
            return decision

        if now < self.cooldown_until:
            decision.reason = "cooldown"
            self.last_decision = decision
            return decision

        # After duplicate resolution the acted-on slide may not be the raw winner,
        # so report the group we're actually moving to.
        target_group = self.entries[target].group if 0 <= target < len(self.entries) else "?"
        delta = target - current_index if current_index is not None else 0
        if delta == 0:
            decision.reason = "on the right slide"
        elif delta == 1:
            # The tuned tail-match owns normal end-of-line advances. We only step
            # in once the singer is audibly *into* the next slide, which means the
            # tail-match missed its cue — a rescue, not a race.
            if winner.progress >= NEXT_MIN_PROGRESS:
                decision.action = "next"
                decision.reason = f"missed advance — {winner.progress:.0%} into next slide"
            else:
                decision.reason = f"next slide only {winner.progress:.0%} in — leaving it to tail match"
        else:
            decision.action = "jump"
            decision.reason = f"singer is on slide {target} ({target_group})"

        self.last_decision = decision
        return decision

    def note_movement(self, new_index, now=None, source="operator"):
        """Called whenever PP7's slide actually changes, whoever caused it.

        Starts the cooldown (the buffer still holds the previous line's audio) and,
        when the move came from a human, records how far ahead of them the
        localizer had already been. That lead time is the number the field test
        is really after."""
        now = time.time() if now is None else now
        self.cooldown_until = now + LOCATE_COOLDOWN
        self.votes.clear()

        lead = None
        seen_at = self.first_confident.pop(new_index, None)
        if source == "operator" and seen_at is not None:
            candidate_lead = now - seen_at
            # A stale sighting from minutes ago isn't a lead, it's a coincidence.
            if 0 <= candidate_lead <= MAX_LEAD_SEC:
                lead = candidate_lead
                perf_metrics['lead_times'].append(lead)
        # Sightings of other slides are stale the moment the song moves on.
        self.first_confident.clear()
        return lead


# ================= RUNTIME STATE =================
poller = None
localizer = SongLocalizer()
cued_slide_index = None
last_key_press = {}
DOUBLE_PRESS_DELAY = 0.4

is_paused = False
is_slow_mode = False
stop_at_section_end = False   # when True, auto-advance holds at section boundaries
held_at_index = None          # slide we're holding on (suppresses repeat fires/messages)

auto_locate = True            # master switch for the song-wide localizer
shadow_mode = False           # localizer observes and logs but never touches PP7
advance_guard = True          # localizer may veto a tail advance that looks too early

# Attribution for slide movements, so lead-time stats only count the operator's
# own moves and not the ones this script made.
last_self_action = {'index': None, 'ts': 0.0}
SELF_ACTION_WINDOW = 2.0


@dataclass
class UIState:
    """Shared snapshot the rich dashboard renders. Background threads (keyboard
    listener, poller) mutate fields here instead of printing — only the main loop
    draws, so a fixed panel never gets corrupted by interleaved thread output."""
    mic_name: str = "—"
    input_level: float = 0.0
    last_sound_ts: float = 0.0
    decode_ms: float = 0.0
    heard: str = ""
    score: int | None = None
    last_action: str = "Ready"
    locate_summary: str = ""
    candidates: list = field(default_factory=list)
    progress: float | None = None


ui = UIState()
live = None
_saved_termios = None


def set_action(msg):
    ui.last_action = msg


def mark_self_action(index):
    """Record that *we* asked PP7 to land on `index`, so the resulting movement
    isn't mistaken for the operator reacting faster than us."""
    last_self_action['index'] = index
    last_self_action['ts'] = time.time()


def movement_source(index, ts):
    if (last_self_action['index'] == index
            and 0 <= ts - last_self_action['ts'] <= SELF_ACTION_WINDOW):
        return "script"
    return "operator"


# ================= TERMINAL / EXIT =================
def disable_terminal_echo():
    """Suppress local terminal echo while the rich dashboard is active.
    pynput's section-jump letters would otherwise be echoed to stdout when the
    terminal has focus, corrupting Live's cursor tracking."""
    global _saved_termios
    try:
        fd = sys.stdin.fileno()
        _saved_termios = termios.tcgetattr(fd)
        attrs = termios.tcgetattr(fd)
        attrs[3] &= ~termios.ECHO  # lflag
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        _saved_termios = None


def restore_terminal_echo():
    global _saved_termios
    if _saved_termios is None:
        return
    try:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, _saved_termios)
    except Exception:
        pass
    _saved_termios = None


def _avg(sum_key, count_key):
    if perf_metrics[count_key] > 0:
        return perf_metrics[sum_key] / perf_metrics[count_key]
    return None


def print_summary():
    print("\n📊 --- Performance Session Summary ---")
    avg_audio = _avg('audio_latency_sum', 'audio_latency_count')
    avg_fuzzy = _avg('fuzzy_latency_sum', 'fuzzy_latency_count')
    avg_api = _avg('api_latency_sum', 'api_latency_count')
    print(f"🎙️  Avg Audio-to-Text Latency: {avg_audio:.2f} ms" if avg_audio is not None
          else "🎙️  Avg Audio-to-Text Latency: N/A")
    print(f"🧠 Avg Localizer + Fuzzy Latency: {avg_fuzzy:.2f} ms" if avg_fuzzy is not None
          else "🧠 Avg Localizer + Fuzzy Latency: N/A")
    print(f"🌐 Avg API Trigger Latency: {avg_api:.2f} ms" if avg_api is not None
          else "🌐 Avg API Trigger Latency: N/A")

    print("\n🎯 --- Full-Auto Localizer ---")
    cycles = perf_metrics['decode_cycles']
    lost = perf_metrics['lost_cycles']
    located = cycles - lost
    print(f"   Decode cycles           : {cycles}")
    print(f"   Located / lost          : {located} / {lost}"
          + (f"  ({located / cycles:.0%} located)" if cycles else ""))
    print(f"   Tail advances           : {perf_metrics['tail_advances']}")
    print(f"   Auto section jumps      : {perf_metrics['auto_jumps']}")
    print(f"   Missed-advance rescues  : {perf_metrics['auto_rescues']}")
    print(f"   Advances vetoed (early) : {perf_metrics['vetoed_advances']}")
    if perf_metrics['shadow_jumps']:
        print(f"   Shadow-mode suppressed  : {perf_metrics['shadow_jumps']}")

    # The headline field-test number: how long the localizer had already been
    # sure of a slide before the operator got PP7 there by hand.
    leads = perf_metrics['lead_times']
    print("\n⏱️  --- Lead Time vs Operator ---")
    if leads:
        ordered = sorted(leads)
        median = ordered[len(ordered) // 2]
        print(f"   Operator moves measured : {len(leads)}")
        print(f"   Median lead             : {median:+.2f} s")
        print(f"   Mean lead               : {sum(leads) / len(leads):+.2f} s")
        print(f"   Best / worst            : {ordered[-1]:+.2f} s / {ordered[0]:+.2f} s")
        print("   (positive = script identified the slide before the operator got there)")
    else:
        print("   No operator-driven moves recorded — run with --shadow during a")
        print("   manually-operated service to collect this comparison.")

    print(f"\n📄 Structured events: {events_filename}")
    print("--------------------------------------\n")


def force_quit(sig, frame):
    try:
        if live is not None:
            live.stop()
    except Exception:
        pass
    restore_terminal_echo()
    print("\n\n🛑 Force terminating the script...")
    log_event('session_end', metrics={k: v for k, v in perf_metrics.items()
                                      if k != 'lead_times'},
              lead_times=[round(x, 3) for x in perf_metrics['lead_times']])
    try:
        if _events_fh:
            _events_fh.close()
    except Exception:
        pass
    print_summary()
    os._exit(0)


signal.signal(signal.SIGINT, force_quit)


# ================= SENSEVOICE ENGINES =================
def load_sensevoice_engine(target_language):
    if not os.path.exists(MODEL_DIR_SV):
        print(f"❌ ERROR: Missing '{MODEL_DIR_SV}' folder.")
        sys.exit(1)

    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{MODEL_DIR_SV}/model.int8.onnx",
        tokens=f"{MODEL_DIR_SV}/tokens.txt",
        num_threads=2,
        use_itn=False,
        language=target_language,
        provider="cpu",
    )


# ================= PP7 POLLER =================
class PP7SmartPoller:
    """Same slide/presentation tracker as the baseline, plus a movement journal.

    Every observed slide change is queued rather than acted on here: the
    localizer is owned by the audio thread, so letting this thread mutate it
    would race. `main` drains the journal each cycle."""

    def __init__(self):
        self.lock = threading.RLock()
        self.session = requests.Session()
        self.base_url = f"http://{PP_HOST}:{PP_PORT}/v1"
        self.current_full_text = ""
        self.current_index = -1
        self.current_uuid = None
        self.slide_cache = []
        self.is_chinese_slide = False
        self.connected = False
        self.cache_version = 0          # bumped whenever slide_cache is replaced
        self.movements = deque(maxlen=64)  # (index, timestamp) of observed changes

    def fetch_full_song(self):
        try:
            resp = self.session.get(f"{self.base_url}/presentation/active", timeout=1)
            if resp.status_code == 200:
                data = resp.json()
                new_cache = []
                for group in data.get('presentation', {}).get('groups', []):
                    group_name = group.get('name', 'Default')
                    for slide in group.get('slides', []):
                        raw = slide.get('text', '')
                        clean = " ".join(raw.replace('\r', ' ').replace('\n', ' ').split())
                        new_cache.append({"text": clean, "group": group_name})
                self.slide_cache = new_cache
                self.cache_version += 1
                return True
        except: pass
        return False

    def get_slide_info_smart(self, data):
        idx, uuid = -1, None
        s_idx = data.get('slide_index')
        p_index = data.get('presentation_index')
        if isinstance(s_idx, int) and s_idx > -1: idx = s_idx
        elif isinstance(p_index, dict): idx = p_index.get('index', -1)
        if isinstance(p_index, dict): uuid = p_index.get('presentation_id', {}).get('uuid')
        return idx, uuid

    def get_target(self):
        with self.lock:
            if not self.current_full_text: return "", self.current_index
            chinese_chars = "".join(CJK_RE.findall(self.current_full_text))

            if len(chinese_chars) > 0:
                self.is_chinese_slide = True
                target = chinese_chars[-10:] if len(chinese_chars) > 10 else chinese_chars
                return target, self.current_index
            else:
                self.is_chinese_slide = False
                text = self.current_full_text
                target = text[-35:] if len(text) > 35 else text
                return target.strip(), self.current_index

    def slide_language(self, index):
        """'cn' or 'en' for a cached slide — drives which engine we decode with."""
        with self.lock:
            if 0 <= index < len(self.slide_cache):
                _, is_cn = normalize_slide_text(self.slide_cache[index]['text'])
                return 'cn' if is_cn else 'en'
        return 'en'

    def update_loop(self):
        last_index = -999
        while True:
            try:
                resp = self.session.get(f"{self.base_url}/presentation/slide_index", timeout=0.2)
                if resp.status_code == 200:
                    self.connected = True
                    idx, new_uuid = self.get_slide_info_smart(resp.json())
                    with self.lock:
                        if new_uuid and new_uuid != self.current_uuid:
                            self.current_uuid = new_uuid
                            self.fetch_full_song()
                            last_index = -999

                        if idx != last_index:
                            self.current_index = idx
                            if 0 <= idx < len(self.slide_cache):
                                self.current_full_text = self.slide_cache[idx]["text"]
                            else:
                                self.current_full_text = ""
                            if last_index != -999 and idx >= 0:
                                self.movements.append((idx, time.time()))
                            last_index = idx
            except Exception:
                self.connected = False
            time.sleep(0.15)


# ================= PP7 ACTIONS =================
def trigger_api(endpoint, message):
    try:
        requests.get(f"http://{PP_HOST}:{PP_PORT}{endpoint}")
        set_action(message)
    except: pass


def trigger_slide(index):
    try:
        t0 = time.perf_counter()
        requests.get(f"http://{PP_HOST}:{PP_PORT}/v1/presentation/active/{index}/trigger")
        perf_metrics['api_latency_sum'] += (time.perf_counter() - t0) * 1000
        perf_metrics['api_latency_count'] += 1
        set_action(f"Jumped to slide {index}")
    except: pass


def trigger_next():
    try:
        t0 = time.perf_counter()
        requests.get(f"http://{PP_HOST}:{PP_PORT}/v1/presentation/active/next/trigger")
        perf_metrics['api_latency_sum'] += (time.perf_counter() - t0) * 1000
        perf_metrics['api_latency_count'] += 1
    except: pass


def is_end_of_section():
    """True if the poller's current slide is the last one in its group (or the
    final slide of the deck)."""
    if poller is None:
        return False
    with poller.lock:
        current_idx = poller.current_index
        if 0 <= current_idx < len(poller.slide_cache):
            current_group = poller.slide_cache[current_idx]['group']
            next_idx = current_idx + 1
            if next_idx < len(poller.slide_cache):
                return current_group != poller.slide_cache[next_idx]['group']
            return True
    return False


def handle_lyric_trigger(origin="tail"):
    """End-of-line advance. Byte-for-byte the baseline's behaviour, so a field
    test comparing the two scripts measures the localizer and nothing else.

    `origin` only changes which counter the move lands in: "tail" is a normal
    end-of-line match, "rescue" is the localizer catching an advance the tail
    match missed. Both go through this one path so a rescue still honours a cued
    section jump and the hold-at-section-end toggle."""
    global cued_slide_index, poller, held_at_index
    end_of_section = is_end_of_section()
    with poller.lock:
        current_idx = poller.current_index

    if stop_at_section_end and end_of_section:
        if held_at_index != current_idx:
            held_at_index = current_idx
            set_action(f"Holding at end of section (slide {current_idx}) — press → to continue")
        return

    if cued_slide_index is not None and end_of_section:
        mark_self_action(cued_slide_index)
        trigger_slide(cued_slide_index)
        set_action(f"Section ended — executed cued jump to slide {cued_slide_index}")
        log_event('advance', mode='cued_jump', to=cued_slide_index, origin=origin)
        cued_slide_index = None
    else:
        mark_self_action(current_idx + 1)
        trigger_next()
        set_action("Triggered next slide")
        log_event('advance', mode='next', to=current_idx + 1, origin=origin)
    perf_metrics['auto_rescues' if origin == "rescue" else 'tail_advances'] += 1


def jump_to_group(group_name, immediate):
    global cued_slide_index, poller
    target_idx = -1

    with poller.lock:
        for idx, slide in enumerate(poller.slide_cache):
            if group_name.lower() == slide['group'].lower():
                target_idx = idx
                break
        if target_idx == -1:
            for idx, slide in enumerate(poller.slide_cache):
                if group_name.lower() in slide['group'].lower():
                    target_idx = idx
                    break

    if target_idx != -1:
        if immediate:
            mark_self_action(target_idx)
            trigger_slide(target_idx)
            set_action(f"Jumped now to {group_name} (slide {target_idx})")
            cued_slide_index = None
        else:
            cued_slide_index = target_idx
            set_action(f"Cued {group_name} (slide {target_idx}) — jumps after this section")


# ================= TAIL-MATCH ADVANCE (unchanged from baseline) =================
def fast_smart_score(heard_text, target, is_chinese_slide):
    if not heard_text or not target: return 0, heard_text
    heard_clean = TAG_RE.sub('', heard_text).strip()

    if is_chinese_slide:
        heard_cn = "".join(CJK_RE.findall(heard_clean))
        if len(heard_cn) < 2: return 0, heard_cn
        heard_trad = cc.convert(heard_cn)

        score = fuzz.partial_ratio(target, heard_trad)

        # --- THE ADVANCED REPETITION ENFORCER (CHINESE) ---
        anchor_phrase = target[-4:] if len(target) >= 4 else target
        target_reps = target.count(anchor_phrase)
        heard_reps = heard_trad.count(anchor_phrase)

        if target_reps > 1 and heard_reps < target_reps:
            score -= 40  # Massive penalty: Singer is only on the first loop

        missing_chars = len(target) - len(heard_trad)
        if missing_chars > 0:
            score -= (missing_chars * 6)

        return score, heard_trad

    else:
        if len(heard_clean) < 5: return 0, heard_clean

        clean_target = PUNCT_RE.sub('', target.lower())
        clean_heard = PUNCT_RE.sub('', heard_clean.lower())

        score = fuzz.partial_ratio(clean_target, clean_heard)

        # --- THE ADVANCED REPETITION ENFORCER (ENGLISH) ---
        target_words = clean_target.split()
        if len(target_words) > 1:
            anchor_word = target_words[-1]
            target_reps = clean_target.count(anchor_word)
            heard_reps = clean_heard.count(anchor_word)

            if target_reps > 1 and heard_reps < target_reps:
                score -= 40  # Massive penalty

        missing_letters = len(clean_target) - len(clean_heard)
        if missing_letters > 0:
            score -= (missing_letters * 3)

        return score, clean_heard


def target_has_repeat(target, is_chinese_slide):
    """Does the slide's target end in a repeated phrase? If so we must decode the
    full buffer so the repetition enforcer can count loops; otherwise the shorter,
    faster decode window is safe."""
    if not target:
        return False
    if is_chinese_slide:
        anchor = target[-4:] if len(target) >= 4 else target
        return target.count(anchor) > 1
    clean = PUNCT_RE.sub('', target.lower())
    words = clean.split()
    if len(words) > 1:
        return clean.count(words[-1]) > 1
    return False


def short_target(target, is_chinese_slide):
    """Tighten the comparison target to the final clinch when there's no
    repeated phrase to count."""
    if not target or target_has_repeat(target, is_chinese_slide):
        return target
    if is_chinese_slide:
        return target[-6:] if len(target) > 6 else target
    if len(target) <= 25:
        return target
    trimmed = target[-25:]
    space = trimmed.find(' ')
    return trimmed[space + 1:].lstrip() if space != -1 else trimmed


# ================= KEYBOARD LISTENER =================
def on_press(key):
    global cued_slide_index, is_paused, is_slow_mode, stop_at_section_end, held_at_index
    global auto_locate, shadow_mode, advance_guard
    try:
        if key == keyboard.Key.right:
            if cued_slide_index is not None and is_end_of_section():
                idx = cued_slide_index
                group = poller.slide_cache[idx]['group'] if poller and 0 <= idx < len(poller.slide_cache) else "?"
                mark_self_action(idx)
                trigger_slide(idx)
                cued_slide_index = None
                set_action(f"Manual → fired cued jump to {group} (slide {idx})")
            else:
                trigger_api("/v1/presentation/active/next/trigger", "Next slide")
            return
        elif key == keyboard.Key.left: trigger_api("/v1/presentation/active/previous/trigger", "Previous slide"); return
        elif key == keyboard.Key.down: trigger_api("/v1/playlist/active/next/trigger", "Next song"); return
        elif key == keyboard.Key.up: trigger_api("/v1/playlist/active/previous/trigger", "Previous song"); return

        if hasattr(key, 'char') and key.char:
            k = key.char.lower()

            if k == '/':
                is_paused = not is_paused
                set_action("Paused — not listening" if is_paused else "Resumed — listening")
                return

            if k == ',':
                is_slow_mode = False
                set_action("Fast song mode")
                return
            if k == '.':
                is_slow_mode = True
                set_action("Slow song mode")
                return

            if k == "'":
                cued_slide_index = None
                set_action("Cleared cued section jump")
                return

            if k == ';':
                stop_at_section_end = not stop_at_section_end
                if not stop_at_section_end:
                    held_at_index = None
                set_action("Hold at section end: ON" if stop_at_section_end
                           else "Hold at section end: OFF")
                return

            # ---- FULL-AUTO CONTROLS ----
            if k == 'a':
                auto_locate = not auto_locate
                localizer.reset()
                set_action("Full auto: ON — tracking the whole song" if auto_locate
                           else "Full auto: OFF — baseline behaviour (current slide only)")
                log_event('toggle', setting='auto_locate', value=auto_locate)
                return

            if k == 'j':
                shadow_mode = not shadow_mode
                set_action("Shadow mode: ON — logging decisions, not acting" if shadow_mode
                           else "Shadow mode: OFF — localizer is driving PP7")
                log_event('toggle', setting='shadow_mode', value=shadow_mode)
                return

            if k == 'k':
                advance_guard = not advance_guard
                set_action(f"Early-advance guard: {'ON' if advance_guard else 'OFF'}")
                log_event('toggle', setting='advance_guard', value=advance_guard)
                return

            if k == 'l':
                localizer.reset()
                localizer.cooldown_until = 0.0
                set_action("Re-locating — will act on the next confident read")
                log_event('relocate_requested')
                return

            if k in HOTKEYS:
                now = time.time()
                group_name = HOTKEYS[k]
                if k in last_key_press and (now - last_key_press[k]) < DOUBLE_PRESS_DELAY:
                    jump_to_group(group_name, immediate=True)
                    last_key_press[k] = 0
                else:
                    jump_to_group(group_name, immediate=False)
                    last_key_press[k] = now
    except Exception: pass


# ================= DASHBOARD =================
def render_dashboard():
    listening = not is_paused
    is_chinese = poller.is_chinese_slide if poller else False
    base_threshold = CN_THRESHOLD if is_chinese else EN_THRESHOLD
    threshold = base_threshold + 10 if is_slow_mode else base_threshold

    grid = Table.grid(padding=(0, 1))
    grid.add_column()

    # Row 1: status chips
    chips = Text()
    chips.append(" ▶ LISTENING " if listening else " ⏸ PAUSED ",
                 style="bold white on green" if listening else "bold white on red")
    chips.append("  ")
    if not auto_locate:
        chips.append(" MANUAL ", style="black on white")
    elif shadow_mode:
        chips.append(" SHADOW ", style="black on bright_black")
    elif localizer.is_lost:
        chips.append(" SEARCHING ", style="bold white on dark_orange")
    else:
        chips.append(" FULL AUTO ", style="bold white on blue")
    chips.append("  ")
    chips.append(" FAST " if not is_slow_mode else " SLOW ",
                 style="black on cyan" if not is_slow_mode else "black on yellow")
    chips.append("  ")
    chips.append(f" HOLD@END {'ON' if stop_at_section_end else 'OFF'} ",
                 style="black on magenta" if stop_at_section_end else "dim")
    chips.append("  ")
    connected = bool(poller and poller.connected)
    chips.append("PP7 ●", style="green" if connected else "red")
    grid.add_row(chips)

    # Row 2: mic + level meter + no-audio watchdog
    now = time.time()
    silent_for = (now - ui.last_sound_ts) if ui.last_sound_ts else 999
    level = min(1.0, ui.input_level / 0.15)
    bars = int(level * 12)
    mic = Text()
    mic.append(f"Mic: {ui.mic_name}  ")
    mic.append("▰" * bars + "▱" * (12 - bars), style="green" if bars > 0 else "dim")
    if listening and silent_for > 5:
        mic.append("  ⚠ NO AUDIO — check mic/source", style="bold white on red")
    grid.add_row(mic)

    # Row 3: current slide + the tail we're listening for
    slide = Text()
    idx = poller.current_index if poller else -1
    if poller and 0 <= idx < len(poller.slide_cache):
        group = poller.slide_cache[idx]['group']
        target, _ = poller.get_target()
        listening_for = short_target(target, is_chinese) if target else target
        slide.append(f"Slide {idx} ", style="bold")
        slide.append(f"[{group}]  ", style="cyan")
        slide.append(f"→ …{listening_for}")
    else:
        slide.append("No active slide", style="dim")
    grid.add_row(slide)

    # Row 4: heard + tail-match score
    lang = "CN/Yue" if is_chinese else "EN"
    heard = Text()
    heard.append(f"Heard[{lang}]: ")
    heard.append(f"{ui.heard or '—':<40} ")
    if ui.score is not None:
        heard.append(f"Match {ui.score}%/{threshold}",
                     style="bold green" if ui.score >= threshold else "yellow")
    grid.add_row(heard)

    # Row 5: where in the whole song the localizer thinks we are
    if auto_locate:
        loc = Text()
        loc.append("Song position: ", style="bold blue")
        if ui.candidates:
            for i, c in enumerate(ui.candidates):
                style = "bold green" if i == 0 else "dim"
                mark = "●" if c.index == idx else "○"
                loc.append(f"{mark}{c.index}[{c.group}] {c.score:.0f}  ", style=style)
        else:
            loc.append("— listening —", style="dim")
        grid.add_row(loc)

        detail = Text()
        if ui.progress is not None:
            filled = int(min(1.0, max(0.0, ui.progress)) * 10)
            detail.append("Through slide ")
            detail.append("█" * filled + "░" * (10 - filled), style="blue")
            detail.append(f" {ui.progress:.0%}   ")
        detail.append(ui.locate_summary or "", style="dim italic")
        grid.add_row(detail)

    # Row 6: cued + decode latency
    info = Text()
    cued_name = None
    if cued_slide_index is not None and poller and 0 <= cued_slide_index < len(poller.slide_cache):
        cued_name = poller.slide_cache[cued_slide_index]['group']
    info.append(f"Cued: {cued_name or '—'}", style="gold1" if cued_name else "dim")
    info.append("    ")
    info.append(f"decode ~{ui.decode_ms:.0f} ms", style="red" if ui.decode_ms >= 600 else "dim")
    info.append("    ")
    info.append(f"jumps {perf_metrics['auto_jumps']}  rescues {perf_metrics['auto_rescues']}"
                f"  advances {perf_metrics['tail_advances']}", style="dim")
    grid.add_row(info)

    grid.add_row(Text(f"Last: {ui.last_action}", style="dim italic"))
    grid.add_row(Text("[a] full-auto  [j] shadow  [k] guard  [l] re-locate   [/] pause  "
                      "[;] hold@end  ['] clear cued  [,/.] fast/slow  ←→ slide  ↑↓ song",
                      style="dim cyan"))

    title = "PP7 Auto-Lyrics — FULL AUTO"
    return Panel(grid, title=title, border_style="blue", padding=(0, 1))


# ================= AUDIO / DECODE HELPERS =================
def decode_buffer(engine, buffer):
    """Run one SenseVoice pass. Returns (text, elapsed_ms)."""
    stream = engine.create_stream()
    stream.accept_waveform(16000, buffer)
    t0 = time.perf_counter()
    engine.decode_stream(stream)
    elapsed = (time.perf_counter() - t0) * 1000
    return stream.result.text, elapsed


def plan_languages(current_lang, cycle, lost):
    """Which engines to run this cycle.

    A single-language deck only ever needs one engine, so full-auto costs nothing
    over the baseline. A bilingual deck normally decodes the current slide's
    language and periodically *probes* the other one — that probe is what catches
    the band moving from the English block to the Chinese block while PP7 is
    still sitting in the English section. When we're lost, probe every cycle."""
    if not auto_locate:
        return [current_lang]
    if len(localizer.languages) == 1:
        # Trust the deck over the current slide: a blank or untexted slide reports
        # 'en' by default, which would be the wrong engine for an all-Chinese song.
        return list(localizer.languages)
    if not localizer.languages:
        return [current_lang]
    other = 'en' if current_lang == 'cn' else 'cn'
    if lost or cycle % PROBE_EVERY == 0:
        return [current_lang, other]
    return [current_lang]


def execute_locate_action(decision, current_idx):
    """Carry out a localizer decision. Returns True if PP7 actually moved."""
    global cued_slide_index
    target = decision.target_index

    def _log(executed, action):
        log_event('locate_action', executed=executed, action=action,
                  target=target, from_index=current_idx, reason=decision.reason,
                  score=round(decision.winner.score, 1) if decision.winner else None,
                  margin=round(decision.margin, 1))

    if shadow_mode:
        perf_metrics['shadow_jumps'] += 1
        set_action(f"[shadow] would {decision.action} → slide {target} ({decision.reason})")
        _log(False, decision.action)
        # Serve the cooldown as if we had acted, so the log reads as one decision
        # per event rather than a repeat every cycle until the operator catches up.
        decision_cooldown()
        return False

    # The operator has asked to stop at section ends. Rather than overriding that,
    # full auto degrades to auto-*cueing*: it still tells them where the band went,
    # they still press → to accept it. That alone fixes the problem this script
    # exists for — the operator no longer has to spot the section by eye.
    if stop_at_section_end and is_end_of_section():
        if decision.action == "next":
            _log(False, 'suppressed_by_hold')
            decision_cooldown()
            return False
        if cued_slide_index != target:
            cued_slide_index = target
            group = poller.slide_cache[target]['group'] if poller and 0 <= target < len(poller.slide_cache) else "?"
            set_action(f"Held at section end — cued {group} (slide {target}); press → to go")
            _log(False, 'cue')
        decision_cooldown()
        return False

    if decision.action == "next":
        # Route through the normal advance path so a cued section jump still wins.
        handle_lyric_trigger(origin="rescue")
        set_action(f"Auto-caught missed advance → slide {target}")
    else:
        mark_self_action(target)
        trigger_slide(target)
        perf_metrics['auto_jumps'] += 1
        set_action(f"Auto-jumped to slide {target} ({decision.reason})")

    _log(True, decision.action)
    return True


def decision_cooldown():
    """Hold the localizer quiet for a beat after it reaches a conclusion, whether
    or not that conclusion moved PP7."""
    localizer.cooldown_until = time.time() + LOCATE_COOLDOWN
    localizer.votes.clear()


def drain_movements():
    """Fold PP7 slide changes into the localizer from the audio thread.

    The poller queues them rather than applying them so only one thread ever
    mutates the localizer."""
    if poller is None:
        return
    while True:
        try:
            index, ts = poller.movements.popleft()
        except IndexError:
            return
        source = movement_source(index, ts)
        lead = localizer.note_movement(index, now=ts, source=source)
        log_event('slide_moved', index=index, source=source,
                  lead_sec=round(lead, 3) if lead is not None else None)
        if lead is not None:
            set_action(f"Operator moved to slide {index} — script had it {lead:.1f}s earlier")


# ================= MAIN =================
def main(args):
    global poller, live, shadow_mode, auto_locate, _events_fh, held_at_index

    shadow_mode = args.shadow
    auto_locate = not args.no_auto

    _events_fh = open(events_filename, "a", encoding="utf-8")
    log_event('session_start', shadow=shadow_mode, auto_locate=auto_locate)

    print("🧠 Booting SenseVoice Engine (Strict English Mode)...")
    rec_sv_en = load_sensevoice_engine("en")
    print("🧠 Booting SenseVoice Engine (Strict Cantonese Mode)...")
    rec_sv_cn = load_sensevoice_engine("yue")
    engines = {'en': rec_sv_en, 'cn': rec_sv_cn}

    p = pyaudio.PyAudio()

    if args.mic is None:
        print("\n🎤 Available Microphones:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0: print(f"   [{i}] {info.get('name')}")
        try: mic_idx = int(input("\n👉 Mic Index (Press Enter for default): "))
        except: mic_idx = None
    else:
        mic_idx = args.mic

    try:
        if mic_idx is None:
            ui.mic_name = p.get_default_input_device_info().get('name', 'System default')
        else:
            ui.mic_name = p.get_device_info_by_index(mic_idx).get('name', f'Mic {mic_idx}')
    except Exception:
        ui.mic_name = 'System default' if mic_idx is None else f'Mic {mic_idx}'

    poller = PP7SmartPoller()
    threading.Thread(target=poller.update_loop, daemon=True).start()

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                    input_device_index=mic_idx, frames_per_buffer=2048)
    print(f"\n✅ Listening on: {ui.mic_name}")
    print(f"   Full auto: {'ON' if auto_locate else 'OFF'}"
          f"{'  (SHADOW — logging only, not driving PP7)' if shadow_mode else ''}\n")

    current_slide = -1
    known_cache_version = -1
    audio_buffer = np.array([], dtype=np.float32)
    PROCESS_INTERVAL = int(16000 * 0.4)
    MAX_BUFFER_SIZE = int(16000 * 8.0)
    SHORT_BUFFER_SIZE = int(16000 * 4.0)
    samples_since_last_process = 0
    cycle = 0
    ui.last_sound_ts = time.time()

    disable_terminal_echo()
    with Live(render_dashboard(), console=Console(), refresh_per_second=8, screen=False) as live:
        while True:
            try:
                # Rebuild the search index whenever ProPresenter changes song.
                # cache_version only increments on a *successful* fetch, so this
                # never fires against the empty cache we start with.
                if poller.cache_version and poller.cache_version != known_cache_version:
                    with poller.lock:
                        cache_copy = list(poller.slide_cache)
                        known_cache_version = poller.cache_version
                    localizer.build(cache_copy)
                    log_event('song_loaded', slides=len(cache_copy),
                              languages=sorted(localizer.languages),
                              duplicate_groups=len(localizer.dup_groups))
                    set_action(f"Loaded {len(cache_copy)} slides "
                               f"({'/'.join(sorted(localizer.languages)) or 'empty'})")

                drain_movements()

                target, idx = poller.get_target()
                is_chinese = poller.is_chinese_slide

                # Without a slide cache there's nothing to match against at all.
                if idx == -1 or not localizer.entries:
                    live.update(render_dashboard())
                    time.sleep(0.1)
                    continue

                if idx != current_slide:
                    current_slide = idx
                    held_at_index = None
                    audio_buffer = np.array([], dtype=np.float32)
                    samples_since_last_process = 0
                    stream.read(stream.get_read_available(), exception_on_overflow=False)
                    live.update(render_dashboard())
                    continue

                # Drain the WHOLE input backlog, not just one fixed chunk, so a slow
                # decode can't let latency grow without bound.
                avail = stream.get_read_available()
                nframes = avail if avail > 2048 else 2048
                data = stream.read(nframes, exception_on_overflow=False)

                if is_paused:
                    audio_buffer = np.array([], dtype=np.float32)
                    samples_since_last_process = 0
                    live.update(render_dashboard())
                    continue

                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                if len(samples) > 0:
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    ui.input_level = rms
                    if rms > 0.003:
                        ui.last_sound_ts = time.time()

                audio_buffer = np.concatenate((audio_buffer, samples))
                samples_since_last_process += len(samples)

                if len(audio_buffer) > MAX_BUFFER_SIZE:
                    audio_buffer = audio_buffer[-MAX_BUFFER_SIZE:]

                if samples_since_last_process < PROCESS_INTERVAL:
                    live.update(render_dashboard())
                    continue

                # ---------------- decode ----------------
                cycle += 1
                perf_metrics['decode_cycles'] += 1
                samples_since_last_process = 0

                current_lang = poller.slide_language(idx)
                has_repeat = target_has_repeat(target, is_chinese) if target else False
                lost = auto_locate and localizer.is_lost
                # A repeated ending needs the full buffer to count loops; so does a
                # lost localizer, which needs every second of context it can get.
                decode_max = MAX_BUFFER_SIZE if (has_repeat or lost) else SHORT_BUFFER_SIZE
                window = audio_buffer[-decode_max:] if len(audio_buffer) > decode_max else audio_buffer

                texts, total_ms = {}, 0.0
                for lang in plan_languages(current_lang, cycle, lost):
                    text, ms = decode_buffer(engines[lang], window)
                    texts[lang] = text
                    total_ms += ms
                ui.decode_ms = total_ms
                perf_metrics['audio_latency_sum'] += total_ms
                perf_metrics['audio_latency_count'] += 1

                # ---------------- localize ----------------
                decision = None
                if auto_locate:
                    t0 = time.perf_counter()
                    decision = localizer.observe(
                        normalize_heard(texts.get('cn', ''), True),
                        normalize_heard(texts.get('en', ''), False),
                        idx,
                    )
                    perf_metrics['fuzzy_latency_sum'] += (time.perf_counter() - t0) * 1000
                    perf_metrics['fuzzy_latency_count'] += 1

                    ui.candidates = decision.candidates
                    ui.progress = decision.winner.progress if decision.winner else None
                    ui.locate_summary = decision.reason
                    if not decision.confident:
                        perf_metrics['lost_cycles'] += 1

                    log_event('locate', slide=idx, confident=decision.confident,
                              target=decision.target_index,
                              margin=round(decision.margin, 1),
                              action=decision.action, reason=decision.reason,
                              heard={k: v for k, v in texts.items() if v},
                              top=[{'i': c.index, 'g': c.group, 's': round(c.score, 1),
                                    'p': round(c.progress, 2)} for c in decision.candidates])

                # ---------------- tail-match advance ----------------
                fired = False
                result = texts.get(current_lang, "")
                if target and result:
                    score_target = target if has_repeat else short_target(target, is_chinese)
                    score, clean_heard = fast_smart_score(result, score_target, is_chinese)
                    ui.heard = clean_heard[-40:] if len(clean_heard) > 40 else clean_heard
                    ui.score = score

                    base_threshold = CN_THRESHOLD if is_chinese else EN_THRESHOLD
                    threshold = base_threshold + 10 if is_slow_mode else base_threshold

                    if score >= threshold:
                        # The localizer knows how far into the line the singer
                        # actually is. If it is confident we're barely into this
                        # slide, a tail match is a mishearing, not an ending.
                        too_early = (advance_guard and decision is not None
                                     and decision.confident
                                     and decision.target_index == idx
                                     and decision.winner is not None
                                     and decision.winner.progress < ADVANCE_MIN_PROGRESS)
                        if too_early:
                            perf_metrics['vetoed_advances'] += 1
                            set_action(f"Held back early match — only "
                                       f"{decision.winner.progress:.0%} through the line")
                            log_event('advance_vetoed', slide=idx, score=score,
                                      progress=round(decision.winner.progress, 2))
                        else:
                            missing_chars = max(0, len(score_target) - len(clean_heard))
                            if missing_chars > 0:
                                if is_chinese:
                                    delay = min(missing_chars * 0.5, 3.5) if is_slow_mode else min(missing_chars * 0.3, 2.5)
                                    unit = "chars"
                                else:
                                    delay = min(missing_chars * 0.25, 4.0) if is_slow_mode else min(missing_chars * 0.15, 2.5)
                                    unit = "letters"
                                set_action(f"Early match ({missing_chars} {unit} left) — delaying {delay:.1f}s")
                                live.update(render_dashboard())
                                time.sleep(delay)

                            handle_lyric_trigger()
                            fired = True

                # ---------------- act on the localizer ----------------
                if not fired and decision is not None and decision.action != "none":
                    fired = execute_locate_action(decision, idx)

                if fired:
                    live.update(render_dashboard())
                    # Let the singer finish the line, then wipe buffers to destroy
                    # ghost audio before the next slide is scored.
                    time.sleep(0.8)
                    audio_buffer = np.array([], dtype=np.float32)
                    samples_since_last_process = 0
                    stream.read(stream.get_read_available(), exception_on_overflow=False)
                    localizer.cooldown_until = time.time() + LOCATE_COOLDOWN
                    localizer.votes.clear()

                live.update(render_dashboard())

            except KeyboardInterrupt: break
            except Exception: pass
    restore_terminal_echo()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ProPresenter 7 auto-lyrics with full-auto song-wide position tracking.")
    parser.add_argument("--shadow", action="store_true",
                        help="Log every localizer decision without touching PP7. "
                             "Run this alongside a manually-operated service to measure "
                             "how far ahead of the operator the script would have been.")
    parser.add_argument("--no-auto", action="store_true",
                        help="Start with the song-wide localizer off (baseline behaviour). "
                             "Toggle back on at any time with the 'a' key.")
    parser.add_argument("--mic", type=int, default=None,
                        help="Microphone index; skips the interactive prompt.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
