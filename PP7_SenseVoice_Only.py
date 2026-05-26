import requests
import sherpa_onnx
from thefuzz import fuzz
import pyaudio
import numpy as np
import threading
import time
import re
import sys
import os
import signal
import termios
import opencc
import logging
from dataclasses import dataclass
from datetime import datetime
from pynput import keyboard
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ================= PERFORMANCE LOGGING SETUP =================
log_filename = f"performance_{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Global metrics for exit summary
perf_metrics = {
    'audio_latency_sum': 0.0,
    'audio_latency_count': 0,
    'fuzzy_latency_sum': 0.0,
    'fuzzy_latency_count': 0,
    'api_latency_sum': 0.0,
    'api_latency_count': 0
}
# =============================================================

# ================= CONFIGURATION =================
PP_HOST = "127.0.0.1"
PP_PORT = 1025          
EN_THRESHOLD = 65  # 65 when plugged in 
CN_THRESHOLD = 55 # 55 when plugged in 
MODEL_DIR_SV = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"

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

def force_quit(sig, frame):
    try:
        if live is not None:
            live.stop()
    except Exception:
        pass
    restore_terminal_echo()
    print("\n\n🛑 Force terminating the script...")
    print("\n📊 --- Performance Session Summary ---")
    
    if perf_metrics['audio_latency_count'] > 0:
        avg_audio = perf_metrics['audio_latency_sum'] / perf_metrics['audio_latency_count']
        print(f"🎙️  Avg Audio-to-Text Latency: {avg_audio:.2f} ms")
    else:
        print("🎙️  Avg Audio-to-Text Latency: N/A")
        
    if perf_metrics['fuzzy_latency_count'] > 0:
        avg_fuzzy = perf_metrics['fuzzy_latency_sum'] / perf_metrics['fuzzy_latency_count']
        print(f"🧠 Avg Fuzzy Match Latency: {avg_fuzzy:.2f} ms")
    else:
        print("🧠 Avg Fuzzy Match Latency: N/A")
        
    if perf_metrics['api_latency_count'] > 0:
        avg_api = perf_metrics['api_latency_sum'] / perf_metrics['api_latency_count']
        print(f"🌐 Avg API Trigger Latency: {avg_api:.2f} ms")
    else:
        print("🌐 Avg API Trigger Latency: N/A")
        
    print("--------------------------------------\n")
    os._exit(0)
signal.signal(signal.SIGINT, force_quit)

poller = None
cued_slide_index = None
last_key_press = {}
DOUBLE_PRESS_DELAY = 0.4

# NEW: Global Pause State
is_paused = False
is_slow_mode = False
stop_at_section_end = False   # when True, auto-advance holds at section boundaries
held_at_index = None          # slide we're currently holding on (suppresses repeat fires/messages)


@dataclass
class UIState:
    """Shared snapshot the rich dashboard renders. Background threads (keyboard
    listener, poller) mutate fields here instead of printing — only the main loop
    draws, so a fixed panel never gets corrupted by interleaved thread output."""
    mic_name: str = "—"
    input_level: float = 0.0          # latest chunk RMS (0..~0.3)
    last_sound_ts: float = 0.0        # last time RMS crossed the audible floor
    decode_ms: float = 0.0            # latest SenseVoice decode time
    heard: str = ""                   # latest transcription tail
    score: int | None = None          # latest match score
    last_action: str = "Ready"        # most recent event (jump / trigger / toggle)


ui = UIState()
live = None  # the rich.Live handle, set in main(); force_quit stops it on SIGINT
_saved_termios = None  # terminal attrs to restore on exit (echo on)


def set_action(msg):
    ui.last_action = msg


def disable_terminal_echo():
    """Suppress local terminal echo while the rich dashboard is active.
    pynput's section-jump letters (and `' ; / , .`) would otherwise be echoed
    to stdout when the terminal has focus, corrupting Live's cursor tracking
    and causing the panel to redraw on a new line each keypress."""
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
        provider="cpu"
    )

print("🧠 Booting SenseVoice Engine (Strict English Mode)...")
rec_sv_en = load_sensevoice_engine("en")

print("🧠 Booting SenseVoice Engine (Strict Cantonese Mode)...")
rec_sv_cn = load_sensevoice_engine("yue")

class PP7SmartPoller:
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
            chinese_chars = "".join(re.findall(r'[\u4e00-\u9FFF]', self.current_full_text))
            
            if len(chinese_chars) > 0:
                self.is_chinese_slide = True
                target = chinese_chars[-10:] if len(chinese_chars) > 10 else chinese_chars
                return target, self.current_index
            else:
                self.is_chinese_slide = False
                text = self.current_full_text
                target = text[-35:] if len(text) > 35 else text
                return target.strip(), self.current_index

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
                            last_index = idx
            except Exception:
                self.connected = False
            time.sleep(0.15)

# --- API TRIGGER FUNCTIONS ---
def trigger_api(endpoint, message):
    try:
        requests.get(f"http://{PP_HOST}:{PP_PORT}{endpoint}")
        set_action(message)
    except: pass

def trigger_slide(index):
    try:
        requests.get(f"http://{PP_HOST}:{PP_PORT}/v1/presentation/active/{index}/trigger")
        set_action(f"Jumped to slide {index}")
    except: pass

def is_end_of_section():
    """True if the poller's current slide is the last one in its group (or the
    final slide of the deck). Shared by the auto-trigger and the right-arrow
    manual override so a cued section jump fires consistently at boundaries."""
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


def handle_lyric_trigger():
    global cued_slide_index, poller, held_at_index
    end_of_section = is_end_of_section()
    with poller.lock:
        current_idx = poller.current_index

    # Persistent hold-at-section-end. Overrides cued jumps. The held_at_index guard
    # prevents the message/return from spamming every decode cycle while we sit on
    # the last slide of the section.
    if stop_at_section_end and end_of_section:
        if held_at_index != current_idx:
            held_at_index = current_idx
            set_action(f"Holding at end of section (slide {current_idx}) — press → to continue")
        return

    if cued_slide_index is not None and end_of_section:
        trigger_slide(cued_slide_index)
        set_action(f"Section ended — executed cued jump to slide {cued_slide_index}")
        cued_slide_index = None
    else:
        try:
            requests.get(f"http://{PP_HOST}:{PP_PORT}/v1/presentation/active/next/trigger")
            set_action("Triggered next slide")
        except: pass

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
            trigger_slide(target_idx)
            set_action(f"Jumped now to {group_name} (slide {target_idx})")
            cued_slide_index = None
        else:
            cued_slide_index = target_idx
            set_action(f"Cued {group_name} (slide {target_idx}) — jumps after this section")

# --- KEYBOARD LISTENER ---
def on_press(key):
    global cued_slide_index, is_paused, is_slow_mode, stop_at_section_end, held_at_index
    try:
        if key == keyboard.Key.right:
            # Manual override at section boundaries: if we're on the last slide of
            # the current section AND a cue is set, → fires the cued jump (same
            # behavior as automation, just operator-initiated). Mid-section, → is
            # always a normal next-slide so the cue stays parked.
            if cued_slide_index is not None and is_end_of_section():
                idx = cued_slide_index
                group = poller.slide_cache[idx]['group'] if poller and 0 <= idx < len(poller.slide_cache) else "?"
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

            # --- START / PAUSE TOGGLE ---
            if k == '/':
                is_paused = not is_paused
                set_action("Paused — not listening" if is_paused else "Resumed — listening")
                return

            # --- FAST / SLOW SONG TOGGLE ---
            if k == ',':
                is_slow_mode = False
                set_action("Fast song mode")
                return
            if k == '.':
                is_slow_mode = True
                set_action("Slow song mode")
                return

            # --- CLEAR CUED JUMP ---
            if k == "'":
                cued_slide_index = None
                set_action("Cleared cued section jump")
                return

            # --- TOGGLE HOLD AT SECTION END ---
            if k == ';':
                stop_at_section_end = not stop_at_section_end
                if not stop_at_section_end:
                    held_at_index = None
                set_action("Hold at section end: ON" if stop_at_section_end
                           else "Hold at section end: OFF")
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

def fast_smart_score(heard_text, target, is_chinese_slide):
    if not heard_text or not target: return 0, heard_text
    heard_clean = re.sub(r'<\|.*?\|>', '', heard_text).strip()
    
    if is_chinese_slide:
        heard_cn = "".join(re.findall(r'[\u4e00-\u9FFF]', heard_clean))
        if len(heard_cn) < 2: return 0, heard_cn 
        heard_trad = cc.convert(heard_cn)
        
        score = fuzz.partial_ratio(target, heard_trad)
        
        # --- THE ADVANCED REPETITION ENFORCER (CHINESE) ---
        # Grab the last 4 characters to act as our "anchor" phrase
        anchor_phrase = target[-4:] if len(target) >= 4 else target
        target_reps = target.count(anchor_phrase)
        heard_reps = heard_trad.count(anchor_phrase)
        
        # If the end of the slide is a repeated phrase, enforce the loop!
        if target_reps > 1 and heard_reps < target_reps:
            score -= 40  # Massive penalty: Singer is only on the first loop
            
        # Dynamic penalty: -6 points per missing Chinese character
        missing_chars = len(target) - len(heard_trad)
        if missing_chars > 0:
            score -= (missing_chars * 6)
            
        return score, heard_trad
        
    else:
        if len(heard_clean) < 5: return 0, heard_clean 
        
        clean_target = re.sub(r'[^\w\s]', '', target.lower())
        clean_heard = re.sub(r'[^\w\s]', '', heard_clean.lower())
        
        score = fuzz.partial_ratio(clean_target, clean_heard)
        
        # --- THE ADVANCED REPETITION ENFORCER (ENGLISH) ---
        target_words = clean_target.split()
        if len(target_words) > 1:
            anchor_word = target_words[-1]
            target_reps = clean_target.count(anchor_word)
            heard_reps = clean_heard.count(anchor_word)
            
            if target_reps > 1 and heard_reps < target_reps:
                score -= 40  # Massive penalty
                
        # Dynamic penalty: -3 points per missing English letter
        missing_letters = len(clean_target) - len(clean_heard)
        if missing_letters > 0:
            score -= (missing_letters * 3)

        return score, heard_clean


def target_has_repeat(target, is_chinese_slide):
    """Does the slide's target end in a repeated phrase? If so we must decode the
    full buffer so the repetition enforcer can count loops; otherwise the shorter,
    faster decode window is safe."""
    if not target:
        return False
    if is_chinese_slide:
        anchor = target[-4:] if len(target) >= 4 else target
        return target.count(anchor) > 1
    clean = re.sub(r'[^\w\s]', '', target.lower())
    words = clean.split()
    if len(words) > 1:
        return clean.count(words[-1]) > 1
    return False


def short_target(target, is_chinese_slide):
    """Tighten the comparison target to the final clinch when there's no
    repeated phrase to count. A shorter target gives partial_ratio a sharper
    endpoint match (fewer non-matching middle chars dilute the score), which
    lifts trigger accuracy on plain non-repeating lines. Repeated endings keep
    the full target so the repetition enforcer can still count loops."""
    if not target or target_has_repeat(target, is_chinese_slide):
        return target
    if is_chinese_slide:
        # poller already trimmed to last 10 CN chars; the final 6 carry the clinch.
        return target[-6:] if len(target) > 6 else target
    # poller already trimmed to last 35 EN chars; tighten to ~25, snapped to a
    # word boundary so we don't end up matching a partial-word fragment.
    if len(target) <= 25:
        return target
    trimmed = target[-25:]
    space = trimmed.find(' ')
    return trimmed[space + 1:].lstrip() if space != -1 else trimmed


def render_dashboard():
    """Build the live rich panel from current globals + ui + poller state."""
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
    mic.append("▰" * bars + "▱" * (12 - bars),
               style="green" if bars > 0 else "dim")
    if listening and silent_for > 5:
        mic.append("  ⚠ NO AUDIO — check mic/source", style="bold white on red")
    grid.add_row(mic)

    # Row 3: current slide + target (show the tightened target — what we actually match)
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

    # Row 4: heard + match score
    lang = "CN/Yue" if is_chinese else "EN"
    heard = Text()
    heard.append(f"Heard[{lang}]: ")
    heard.append(f"{ui.heard or '—':<40} ")
    if ui.score is not None:
        heard.append(f"Match {ui.score}%/{threshold}",
                     style="bold green" if ui.score >= threshold else "yellow")
    grid.add_row(heard)

    # Row 5: cued + decode latency
    info = Text()
    cued_name = None
    if cued_slide_index is not None and poller and 0 <= cued_slide_index < len(poller.slide_cache):
        cued_name = poller.slide_cache[cued_slide_index]['group']
    info.append(f"Cued: {cued_name or '—'}", style="gold1" if cued_name else "dim")
    info.append("    ")
    info.append(f"decode ~{ui.decode_ms:.0f} ms",
                style="red" if ui.decode_ms >= 600 else "dim")
    grid.add_row(info)

    # Row 6: last action
    grid.add_row(Text(f"Last: {ui.last_action}", style="dim italic"))

    # Footer: key legend
    grid.add_row(Text("[/] pause   [;] hold@end   ['] clear cued   [,/.] fast/slow   ←→ slide   ↑↓ song",
                      style="dim cyan"))

    return Panel(grid, title="PP7 Auto-Lyrics", border_style="blue", padding=(0, 1))


def main():
    global poller, is_paused, is_slow_mode, live
    p = pyaudio.PyAudio()

    print("\n🎤 Available Microphones:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0: print(f"   [{i}] {info.get('name')}")

    try: mic_idx = int(input("\n👉 Mic Index (Press Enter for default): "))
    except: mic_idx = None

    # Resolve the device we actually bound to, so the operator can see at a glance
    # whether macOS quietly rerouted them to the built-in mic instead of the mixer.
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

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, input_device_index=mic_idx, frames_per_buffer=2048)
    print(f"\n✅ Listening on: {ui.mic_name}\n")

    current_slide = -1
    audio_buffer = np.array([], dtype=np.float32)
    PROCESS_INTERVAL = int(16000 * 0.4)
    MAX_BUFFER_SIZE = int(16000 * 8.0)
    SHORT_BUFFER_SIZE = int(16000 * 4.0)   # faster decode window for non-repeating lines
    samples_since_last_process = 0
    ui.last_sound_ts = time.time()

    disable_terminal_echo()
    with Live(render_dashboard(), console=Console(), refresh_per_second=8, screen=False) as live:
        while True:
            try:
                target, idx = poller.get_target()
                is_chinese = poller.is_chinese_slide

                if idx == -1 or not target:
                    live.update(render_dashboard())
                    time.sleep(0.1)
                    continue

                if idx != current_slide:
                    current_slide = idx
                    audio_buffer = np.array([], dtype=np.float32)
                    stream.read(stream.get_read_available(), exception_on_overflow=False)
                    live.update(render_dashboard())
                    continue

                # Drain the WHOLE input backlog, not just one fixed chunk. If a decode
                # ran long and audio piled up in PortAudio, reading a fixed 2048 would
                # never catch up and latency would grow without bound.
                avail = stream.get_read_available()
                nframes = avail if avail > 2048 else 2048
                data = stream.read(nframes, exception_on_overflow=False)

                # If the user paused the script, throw away the audio and skip processing
                if is_paused:
                    audio_buffer = np.array([], dtype=np.float32)
                    samples_since_last_process = 0
                    live.update(render_dashboard())
                    continue

                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                # Input-level watchdog: track RMS so the dashboard can flag a dead mic
                # (e.g. a silent fallback to an unplugged interface).
                if len(samples) > 0:
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    ui.input_level = rms
                    if rms > 0.003:
                        ui.last_sound_ts = time.time()

                audio_buffer = np.concatenate((audio_buffer, samples))
                samples_since_last_process += len(samples)

                if len(audio_buffer) > MAX_BUFFER_SIZE:
                    audio_buffer = audio_buffer[-MAX_BUFFER_SIZE:]

                if samples_since_last_process >= PROCESS_INTERVAL:
                    active_engine = rec_sv_cn if is_chinese else rec_sv_en

                    # Dynamic decode window: a normal line only needs the last few
                    # seconds (decodes ~2x faster); repeated endings need the full
                    # buffer so the repetition enforcer can count loops.
                    has_repeat = target_has_repeat(target, is_chinese)
                    decode_max = MAX_BUFFER_SIZE if has_repeat else SHORT_BUFFER_SIZE
                    decode_buffer = audio_buffer[-decode_max:] if len(audio_buffer) > decode_max else audio_buffer

                    s_stream = active_engine.create_stream()
                    s_stream.accept_waveform(16000, decode_buffer)
                    t0 = time.perf_counter()
                    active_engine.decode_stream(s_stream)
                    ui.decode_ms = (time.perf_counter() - t0) * 1000
                    perf_metrics['audio_latency_sum'] += ui.decode_ms
                    perf_metrics['audio_latency_count'] += 1
                    result = s_stream.result.text

                    if result:
                        # Tighten the comparison target on non-repeating lines so
                        # partial_ratio scores the actual endpoint, not a long tail
                        # diluted by mid-line content.
                        score_target = target if has_repeat else short_target(target, is_chinese)
                        score, clean_heard = fast_smart_score(result, score_target, is_chinese)
                        ui.heard = clean_heard[-40:] if len(clean_heard) > 40 else clean_heard
                        ui.score = score

                        # Adjust thresholds for slow mode
                        base_threshold = CN_THRESHOLD if is_chinese else EN_THRESHOLD
                        threshold = base_threshold + 10 if is_slow_mode else base_threshold

                        if score >= threshold:
                            # --- DYNAMIC PRE-TRIGGER DELAY ---
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
                            live.update(render_dashboard())

                            # Let the singer finish the line, then wipe buffers to
                            # destroy ghost audio before the next slide is scored.
                            time.sleep(0.8)
                            audio_buffer = np.array([], dtype=np.float32)
                            samples_since_last_process = 0
                            stream.read(stream.get_read_available(), exception_on_overflow=False)

                    samples_since_last_process = 0

                live.update(render_dashboard())

            except KeyboardInterrupt: break
            except Exception: pass
    restore_terminal_echo()


if __name__ == "__main__":
    main()