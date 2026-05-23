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
import opencc
import logging
from datetime import datetime
from pynput import keyboard

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
                                target, _ = self.get_target()
                                print(f"\n\n🎯 Slide {idx} [{self.slide_cache[idx]['group']}] Target: \"...{target}\"")
                            else:
                                self.current_full_text = ""
                            last_index = idx
            except: pass
            time.sleep(0.15)

# --- API TRIGGER FUNCTIONS ---
def trigger_api(endpoint, message):
    try:
        requests.get(f"http://{PP_HOST}:{PP_PORT}{endpoint}")
        print(f"\n{message}\n")
    except: pass

def trigger_slide(index):
    try:
        requests.get(f"http://{PP_HOST}:{PP_PORT}/v1/presentation/active/{index}/trigger")
        print(f"\n🚀 === JUMPED TO SLIDE {index} ===\n")
    except: pass

def handle_lyric_trigger():
    global cued_slide_index, poller, held_at_index
    is_end_of_section = False
    with poller.lock:
        current_idx = poller.current_index
        if 0 <= current_idx < len(poller.slide_cache):
            current_group = poller.slide_cache[current_idx]['group']
            next_idx = current_idx + 1
            if next_idx < len(poller.slide_cache):
                if current_group != poller.slide_cache[next_idx]['group']:
                    is_end_of_section = True
            else:
                is_end_of_section = True

    # Persistent hold-at-section-end. Overrides cued jumps. The held_at_index guard
    # prevents the message/return from spamming every decode cycle while we sit on
    # the last slide of the section.
    if stop_at_section_end and is_end_of_section:
        if held_at_index != current_idx:
            held_at_index = current_idx
            print(f"\n🛑 Holding at end of section (slide {current_idx}). Press → to continue.")
        return

    if cued_slide_index is not None and is_end_of_section:
        print(f"\n🚀 Section Ended! Executing Cued Jump to Slide {cued_slide_index}...")
        trigger_slide(cued_slide_index)
        cued_slide_index = None 
    else:
        try:
            requests.get(f"http://{PP_HOST}:{PP_PORT}/v1/presentation/active/next/trigger")
            print("\n🚀 === TRIGGERED NEXT ===\n")
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
            print(f"\n⚡ Double Press! Jumping instantly to: {group_name} (Slide {target_idx})")
            trigger_slide(target_idx)
            cued_slide_index = None
        else:
            print(f"\n📌 Cued next section: {group_name} (Slide {target_idx}). Will jump after current section ends.")
            cued_slide_index = target_idx

# --- KEYBOARD LISTENER ---
def on_press(key):
    global cued_slide_index, is_paused, is_slow_mode, stop_at_section_end, held_at_index
    try:
        if key == keyboard.Key.right: trigger_api("/v1/presentation/active/next/trigger", "➡️  === NEXT SLIDE ==="); return
        elif key == keyboard.Key.left: trigger_api("/v1/presentation/active/previous/trigger", "⬅️  === PREV SLIDE ==="); return
        elif key == keyboard.Key.down: trigger_api("/v1/playlist/active/next/trigger", "⬇️  === NEXT SONG ==="); return
        elif key == keyboard.Key.up: trigger_api("/v1/playlist/active/previous/trigger", "⬆️  === PREV SONG ==="); return

        if hasattr(key, 'char') and key.char:
            k = key.char.lower()
            
            # --- START / PAUSE TOGGLE ---
            if k == '/':
                is_paused = not is_paused
                state_msg = "⏸️  PAUSED" if is_paused else "▶️  RESUMED"
                print(f"\n\n{state_msg} - Auto-lyrics are {'stopped' if is_paused else 'listening'}.\n")
                return
            
            # --- FAST / SLOW SONG TOGGLE ---
            if k == ',':
                is_slow_mode = False
                print(f"\n\n⏩ FAST SONG MODE - Using default thresholds and delays.\n")
                return
            if k == '.':
                is_slow_mode = True
                print(f"\n\n🐢 SLOW SONG MODE - Increased thresholds and longer delays.\n")
                return
            
            # --- CLEAR CUED JUMP ---
            if k == "'":
                cued_slide_index = None
                print("\n\n🧹 Cleared cued section jump.\n")
                return

            # --- TOGGLE HOLD AT SECTION END ---
            if k == ';':
                stop_at_section_end = not stop_at_section_end
                if not stop_at_section_end:
                    held_at_index = None
                msg = ("🛑 HOLD AT SECTION END: ON — auto-advance pauses at each section boundary."
                       if stop_at_section_end else
                       "▶️  HOLD AT SECTION END: OFF — auto-advance crosses sections normally.")
                print(f"\n\n{msg}\n")
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
def main():
    global poller, is_paused, is_slow_mode
    p = pyaudio.PyAudio()
    
    print("\n🎤 Available Microphones:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0: print(f"   [{i}] {info.get('name')}")
            
    try: mic_idx = int(input("\n👉 Mic Index (Press Enter for default): "))
    except: mic_idx = None

    poller = PP7SmartPoller()
    threading.Thread(target=poller.update_loop, daemon=True).start()

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, input_device_index=mic_idx, frames_per_buffer=2048)
    print("\n✅ System Ready! '/' pause · ';' hold@section-end · ''' clear cued jump")
    
    current_slide = -1
    
    audio_buffer = np.array([], dtype=np.float32)
    PROCESS_INTERVAL = int(16000 * 0.4) 
    MAX_BUFFER_SIZE = int(16000 * 8.0)  
    samples_since_last_process = 0

    while True:
        try:
            target, idx = poller.get_target()
            is_chinese = poller.is_chinese_slide

            if idx == -1 or not target: 
                time.sleep(0.1)
                continue

            if idx != current_slide:
                current_slide = idx
                audio_buffer = np.array([], dtype=np.float32) 
                stream.read(stream.get_read_available(), exception_on_overflow=False)
                continue

            # We must ALWAYS read from the stream to prevent PyAudio from overflowing
            data = stream.read(2048, exception_on_overflow=False)
            
            # If the user paused the script, throw away the audio and skip processing
            if is_paused:
                audio_buffer = np.array([], dtype=np.float32)
                samples_since_last_process = 0
                continue

            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            audio_buffer = np.concatenate((audio_buffer, samples))
            samples_since_last_process += len(samples)

            if len(audio_buffer) > MAX_BUFFER_SIZE:
                audio_buffer = audio_buffer[-MAX_BUFFER_SIZE:]

            if samples_since_last_process >= PROCESS_INTERVAL:
                active_engine = rec_sv_cn if is_chinese else rec_sv_en
                
                s_stream = active_engine.create_stream()
                s_stream.accept_waveform(16000, audio_buffer)
                active_engine.decode_stream(s_stream)
                result = s_stream.result.text
                
                if result:
                    score, clean_heard = fast_smart_score(result, target, is_chinese)
                    lang_tag = "CN/Yue" if is_chinese else "EN"
                    
                    display_text = clean_heard[-40:] if len(clean_heard) > 40 else clean_heard
                    sys.stdout.write(f"\r👂 Heard [{lang_tag}]: {display_text:<40} | Match: {score}%   ")
                    sys.stdout.flush()

                    # Adjust thresholds for slow mode
                    base_threshold = CN_THRESHOLD if is_chinese else EN_THRESHOLD
                    threshold = base_threshold + 10 if is_slow_mode else base_threshold
                    
                    if score >= threshold:
                        # --- DYNAMIC DELAY LOGIC ---
                        missing_chars = max(0, len(target) - len(clean_heard))
                        if missing_chars > 0:
                            if is_chinese:
                                if is_slow_mode:
                                    delay = min(missing_chars * 0.5, 3.5)
                                else:
                                    # HEAVY PENALTY: 0.5s per missing Chinese character (up from 0.35s)
                                    # Cap increased to 2.5 seconds
                                    delay = min(missing_chars * 0.3, 2.5) 
                                unit = "chars"
                            else:
                                if is_slow_mode:
                                    delay = min(missing_chars * 0.25, 4.0)
                                else:
                                    # HEAVY PENALTY: 0.15s per missing English letter (approx 0.7s per word)
                                    # Cap increased to 2.5 seconds
                                    delay = min(missing_chars * 0.15, 2.5) 
                                unit = "letters"
                                
                            sys.stdout.write(f"\r⏱️ Early match ({missing_chars} {unit} left)! Delaying {delay:.1f}s...      ")
                            sys.stdout.flush()
                            time.sleep(delay)

                        # Trigger the slide
                        handle_lyric_trigger()
                        
                        # 1. Sleep to let the singer finish the line and take a breath
                        time.sleep(0.8) 
                        
                        # 2. CLEAR BUFFERS AFTER SLEEPING (Destroys ghost audio)
                        audio_buffer = np.array([], dtype=np.float32) 
                        samples_since_last_process = 0
                        stream.read(stream.get_read_available(), exception_on_overflow=False)

                samples_since_last_process = 0 

        except KeyboardInterrupt: break
        except Exception: pass

if __name__ == "__main__":
    main()