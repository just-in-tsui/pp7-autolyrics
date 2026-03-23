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
import glob
import signal
import opencc
from pynput import keyboard

# ================= CONFIGURATION =================
PP_HOST = "127.0.0.1"
PP_PORT = 1026          
EN_THRESHOLD = 75     
CN_THRESHOLD = 45 
MODEL_DIR_EN = "model-en" 
MODEL_DIR_SV = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"

# ---------------- HOTKEY CONFIGURATION ----------------
HOTKEYS = {
    # English Sections
    'u': 'English Verse 1',
    'i': 'English Verse 2',
    'o': 'English Verse 3',
    'p': 'English Verse 4',
    't': 'English Pre-Chorus 1',
    'g': 'English Pre-Chorus 2',
    'y': 'English Chorus 1',
    'h': 'English Chorus 2',
    'r': 'English Bridge',
    'w': 'English Ending',
    
    # Chinese Sections
    'v': 'Chinese Verse 1',
    'b': 'Chinese Verse 2',
    'n': 'Chinese Verse 3',
    'm': 'Chinese Verse 4',
    'x': 'Chinese Pre-Chorus 1',
    'd': 'Chinese Pre-Chorus 2',
    'c': 'Chinese Chorus 1',
    'f': 'Chinese Chorus 2',
    's': 'Chinese Bridge',
    'z': 'Chinese Ending'
}
# =================================================

cc = opencc.OpenCC('s2t.json')

def force_quit(sig, frame):
    print("\n\n🛑 Force terminating the script...")
    os._exit(0)
signal.signal(signal.SIGINT, force_quit)

# --- GLOBAL VARIABLES FOR CUEING ---
poller = None
cued_slide_index = None
last_key_press = {}
DOUBLE_PRESS_DELAY = 0.4

# --- MODEL LOADERS ---
def load_pure_english_model(model_dir):
    if not os.path.exists(model_dir):
        print(f"❌ ERROR: Missing '{model_dir}' folder.")
        sys.exit(1)
        
    encoder = [f for f in glob.glob(f"{model_dir}/**/*encoder*.onnx", recursive=True) if "int8" not in f]
    decoder = [f for f in glob.glob(f"{model_dir}/**/*decoder*.onnx", recursive=True) if "int8" not in f]
    joiner = [f for f in glob.glob(f"{model_dir}/**/*joiner*.onnx", recursive=True) if "int8" not in f]
    tokens = glob.glob(f"{model_dir}/**/*tokens.txt", recursive=True)
    
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=encoder[0], decoder=decoder[0], joiner=joiner[0], tokens=tokens[0],
        num_threads=2, sample_rate=16000, feature_dim=80, provider="cpu", model_type="zipformer2"
    )

def load_sensevoice_engine():
    if not os.path.exists(MODEL_DIR_SV):
        print(f"❌ ERROR: Missing '{MODEL_DIR_SV}' folder.")
        sys.exit(1)
        
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{MODEL_DIR_SV}/model.int8.onnx",
        tokens=f"{MODEL_DIR_SV}/tokens.txt",
        num_threads=2, use_itn=False, provider="cpu"
    )

print("🧠 Booting Pure English Engine (Online)...")
rec_en = load_pure_english_model(MODEL_DIR_EN)

print("🧠 Booting SenseVoice Engine (Offline)...")
rec_sv = load_sensevoice_engine()

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
                # TWEAKED: Only target the absolute last 4 to 5 Chinese characters
                target = chinese_chars[-5:] if len(chinese_chars) > 5 else chinese_chars
                return target, self.current_index
            else:
                self.is_chinese_slide = False
                text = self.current_full_text
                # TWEAKED: Only target the last ~25 characters (approx the last 4 to 6 English words)
                target = text[-25:] if len(text) > 25 else text
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
    """A generic helper to fire REST triggers to ProPresenter"""
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
    global cued_slide_index, poller
    
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
    else:
        print(f"\n⚠️ Group '{group_name}' not found in the current song!")

# --- KEYBOARD LISTENER ---
def on_press(key):
    global cued_slide_index
    try:
        # 1. Handle Arrow Keys
        if key == keyboard.Key.right:
            trigger_api("/v1/presentation/active/next/trigger", "➡️  === TRIGGERED NEXT SLIDE ===")
            return
        elif key == keyboard.Key.left:
            trigger_api("/v1/presentation/active/previous/trigger", "⬅️  === TRIGGERED PREVIOUS SLIDE ===")
            return
        elif key == keyboard.Key.down:
            trigger_api("/v1/playlist/active/next/trigger", "⬇️  === TRIGGERED NEXT SONG IN PLAYLIST ===")
            return
        elif key == keyboard.Key.up:
            trigger_api("/v1/playlist/active/previous/trigger", "⬆️  === TRIGGERED PREVIOUS SONG IN PLAYLIST ===")
            return

        # 2. Handle Alphanumeric Hotkeys
        if hasattr(key, 'char') and key.char:
            k = key.char.lower()
            if k in HOTKEYS:
                now = time.time()
                group_name = HOTKEYS[k]
                
                if k in last_key_press and (now - last_key_press[k]) < DOUBLE_PRESS_DELAY:
                    jump_to_group(group_name, immediate=True)
                    last_key_press[k] = 0 
                else:
                    jump_to_group(group_name, immediate=False)
                    last_key_press[k] = now
    except Exception:
        pass

def fast_smart_score(heard_text, target, is_chinese_slide):
    if not heard_text or not target: return 0, heard_text
    heard_clean = re.sub(r'<\|.*?\|>', '', heard_text).strip()
    
    if is_chinese_slide:
        heard_cn = "".join(re.findall(r'[\u4e00-\u9FFF]', heard_clean))
        if len(heard_cn) < 2: return 0, heard_cn
        heard_trad = cc.convert(heard_cn)
        
        score = fuzz.partial_ratio(target, heard_trad)
        
        # PENALTY: If they haven't sung the end of the phrase yet, dock 25 points
        if len(heard_trad) < len(target) - 1:
            score -= 25
            
        return score, heard_trad
    else:
        if len(heard_clean) < 8: return 0, heard_clean
        score = fuzz.partial_ratio(target.lower(), heard_clean.lower())
        
        # PENALTY: If the transcribed text is too short, dock 30 points
        if len(heard_clean) < len(target) - 4: 
            score -= 30
            
        return score, heard_clean

def main():
    global poller
    p = pyaudio.PyAudio()
    
    print("\n🎤 Available Microphones:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:
            print(f"   [{i}] {info.get('name')}")
            
    try:
        mic_idx = int(input("\n👉 Mic Index (Press Enter for default): "))
    except: mic_idx = None

    poller = PP7SmartPoller()
    threading.Thread(target=poller.update_loop, daemon=True).start()

    # Start Global Keyboard Listener
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, input_device_index=mic_idx, frames_per_buffer=2048)
    
    print(f"\n✅ Engine & Keyboard Hotkeys Ready! Monitoring...")
    
    current_slide = -1
    sherpa_stream_en = None 
    
    cn_audio_buffer = np.array([], dtype=np.float32)
    PROCESS_INTERVAL = int(16000 * 0.4) 
    MAX_BUFFER_SIZE = int(16000 * 4.0)  
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
                if not is_chinese:
                    sherpa_stream_en = rec_en.create_stream() 
                else:
                    cn_audio_buffer = np.array([], dtype=np.float32) 
                stream.read(stream.get_read_available(), exception_on_overflow=False)
                continue

            data = stream.read(2048, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            if not is_chinese:
                # --- ENGLISH MODE ---
                sherpa_stream_en.accept_waveform(16000, samples)
                while rec_en.is_ready(sherpa_stream_en):
                    rec_en.decode_stream(sherpa_stream_en)

                result = rec_en.get_result(sherpa_stream_en)
                if result:
                    score, clean_heard = fast_smart_score(result, target, False)
                    sys.stdout.write(f"\r👂 Heard [EN]: {clean_heard[-40:]:<40} | Match: {score}%   ")
                    sys.stdout.flush()

                    if score >= EN_THRESHOLD:
                        handle_lyric_trigger()
                        sherpa_stream_en = rec_en.create_stream() 
                        stream.read(stream.get_read_available(), exception_on_overflow=False) 
                        time.sleep(0.5) 

            else:
                # --- CANTONESE MODE ---
                cn_audio_buffer = np.concatenate((cn_audio_buffer, samples))
                samples_since_last_process += len(samples)

                if len(cn_audio_buffer) > MAX_BUFFER_SIZE:
                    cn_audio_buffer = cn_audio_buffer[-MAX_BUFFER_SIZE:]

                if samples_since_last_process >= PROCESS_INTERVAL:
                    s_stream = rec_sv.create_stream()
                    s_stream.accept_waveform(16000, cn_audio_buffer)
                    rec_sv.decode_stream(s_stream)
                    result = s_stream.result.text
                    
                    if result:
                        score, clean_heard = fast_smart_score(result, target, True)
                        display_text = clean_heard[-30:] if len(clean_heard) > 30 else clean_heard
                        sys.stdout.write(f"\r👂 Heard [CN]: {display_text:<30} | Match: {score}%   ")
                        sys.stdout.flush()

                        if score >= CN_THRESHOLD:
                            handle_lyric_trigger()
                            cn_audio_buffer = np.array([], dtype=np.float32) 
                            samples_since_last_process = 0
                            stream.read(stream.get_read_available(), exception_on_overflow=False) 
                            time.sleep(0.5) 

                    samples_since_last_process = 0 

        except KeyboardInterrupt: 
            break
        except Exception as e: 
            pass

if __name__ == "__main__":
    main()