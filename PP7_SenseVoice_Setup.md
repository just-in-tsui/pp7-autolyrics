# PP7 Auto-Lyric System (SenseVoice) Documentation

## 1. Overview Logic Flow

This script acts as a bridge between a live singer and **ProPresenter 7**, automatically advancing slides by listening to the singer's voice in real-time. 

### Core Loop:
1. **State Polling (ProPresenter API):** The `PP7SmartPoller` constantly polls the ProPresenter 7 Network API (`127.0.0.1:1025`) in a background thread. It fetches the text of the currently active slide and caches the entire presentation.
2. **Target Extraction & Language Detection:** When a new slide is active, the script analyzes the text. If it detects Chinese characters, it enters **Cantonese/Yue mode**. Otherwise, it defaults to **English mode**. It then extracts the last few words/characters of the slide to serve as the "Target Phrase" it needs to listen for.
3. **Audio Capture:** Using `pyaudio`, the script continuously records audio from the selected microphone into a rolling numpy buffer.
4. **Speech-to-Text (SenseVoice):** Every ~0.4 seconds, the audio buffer is fed into a local `sherpa-onnx` SenseVoice engine running on the CPU. The script runs two isolated engines in memory to handle English and Cantonese with high precision.
5. **Fuzzy Matching & Triggering:** 
   - The transcribed text is converted from Simplified to Traditional Chinese (if applicable) using `opencc`.
   - The script uses `thefuzz` to compare the transcription against the "Target Phrase".
   - If the match score exceeds the defined threshold (55% for English, 45% for Chinese), it triggers an API call to ProPresenter to advance to the next slide.
   - **Dynamic Delay:** If the singer matched the target early (e.g., missed the last word), the script applies a calculated micro-delay so the slide doesn't advance prematurely.
6. **Keyboard Overrides:** A global hotkey listener (`pynput`) runs in the background. It allows the operator to pause/resume the listener (using `/`) or queue up jumps to specific song sections (Verse, Chorus, Bridge) using dedicated keyboard keys, which will trigger instantly on a double-tap or wait until the current section finishes.

---

## 2. Technical Setup Guide (For New Machines)

Follow these steps to deploy the environment on a new macOS, Windows, or Linux machine.

### Prerequisites

1. **Python 3.9+** (Python 3.12 is recommended based on your current setup).
2. **PortAudio:** Required for `pyaudio` to capture microphone input.
   - **macOS:** Open Terminal and install via Homebrew:
     ```bash
     brew install portaudio
     ```
   - **Windows:** Generally not required, as `pyaudio` installs with pre-built binaries.
   - **Linux:** `sudo apt-get install portaudio19-dev`

### Step 1: Clone/Copy the Project
Ensure your project folder contains `PP7_SenseVoice_Only.py`.

### Step 2: Create a Virtual Environment
It's highly recommended to isolate the dependencies.
```bash
# Navigate to the project folder
cd /path/to/OptionPricer

# Create a virtual environment
python3 -m venv optVenv

# Activate the environment
# On macOS/Linux:
source optVenv/bin/activate
# On Windows:
# optVenv\Scripts\activate
```

### Step 3: Install Required Libraries
Install the necessary Python packages:
```bash
pip install requests sherpa-onnx thefuzz pyaudio numpy opencc pynput
```

### Step 4: Download & Extract the SenseVoice Model
The script relies on an offline, CPU-optimized SenseVoice model to transcribe audio.
1. Download the `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` archive. *(Note: you already have the `.tar.bz2` version in your folder).*
2. Extract it into your project directory. Ensure the folder name matches the `MODEL_DIR_SV` variable in the script exactly.
   
   Your structure should look like this:
   ```text
   OptionPricer/
   ├── PP7_SenseVoice_Only.py
   └── sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/
       ├── model.int8.onnx
       ├── tokens.txt
       └── ...
   ```

### Step 5: Configure ProPresenter 7
1. Open ProPresenter 7.
2. Go to **ProPresenter > Settings (or Preferences) > Network**.
3. **Check** the box to Enable Network.
4. Set the Port to `1025`.
5. Ensure no password is set, or if one is required, you will need to update the `session.get()` calls in the script to include authentication headers.

### Step 6: Running the System
1. Ensure your microphone/audio interface is plugged in.
2. Start the script:
   ```bash
   python PP7_SenseVoice_Only.py
   ```
3. **Select Microphone:** The console will list all available audio devices. Type the index number of the microphone you want to listen to and press `Enter` (or just press `Enter` to use the system default).
4. **Start Presenting:** Click on a slide in ProPresenter. The console should immediately log the "Target Phrase" it is listening for.

### Troubleshooting
* **`ModuleNotFoundError: No module named 'pyaudio'` or build errors during install:** Ensure PortAudio is installed on your OS (`brew install portaudio`).
* **`❌ ERROR: Missing 'sherpa-onnx-sense-voice...' folder.`:** The script cannot find the AI model. Check that the folder is unzipped and named correctly in the exact same directory as the `.py` script.
* **ProPresenter doesn't advance:** Ensure ProPresenter's Network setting is enabled on port `1025` and that no firewall is blocking `127.0.0.1`.
* **Keyboard Hotkeys require permissions:** On macOS, `pynput` requires Accessibility permissions. You may need to go to System Settings > Privacy & Security > Accessibility and allow your Terminal or IDE.
