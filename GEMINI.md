# PP7 Auto-Lyric System (SenseVoice)

An automated slide advancement system for ProPresenter 7 that uses real-time speech-to-text to follow a live singer.

## Project Overview
This project acts as an intelligent bridge between a live microphone input and ProPresenter 7. It listens to the singer, transcribes the audio locally using the **SenseVoice** AI model, and automatically triggers the "Next Slide" command when it detects the end of the current slide's lyrics.

### Core Technologies
- **Language:** Python 3.9+
- **Speech-to-Text:** `sherpa-onnx` with `SenseVoice` (offline, CPU-optimized).
- **ProPresenter API:** Integration via the ProPresenter 7 Network API (Port 1025).
- **Audio Processing:** `pyaudio` (requires `PortAudio`) and `numpy`.
- **Fuzzy Matching:** `thefuzz` / `RapidFuzz` for resilient lyric detection.
- **Transcription Translation:** `opencc` for Simplified to Traditional Chinese conversion.
- **Automation:** `pynput` for global keyboard hotkeys and manual overrides.

## System Architecture
1.  **Polling Engine (`PP7SmartPoller`):** A background thread that constantly monitors ProPresenter 7 to stay synced with the active presentation and slide text.
2.  **Target Extraction:** Automatically identifies the "Target Phrase" (the last few words/characters of a slide) and detects the language (English vs. Cantonese/Yue).
3.  **STT Pipeline:** Uses two isolated `SenseVoice` engines in memory to handle English and Cantonese with high precision.
4.  **Fuzzy Triggering:** Compares live transcription against the target phrase. If the match score exceeds a threshold, it triggers the next slide.
5.  **Dynamic Delay:** Intelligently calculates a micro-delay if the singer finishes a phrase early, ensuring natural transitions.

## Setup and Installation

### Prerequisites
- **Python 3.9+** (3.12 recommended).
- **PortAudio:** Required for microphone access.
    - macOS: `brew install portaudio`
    - Linux: `sudo apt-get install portaudio19-dev`

### Installation Steps
1.  **Virtual Environment:**
    ```bash
    python3 -m venv optVenv
    source optVenv/bin/activate
    ```
2.  **Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Model Setup:**
    Ensure the `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` directory exists in the root and contains the `.onnx` model and `tokens.txt`.

## Running the System
1.  **ProPresenter Config:** Enable **Network** in Settings > Network (Port 1025, no password).
2.  **Execution:**
    ```bash
    python PP7_SenseVoice_Only.py
    ```
3.  **Selection:** Choose the correct microphone index from the list provided on startup.

## Operational Controls

### Auto-Lyric Controls
- `/`: **Pause/Resume** the listener. Useful for speeches or unplanned segments.

### Manual Navigation
- `Left/Right Arrows`: Previous/Next Slide.
- `Up/Down Arrows`: Previous/Next Song in Playlist.

### Section Jumps (Hotkeys)
The system supports "Cued Jumps." Pressing a key once will queue that section to trigger after the current one finishes. **Double-tapping** triggers the jump instantly.
- **English:** `u` (Verse 1), `y` (Chorus 1), `r` (Bridge), `w` (Ending).
- **Chinese:** `v` (Verse 1), `c` (Chorus 1), `s` (Bridge), `z` (Ending).
*(See `HOTKEYS` dictionary in `PP7_SenseVoice_Only.py` for full mapping).*

## Development Conventions
- **Thresholds:** `EN_THRESHOLD` (55%) and `CN_THRESHOLD` (45%) can be adjusted in the script to tune sensitivity.
- **Language Detection:** Detection is character-based; slides with Chinese characters automatically switch the engine to `yue` (Cantonese) mode.
- **Safety:** The script includes a `force_quit` handler (SIGINT) for clean exits.
