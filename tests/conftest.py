"""Shared pytest fixtures for the PP7 auto-lyric test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def slide_caches_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "slide_caches"


@pytest.fixture(scope="session")
def pp7_responses_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "pp7_responses"


@pytest.fixture(scope="session")
def audio_fixtures_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "audio"


def _load_engine(language: str):
    """Load a SenseVoice engine, skipping the test if the model is missing."""
    if not MODEL_DIR.exists():
        pytest.skip(f"SenseVoice model not found at {MODEL_DIR}")
    import sherpa_onnx

    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(MODEL_DIR / "model.int8.onnx"),
        tokens=str(MODEL_DIR / "tokens.txt"),
        num_threads=2,
        use_itn=False,
        language=language,
        provider="cpu",
    )


@pytest.fixture(scope="session")
def sv_en_engine():
    return _load_engine("en")


@pytest.fixture(scope="session")
def sv_yue_engine():
    return _load_engine("yue")


@pytest.fixture(scope="session")
def load_wav():
    """Returns a function (path) -> float32 numpy array of mono samples at 16kHz."""

    def _load(path: Path) -> np.ndarray:
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed; cannot load WAV fixtures")
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if sr != 16000:
            pytest.skip(f"{path.name} sample rate is {sr}, expected 16000")
        if data.ndim == 2:
            data = data.mean(axis=1).astype(np.float32)
        return data

    return _load
