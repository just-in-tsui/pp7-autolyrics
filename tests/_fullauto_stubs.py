"""Imports PP7_SenseVoice_FullAuto for tests that don't need real hardware.

The script imports pyaudio / sherpa_onnx / pynput at module level, none of which
the localizer logic touches. Only modules that are genuinely absent get stubbed,
so on a real dev machine the tests run against the real imports and nothing is
masked.
"""

import sys
import types
from importlib import import_module
from importlib.util import find_spec


def _stub_missing(name, build):
    if find_spec(name) is None and name not in sys.modules:
        sys.modules[name] = build()


def _pyaudio():
    m = types.ModuleType("pyaudio")
    m.paInt16 = 8
    m.PyAudio = object
    return m


def _sherpa():
    m = types.ModuleType("sherpa_onnx")
    m.OfflineRecognizer = object
    return m


def _pynput():
    kb = types.ModuleType("pynput.keyboard")

    class _Key:
        right = "right"
        left = "left"
        up = "up"
        down = "down"

    kb.Key = _Key
    kb.Listener = object
    root = types.ModuleType("pynput")
    root.keyboard = kb
    sys.modules.setdefault("pynput.keyboard", kb)
    return root


_stub_missing("pyaudio", _pyaudio)
_stub_missing("sherpa_onnx", _sherpa)
_stub_missing("pynput", _pynput)

fa = import_module("PP7_SenseVoice_FullAuto")

# A deck containing every ambiguity the localizer actually has to survive: a
# verbatim repeated chorus, sections sharing an opening phrase, and a CJK block.
SONG = [
    {"text": "Amazing grace how sweet the sound",     "group": "English Verse 1"},   # 0
    {"text": "That saved a wretch like me",           "group": "English Verse 1"},   # 1
    {"text": "I once was lost but now am found",      "group": "English Verse 2"},   # 2
    {"text": "Was blind but now I see",               "group": "English Verse 2"},   # 3
    {"text": "My chains are gone I've been set free", "group": "English Chorus 1"},  # 4
    {"text": "My God my Saviour has ransomed me",     "group": "English Chorus 1"},  # 5
    {"text": "My chains are gone I've been set free", "group": "English Chorus 2"},  # 6
    {"text": "My God my Saviour has ransomed me",     "group": "English Chorus 2"},  # 7
    {"text": "祢的信實極其廣大",                        "group": "Chinese Verse 1"},   # 8
    {"text": "我心讚美主的恩典",                        "group": "Chinese Verse 1"},   # 9
]
