"""Tests for HotkeyService's queue → controller dispatch.

We do NOT spawn the real pynput child here (it would install a global keyboard
tap and need Input Monitoring permission). Instead we inject a plain queue and
verify poll() drains it and dispatches normalized events to the controller.
"""

import queue

from controller import AppController
from hotkeys import HotkeyService

CACHE = [
    {"text": "a", "group": "English Verse 1"},
    {"text": "b", "group": "English Chorus 1"},
]


class _RecordingController:
    def __init__(self):
        self.events = []

    def handle_key_event(self, kind, value):
        self.events.append((kind, value))


def test_poll_dispatches_all_queued_events():
    rc = _RecordingController()
    svc = HotkeyService(rc)
    svc._queue = queue.Queue()
    svc._queue.put(("char", "/"))
    svc._queue.put(("special", "right"))
    svc._queue.put(("char", "y"))

    svc.poll()

    assert rc.events == [("char", "/"), ("special", "right"), ("char", "y")]


def test_poll_noop_when_not_started():
    rc = _RecordingController()
    svc = HotkeyService(rc)  # never started, _queue is None
    svc.poll()  # must not raise
    assert rc.events == []


def test_poll_drives_real_controller_actions():
    c = AppController()
    c.poller.slide_cache = list(CACHE)
    c.poller.current_index = 0
    svc = HotkeyService(c)
    svc._queue = queue.Queue()
    svc._queue.put(("char", "/"))   # pause toggle
    svc.poll()
    assert c.is_paused is True
