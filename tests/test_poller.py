"""Tests for PP7SmartPoller using `responses` to mock the ProPresenter HTTP API.

Covers:
- fetch_full_song parses multi-group payloads and normalizes whitespace
- The polling loop detects a UUID change and refetches the presentation
- Network errors are swallowed silently (intentional behavior — live performance
  must not crash on a transient PP7 hiccup)
"""

import json
import threading
import time
from pathlib import Path

import pytest
import requests
import responses

import pp7_poller
from pp7_poller import PP7SmartPoller

POLLER_BASE = "http://127.0.0.1:1025/v1"


def _load_json(path: Path):
    return json.loads(path.read_text())


# ----- fetch_full_song -----------------------------------------------------

@responses.activate
def test_fetch_full_song_populates_cache(pp7_responses_dir):
    body = _load_json(pp7_responses_dir / "presentation_active_english.json")
    responses.add(responses.GET, f"{POLLER_BASE}/presentation/active", json=body, status=200)

    poller = PP7SmartPoller()
    assert poller.fetch_full_song() is True

    assert len(poller.slide_cache) == 4
    assert poller.slide_cache[0]["group"] == "Verse 1"
    assert poller.slide_cache[2]["group"] == "Chorus 1"


@responses.activate
def test_fetch_full_song_normalizes_whitespace(pp7_responses_dir):
    body = _load_json(pp7_responses_dir / "presentation_active_english.json")
    responses.add(responses.GET, f"{POLLER_BASE}/presentation/active", json=body, status=200)

    poller = PP7SmartPoller()
    poller.fetch_full_song()

    # The fixture has "\r\n" in slide 0 — must be collapsed to single spaces
    assert "\r" not in poller.slide_cache[0]["text"]
    assert "\n" not in poller.slide_cache[0]["text"]
    assert poller.slide_cache[0]["text"] == "Amazing grace how sweet the sound"


@responses.activate
def test_fetch_full_song_handles_chinese_payload(pp7_responses_dir):
    body = _load_json(pp7_responses_dir / "presentation_active_chinese.json")
    responses.add(responses.GET, f"{POLLER_BASE}/presentation/active", json=body, status=200)

    poller = PP7SmartPoller()
    assert poller.fetch_full_song() is True

    assert poller.slide_cache[0]["group"] == "主歌 1"
    assert "奇異恩典" in poller.slide_cache[0]["text"]


@responses.activate
def test_fetch_full_song_returns_false_on_network_error():
    # No response registered — requests will raise ConnectionError, swallowed silently
    poller = PP7SmartPoller()
    poller.slide_cache = [{"text": "pre-existing", "group": "before"}]

    assert poller.fetch_full_song() is False
    # Original cache should be untouched (the try/except returns before assigning)
    assert poller.slide_cache == [{"text": "pre-existing", "group": "before"}]


@responses.activate
def test_fetch_full_song_returns_false_on_non_200_status():
    responses.add(responses.GET, f"{POLLER_BASE}/presentation/active", status=503)

    poller = PP7SmartPoller()
    poller.slide_cache = [{"text": "pre-existing", "group": "before"}]
    assert poller.fetch_full_song() is False
    assert poller.slide_cache == [{"text": "pre-existing", "group": "before"}]


# ----- custom host / port construction -------------------------------------

def test_poller_uses_configured_host_and_port():
    poller = PP7SmartPoller(host="192.168.1.50", port=9999)
    assert poller.base_url == "http://192.168.1.50:9999/v1"


# ----- update_loop integration (uuid-change-triggers-refetch) -------------

@responses.activate
def test_update_loop_detects_uuid_change_and_refetches(
    pp7_responses_dir, monkeypatch
):
    """When the slide_index endpoint reports a new presentation UUID, the
    poller must call /presentation/active to refresh the slide cache.

    Verified by running update_loop in a daemon thread for a few iterations.
    """
    # Speed up the loop's 150ms inter-poll sleep
    monkeypatch.setattr(pp7_poller.time, "sleep", lambda _s: None)

    slide_index_body = _load_json(pp7_responses_dir / "slide_index_0.json")
    active_body = _load_json(pp7_responses_dir / "presentation_active_english.json")

    responses.add(
        responses.GET, f"{POLLER_BASE}/presentation/slide_index",
        json=slide_index_body, status=200,
    )
    responses.add(
        responses.GET, f"{POLLER_BASE}/presentation/active",
        json=active_body, status=200,
    )

    poller = PP7SmartPoller()
    thread = threading.Thread(target=poller.update_loop, daemon=True)
    thread.start()

    # Spin until the poller picks up the cache, or fail after 2s
    deadline = time.time() + 2.0
    while time.time() < deadline and not poller.slide_cache:
        time.sleep(0.01)

    assert poller.current_uuid == "aaaa-1111"
    assert len(poller.slide_cache) == 4
    assert poller.current_index == 0
    assert "Amazing grace" in poller.current_full_text


@responses.activate
def test_update_loop_silent_on_network_error(monkeypatch):
    """A transient network error must not crash the poll loop or leak an
    exception — the live script relies on this resilience."""
    monkeypatch.setattr(pp7_poller.time, "sleep", lambda _s: None)

    # No responses registered → every request raises ConnectionError
    poller = PP7SmartPoller()
    thread = threading.Thread(target=poller.update_loop, daemon=True)
    thread.start()
    time.sleep(0.1)  # let it churn through several iterations

    # No exception propagated; state is still pristine
    assert poller.slide_cache == []
    assert poller.current_index == -1
    assert thread.is_alive(), "loop must continue running after errors"
