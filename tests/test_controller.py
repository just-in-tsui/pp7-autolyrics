"""Tests for AppController action logic — cued jumps, end-of-section detection,
mode/pause flags, nav endpoints, keyboard mapping.

No audio, no UI, no SenseVoice model. The poller's slide_cache is set directly
(no HTTP); PP7 trigger calls are mocked with `responses`.
"""

import responses

from controller import AppController

BASE = "http://127.0.0.1:1025"

CACHE = [
    {"text": "verse one a", "group": "English Verse 1"},
    {"text": "verse one b", "group": "English Verse 1"},
    {"text": "chorus a", "group": "English Chorus 1"},
    {"text": "chorus b", "group": "English Chorus 1"},
]


def make_controller(current_index=0):
    c = AppController()
    c.poller.slide_cache = list(CACHE)
    c.poller.current_index = current_index
    return c


# ----- jump_to_group resolution -------------------------------------------

def test_jump_to_group_exact_match_cues():
    c = make_controller()
    c.jump_to_group("English Chorus 1", immediate=False)
    assert c.cued_slide_index == 2
    assert c.cued_group_name == "English Chorus 1"
    assert c.status.cued_group == "English Chorus 1"


def test_jump_to_group_substring_match():
    c = make_controller()
    c.jump_to_group("Chorus 1", immediate=False)  # substring of "English Chorus 1"
    assert c.cued_slide_index == 2


def test_jump_to_group_no_match_is_noop():
    c = make_controller()
    c.jump_to_group("Bridge", immediate=False)  # not in this cache
    assert c.cued_slide_index is None
    assert c.cued_group_name is None


@responses.activate
def test_jump_to_group_immediate_triggers_and_clears():
    responses.add(responses.GET, f"{BASE}/v1/presentation/active/2/trigger", status=200)
    c = make_controller()
    c.jump_to_group("English Chorus 1", immediate=True)
    assert c.cued_slide_index is None
    assert c.cued_group_name is None
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url.endswith("/active/2/trigger")


# ----- handle_lyric_trigger end-of-section detection ----------------------

@responses.activate
def test_cued_jump_fires_at_section_boundary():
    responses.add(responses.GET, f"{BASE}/v1/presentation/active/2/trigger", status=200)
    c = make_controller(current_index=1)  # last slide of the Verse 1 group
    c.cued_slide_index = 2
    c.cued_group_name = "English Chorus 1"
    c.handle_lyric_trigger()
    assert c.cued_slide_index is None
    assert responses.calls[0].request.url.endswith("/active/2/trigger")


@responses.activate
def test_mid_section_advances_next_and_keeps_cue():
    responses.add(responses.GET, f"{BASE}/v1/presentation/active/next/trigger", status=200)
    c = make_controller(current_index=0)  # next slide is same group -> not end of section
    c.cued_slide_index = 2
    c.cued_group_name = "English Chorus 1"
    c.handle_lyric_trigger()
    assert c.cued_slide_index == 2  # cue preserved
    assert responses.calls[0].request.url.endswith("/active/next/trigger")


# ----- flags surface in status --------------------------------------------

def test_toggle_pause_updates_status():
    c = make_controller()
    assert c.is_paused is False and c.status.listening is True
    c.toggle_pause()
    assert c.is_paused is True and c.status.listening is False
    c.toggle_pause()
    assert c.status.listening is True


def test_fast_slow_mode_updates_status():
    c = make_controller()
    c.set_slow_mode()
    assert c.is_slow_mode is True and c.status.slow_mode is True
    c.set_fast_mode()
    assert c.is_slow_mode is False and c.status.slow_mode is False


def test_clear_cued():
    c = make_controller()
    c.jump_to_group("English Chorus 1", immediate=False)
    assert c.cued_slide_index == 2
    c.clear_cued()
    assert c.cued_slide_index is None
    assert c.status.cued_group is None


# ----- navigation endpoints ------------------------------------------------

@responses.activate
def test_nav_hits_correct_endpoints():
    for path in [
        "/v1/presentation/active/next/trigger",
        "/v1/presentation/active/previous/trigger",
        "/v1/playlist/active/next/trigger",
        "/v1/playlist/active/previous/trigger",
    ]:
        responses.add(responses.GET, f"{BASE}{path}", status=200)
    c = make_controller()
    c.next_slide(); c.prev_slide(); c.next_song(); c.prev_song()
    urls = [call.request.url for call in responses.calls]
    assert any(u.endswith("/presentation/active/next/trigger") for u in urls)
    assert any(u.endswith("/presentation/active/previous/trigger") for u in urls)
    assert any(u.endswith("/playlist/active/next/trigger") for u in urls)
    assert any(u.endswith("/playlist/active/previous/trigger") for u in urls)


# ----- normalized key handling (from the out-of-process listener) ----------

def test_char_pause_key():
    c = make_controller()
    c.handle_key_event("char", "/")
    assert c.is_paused is True


def test_char_mode_keys():
    c = make_controller()
    c.handle_key_event("char", ".")
    assert c.is_slow_mode is True
    c.handle_key_event("char", ",")
    assert c.is_slow_mode is False


def test_char_double_tap_is_immediate(monkeypatch):
    c = make_controller()
    calls = []
    monkeypatch.setattr(c, "jump_to_group", lambda name, immediate: calls.append((name, immediate)))
    c.handle_key_event("char", "y")  # English Chorus 1 -> first tap = cue
    c.handle_key_event("char", "y")  # quick second tap = immediate
    assert calls[0] == ("English Chorus 1", False)
    assert calls[1] == ("English Chorus 1", True)


def test_char_clear_cued_key():
    c = make_controller()
    c.jump_to_group("English Chorus 1", immediate=False)
    c.handle_key_event("char", ";")
    assert c.cued_slide_index is None


@responses.activate
def test_special_arrow_events_hit_endpoints():
    for path in [
        "/v1/presentation/active/next/trigger",
        "/v1/presentation/active/previous/trigger",
        "/v1/playlist/active/next/trigger",
        "/v1/playlist/active/previous/trigger",
    ]:
        responses.add(responses.GET, f"{BASE}{path}", status=200)
    c = make_controller()
    c.handle_key_event("special", "right")
    c.handle_key_event("special", "left")
    c.handle_key_event("special", "down")
    c.handle_key_event("special", "up")
    urls = [call.request.url for call in responses.calls]
    assert any(u.endswith("/presentation/active/next/trigger") for u in urls)
    assert any(u.endswith("/presentation/active/previous/trigger") for u in urls)
    assert any(u.endswith("/playlist/active/next/trigger") for u in urls)
    assert any(u.endswith("/playlist/active/previous/trigger") for u in urls)
