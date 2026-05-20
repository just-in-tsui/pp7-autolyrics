"""Unit tests for PP7SmartPoller.get_target — slide-tail extraction and
language detection.

These tests instantiate the poller directly without any HTTP traffic; they only
exercise the in-memory state-to-target transformation.
"""

from pp7_poller import PP7SmartPoller


def _poller_with(text: str, idx: int = 0) -> PP7SmartPoller:
    """Build a poller in a state as if PP7 had just sent a slide with `text`."""
    p = PP7SmartPoller()
    p.current_full_text = text
    p.current_index = idx
    return p


# ----- English slides ------------------------------------------------------

def test_english_long_slide_returns_last_35_chars():
    text = "Amazing grace how sweet the sound that saved a wretch like me"
    p = _poller_with(text)
    target, idx = p.get_target()
    assert target == text[-35:].strip()
    assert idx == 0
    assert p.is_chinese_slide is False


def test_english_short_slide_returns_full_text():
    text = "Saved a wretch"  # 14 chars, well under 35
    p = _poller_with(text)
    target, _ = p.get_target()
    assert target == "Saved a wretch"
    assert p.is_chinese_slide is False


def test_english_target_is_stripped():
    text = "   Amazing grace   "
    p = _poller_with(text)
    target, _ = p.get_target()
    # last 35 chars then stripped — full text is < 35 chars, so we get the
    # stripped form
    assert target == "Amazing grace"


# ----- Chinese / Cantonese slides -----------------------------------------

def test_chinese_long_slide_returns_last_10_cjk_chars():
    text = "奇異恩典何等甘甜我罪已得赦免前我失喪今被尋回"
    p = _poller_with(text)
    target, _ = p.get_target()
    assert target == text[-10:]
    assert len(target) == 10
    assert p.is_chinese_slide is True


def test_chinese_short_slide_returns_full_cjk():
    text = "奇異恩典"
    p = _poller_with(text)
    target, _ = p.get_target()
    assert target == "奇異恩典"
    assert p.is_chinese_slide is True


def test_mixed_language_slide_is_treated_as_chinese():
    # Any CJK char flips the flag — this is current (and important) behavior.
    # If you ever want mixed-language support, this test will catch the change.
    text = "Verse 1 副歌 chorus"
    p = _poller_with(text)
    target, _ = p.get_target()
    # CJK extracted: 副歌 — only 2 chars, returned in full
    assert target == "副歌"
    assert p.is_chinese_slide is True


def test_chinese_non_cjk_chars_are_filtered_from_target():
    # Numbers, latin letters, and punctuation in a CJK slide get stripped
    text = "副歌 1: 哈利路亞! Amen"
    p = _poller_with(text)
    target, _ = p.get_target()
    assert target == "副歌哈利路亞"
    assert p.is_chinese_slide is True


# ----- empty / unset state -------------------------------------------------

def test_empty_text_returns_empty_target():
    p = PP7SmartPoller()  # nothing set
    target, idx = p.get_target()
    assert target == ""
    assert idx == -1


def test_explicitly_empty_full_text():
    p = _poller_with("", idx=3)
    target, idx = p.get_target()
    assert target == ""
    assert idx == 3


# ----- get_slide_info_smart shape handling --------------------------------

def test_slide_info_smart_with_slide_index_int():
    p = PP7SmartPoller()
    idx, uuid = p.get_slide_info_smart({
        "slide_index": 5,
        "presentation_index": {"index": 5, "presentation_id": {"uuid": "abc"}},
    })
    assert idx == 5
    assert uuid == "abc"


def test_slide_info_smart_with_only_presentation_index():
    # No top-level slide_index — falls back to presentation_index.index
    p = PP7SmartPoller()
    idx, uuid = p.get_slide_info_smart({
        "presentation_index": {"index": 2, "presentation_id": {"uuid": "xyz"}},
    })
    assert idx == 2
    assert uuid == "xyz"


def test_slide_info_smart_with_missing_fields():
    p = PP7SmartPoller()
    idx, uuid = p.get_slide_info_smart({})
    assert idx == -1
    assert uuid is None


def test_slide_info_smart_with_negative_slide_index_falls_back():
    # slide_index=-1 means "no active slide"; should fall back to presentation_index
    p = PP7SmartPoller()
    idx, uuid = p.get_slide_info_smart({
        "slide_index": -1,
        "presentation_index": {"index": 7, "presentation_id": {"uuid": "fallback"}},
    })
    assert idx == 7
    assert uuid == "fallback"
