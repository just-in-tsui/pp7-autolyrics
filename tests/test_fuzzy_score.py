"""Unit tests for audio_pipeline.fast_smart_score and compute_pre_trigger_delay.

These tests cover the pure scoring logic: fuzzy match, repetition enforcer,
missing-content penalty, opencc Simplified→Traditional conversion, control-token
stripping. They never load the SenseVoice model.
"""

import pytest

from audio_pipeline import (
    CN_THRESHOLD,
    EN_THRESHOLD,
    compute_pre_trigger_delay,
    fast_smart_score,
)


# ----- exact / empty / too-short cases -------------------------------------

def test_exact_match_en_returns_perfect_score():
    score, heard = fast_smart_score(
        "amazing grace how sweet the sound",
        "amazing grace how sweet the sound",
        is_chinese_slide=False,
    )
    assert score == 100
    assert heard == "amazing grace how sweet the sound"


def test_exact_match_cn_returns_perfect_score():
    score, heard = fast_smart_score("奇異恩典", "奇異恩典", is_chinese_slide=True)
    assert score == 100
    assert heard == "奇異恩典"


def test_empty_heard_returns_zero():
    score, heard = fast_smart_score("", "amazing grace", is_chinese_slide=False)
    assert score == 0
    assert heard == ""


def test_empty_target_returns_zero():
    score, _ = fast_smart_score("amazing grace", "", is_chinese_slide=False)
    assert score == 0


def test_heard_too_short_en_returns_zero():
    # < 5 chars after token stripping → bail out
    score, heard = fast_smart_score("hi", "amazing grace", is_chinese_slide=False)
    assert score == 0
    assert heard == "hi"


def test_heard_too_short_cn_returns_zero():
    # < 2 CJK chars after filtering → bail out
    score, heard = fast_smart_score("你", "我愛你", is_chinese_slide=True)
    assert score == 0
    assert heard == "你"


# ----- input normalization -------------------------------------------------

def test_sensevoice_control_tokens_stripped():
    score, heard = fast_smart_score(
        "<|en|><|HAPPY|>amazing grace how sweet<|nospeech|>",
        "amazing grace how sweet",
        is_chinese_slide=False,
    )
    assert score == 100
    assert heard == "amazing grace how sweet"


def test_en_punctuation_stripped():
    score, heard = fast_smart_score(
        "Amazing, grace! How sweet?", "amazing grace how sweet", is_chinese_slide=False
    )
    assert score == 100
    assert "," not in heard
    assert "!" not in heard


def test_cn_non_cjk_chars_filtered_out():
    # Latin characters and spaces should be stripped before scoring
    score, heard = fast_smart_score(
        "I love 我愛你 yeah", "我愛你", is_chinese_slide=True
    )
    assert score == 100
    assert heard == "我愛你"


def test_opencc_simplified_to_traditional():
    # Heard simplified "爱" must map to traditional "愛" before fuzzy matching
    score, heard = fast_smart_score("爱我中华", "愛我中華", is_chinese_slide=True)
    assert score == 100
    assert heard == "愛我中華"


# ----- missing-content penalty ---------------------------------------------

def test_en_missing_letter_penalty_lowers_score():
    # "amazing grace" missing the last "how sweet" → 9 missing chars → -27
    full_score, _ = fast_smart_score(
        "amazing grace how sweet", "amazing grace how sweet", is_chinese_slide=False
    )
    short_score, short_heard = fast_smart_score(
        "amazing grace", "amazing grace how sweet", is_chinese_slide=False
    )
    assert short_heard == "amazing grace"
    # Full match scores 100; short match scores 100 - 3 * (23-13) = 100 - 30 = 70
    # (clean_target has 23 chars including spaces, clean_heard has 13)
    assert short_score == full_score - 30


def test_cn_missing_char_penalty_lowers_score():
    full_score, _ = fast_smart_score("奇異恩典何等甘甜", "奇異恩典何等甘甜", is_chinese_slide=True)
    short_score, short_heard = fast_smart_score("奇異恩典", "奇異恩典何等甘甜", is_chinese_slide=True)
    assert short_heard == "奇異恩典"
    # 4 missing chars * -6 = -24
    assert short_score == full_score - 24


# ----- repetition enforcer -------------------------------------------------

def test_en_repetition_enforcer_blocks_early_trigger():
    # Singer only sang the first "hallelujah" — algorithm must NOT trigger.
    score_first_rep, _ = fast_smart_score(
        "hallelujah", "hallelujah hallelujah hallelujah", is_chinese_slide=False
    )
    score_all_reps, _ = fast_smart_score(
        "hallelujah hallelujah hallelujah",
        "hallelujah hallelujah hallelujah",
        is_chinese_slide=False,
    )
    assert score_all_reps >= EN_THRESHOLD, "full repetition should clear threshold"
    assert score_first_rep < EN_THRESHOLD, "partial repetition must be blocked"
    # Enforcer applies -40, then missing-letter penalty stacks on top.
    assert score_all_reps - score_first_rep >= 40


def test_cn_repetition_enforcer_blocks_early_trigger():
    score_first_rep, _ = fast_smart_score(
        "哈利路亞", "哈利路亞哈利路亞哈利路亞", is_chinese_slide=True
    )
    score_all_reps, _ = fast_smart_score(
        "哈利路亞哈利路亞哈利路亞",
        "哈利路亞哈利路亞哈利路亞",
        is_chinese_slide=True,
    )
    assert score_all_reps >= CN_THRESHOLD
    assert score_first_rep < CN_THRESHOLD
    assert score_all_reps - score_first_rep >= 40


# ----- reverent-character normalization (祢/祂 → 你/他) ---------------------

def test_reverent_you_normalized_for_matching():
    # Slide uses 祢 (reverent God-you); SenseVoice transcribes the sound as 你.
    # They must match after normalization.
    score, _ = fast_smart_score("走你道路", "走祢道路", is_chinese_slide=True)
    assert score == 100


def test_reverent_he_normalized_for_matching():
    # 祂 (reverent God-he) ↔ 他
    score, _ = fast_smart_score("讚美他", "讚美祂", is_chinese_slide=True)
    assert score == 100


# ----- regression: the 走祢道路 repetition bug ----------------------------
# Live output: target "走祢道路走祢道路", singer heard as "走你道路走你道" (7/8
# chars, ~1.75 of 2 reps). Old algorithm scored this 25% (71 base − 40 flat
# enforcer − 6 missing) and never triggered. After fix it must clear threshold.

def test_repeated_chinese_near_complete_triggers():
    score, _ = fast_smart_score("走你道路走你道", "走祢道路走祢道路", is_chinese_slide=True)
    assert score >= CN_THRESHOLD, f"near-complete repeat should trigger, got {score}"


def test_repeated_chinese_single_rep_still_blocked():
    # Singer sang the looping phrase only once — must NOT trigger early.
    score, _ = fast_smart_score("走你道路", "走祢道路走祢道路", is_chinese_slide=True)
    assert score < CN_THRESHOLD, f"single repeat must be blocked, got {score}"


def test_repeated_chinese_graduated_not_cliff():
    # Score should rise monotonically as more of the repeated phrase is heard,
    # rather than jumping at a single cliff.
    target = "走祢道路走祢道路"
    s1, _ = fast_smart_score("走你道路", target, is_chinese_slide=True)        # 1 rep
    s2, _ = fast_smart_score("走你道路走你", target, is_chinese_slide=True)      # 1.5 reps
    s3, _ = fast_smart_score("走你道路走你道", target, is_chinese_slide=True)    # 1.75 reps
    assert s1 < s2 < s3


def test_en_no_enforcer_when_target_has_no_repetition():
    # "the sound" target — anchor word "sound" appears once → no penalty even if
    # heard is partial
    score_full, _ = fast_smart_score(
        "how sweet the sound", "how sweet the sound", is_chinese_slide=False
    )
    score_partial, _ = fast_smart_score(
        "how sweet the", "how sweet the sound", is_chinese_slide=False
    )
    # Only missing-letter penalty applies, not the -40 enforcer
    assert score_full - score_partial < 40


# ----- compute_pre_trigger_delay ------------------------------------------

def test_delay_zero_when_heard_matches_target_length():
    assert compute_pre_trigger_delay("abc", "abc", is_chinese=False, is_slow_mode=False) == 0.0


def test_delay_zero_when_heard_longer_than_target():
    assert compute_pre_trigger_delay("abc", "abcdef", is_chinese=False, is_slow_mode=False) == 0.0


def test_delay_en_normal_mode():
    # target 5 chars, heard 2 chars → missing=3 → 3 * 0.15 = 0.45
    assert compute_pre_trigger_delay("abcde", "ab", is_chinese=False, is_slow_mode=False) == pytest.approx(0.45)


def test_delay_en_normal_mode_cap_at_2_5():
    # Many missing chars → capped at 2.5
    assert compute_pre_trigger_delay("a" * 100, "ab", is_chinese=False, is_slow_mode=False) == 2.5


def test_delay_en_slow_mode():
    # 4 missing → 4 * 0.25 = 1.0
    assert compute_pre_trigger_delay("abcde", "a", is_chinese=False, is_slow_mode=True) == pytest.approx(1.0)


def test_delay_en_slow_mode_cap_at_4_0():
    assert compute_pre_trigger_delay("a" * 100, "ab", is_chinese=False, is_slow_mode=True) == 4.0


def test_delay_cn_normal_mode():
    # 2 missing CN chars → 2 * 0.3 = 0.6
    assert compute_pre_trigger_delay("一二三", "一", is_chinese=True, is_slow_mode=False) == pytest.approx(0.6)


def test_delay_cn_normal_mode_cap_at_2_5():
    assert compute_pre_trigger_delay("一" * 100, "一", is_chinese=True, is_slow_mode=False) == 2.5


def test_delay_cn_slow_mode():
    # 2 missing → 2 * 0.5 = 1.0
    assert compute_pre_trigger_delay("一二三", "一", is_chinese=True, is_slow_mode=True) == pytest.approx(1.0)


def test_delay_cn_slow_mode_cap_at_3_5():
    assert compute_pre_trigger_delay("一" * 100, "一", is_chinese=True, is_slow_mode=True) == 3.5
