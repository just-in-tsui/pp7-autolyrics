"""Tests for the full-auto song-wide localizer.

These cover the decision logic that decides *where in the song* the singer is —
the part that can move the screen on its own during a live service, so it is the
part that has to be right. No model, microphone or ProPresenter needed.
"""

import pytest

from tests._fullauto_stubs import SONG, fa


@pytest.fixture
def loc():
    localizer = fa.SongLocalizer()
    localizer.build(SONG)
    return localizer


def feed(localizer, en="", cn="", current=0, times=1, t0=1000.0, step=0.5):
    """Run N identical decode cycles and return the last decision."""
    decision = None
    for i in range(times):
        decision = localizer.observe(cn, en, current, now=t0 + i * step)
    return decision


# ----- normalization -------------------------------------------------------
def test_slide_with_any_cjk_is_chinese_and_drops_latin():
    norm, is_cn = fa.normalize_slide_text("祢的信實極其廣大 (Great is thy faithfulness)")
    assert is_cn
    assert norm == "祢的信實極其廣大"


def test_english_slide_normalizes_punctuation_and_case():
    norm, is_cn = fa.normalize_slide_text("My chains are GONE, I've been set free!")
    assert not is_cn
    assert norm == "my chains are gone ive been set free"


def test_heard_strips_sensevoice_tags():
    assert fa.normalize_heard("<|en|><|NEUTRAL|>Amazing Grace!", False) == "amazing grace"


def test_heard_chinese_converted_to_traditional():
    # PP7 slides are Traditional; SenseVoice emits Simplified.
    assert fa.normalize_heard("我心赞美", True) == "我心讚美"


# ----- index building ------------------------------------------------------
def test_build_detects_both_languages_and_verbatim_duplicates(loc):
    assert loc.languages == {"en", "cn"}
    assert sorted(loc.dup_groups.values()) == [[4, 6], [5, 7]]
    assert loc.dup_peers(4) == [4, 6]
    assert loc.dup_peers(0) == [0]


def test_build_ignores_blank_slides():
    localizer = fa.SongLocalizer()
    localizer.build([{"text": "", "group": "Blank"}, {"text": "", "group": "Blank"}])
    assert localizer.dup_groups == {}
    assert localizer.languages == set()


# ----- per-slide scoring ---------------------------------------------------
def test_progress_tracks_how_far_into_the_slide_the_singer_is():
    entry = fa.SlideEntry(0, "V1", "", "was blind but now i see", False)
    early = fa.score_slide(entry, "was blind but")
    late = fa.score_slide(entry, "but now i see")
    assert early.progress < 0.7 < late.progress


def test_short_slide_loses_to_the_slide_that_explains_the_whole_line():
    """A two-word slide inside a long transcript scores 100 on raw partial_ratio.
    The coverage penalty is what stops it out-ranking the real line."""
    short = fa.SlideEntry(0, "G", "", "set free", False)
    full = fa.SlideEntry(1, "G", "", "my chains are gone ive been set free", False)
    heard = "my chains are gone ive been set free"
    assert fa.score_slide(short, heard).raw_score == pytest.approx(100.0)
    assert fa.score_slide(full, heard).score > fa.score_slide(short, heard).score


def test_slide_matching_only_the_start_of_the_transcript_is_penalised():
    """The singer has already sung past a line that matches the *beginning* of
    the buffer — we want where they are now."""
    past = fa.SlideEntry(0, "G", "", "amazing grace", False)
    now = fa.SlideEntry(1, "G", "", "how sweet the sound", False)
    heard = "amazing grace how sweet the sound"
    assert fa.score_slide(now, heard).score > fa.score_slide(past, heard).score


def test_transcript_too_short_to_be_evidence_is_not_scored():
    entry = fa.SlideEntry(0, "G", "", "was blind but now i see", False)
    assert fa.score_slide(entry, "was") is None
    cn = fa.SlideEntry(0, "G", "", "我心讚美主的恩典", True)
    assert fa.score_slide(cn, "我心") is None


# ----- ranking -------------------------------------------------------------
def test_ranks_the_slide_actually_being_sung_first(loc):
    top = loc.rank("", "i once was lost but now am found", 0)[0]
    assert top.index == 2


def test_chinese_line_ranks_against_chinese_slides(loc):
    top = loc.rank("我心讚美主的恩典", "", 8)[0]
    assert top.index == 9


# ----- confidence gating ---------------------------------------------------
def test_needs_consecutive_votes_before_touching_pp7(loc):
    first = feed(loc, en="i once was lost but now am found", current=0)
    assert first.confident and first.action == "none"
    second = feed(loc, en="i once was lost but now am found", current=0, t0=1000.5)
    assert second.action == "jump" and second.target_index == 2


def test_flapping_recogniser_never_reaches_quorum(loc):
    """A, B, A, B must not act. A plain count over the window would let A win."""
    reads = ["i once was lost but now am found", "amazing grace how sweet the sound"] * 3
    for i, text in enumerate(reads):
        decision = loc.observe("", text, 0, now=1000.0 + i * 0.5)
        assert decision.action == "none", f"acted on cycle {i}"


def test_instrumental_vocalising_is_never_confident(loc):
    decision = feed(loc, en="la la la ooh ooh oh", current=0, times=4)
    assert not decision.confident
    assert decision.action == "none"


def test_ambiguous_slide_boundary_refuses_to_guess(loc):
    """Mid-transition the buffer holds the tail of one slide and the head of the
    next. Neither wins by the margin, so the localizer holds still."""
    decision = feed(loc, en="now am found was blind but", current=2, times=3)
    assert not decision.confident
    assert "margin" in decision.reason


def test_cooldown_blocks_action_right_after_a_move(loc):
    feed(loc, en="i once was lost but now am found", current=0)
    loc.cooldown_until = 1010.0
    decision = feed(loc, en="i once was lost but now am found", current=0, t0=1000.5)
    assert decision.action == "none"
    assert decision.reason == "cooldown"


def test_goes_lost_after_a_run_of_unconfident_reads(loc):
    assert not loc.is_lost
    feed(loc, en="la la la ooh ooh oh", current=0, times=fa.LOST_CYCLES)
    assert loc.is_lost


# ----- what it decides to do ----------------------------------------------
def test_recognises_a_section_the_operator_missed(loc):
    decision = feed(loc, en="my chains are gone ive been set free", current=1, times=2)
    assert decision.action == "jump"
    assert decision.target_index == 4


def test_rescues_an_advance_the_tail_match_missed(loc):
    decision = feed(loc, en="was blind but now i see", current=2, times=2)
    assert decision.action == "next"
    assert "missed advance" in decision.reason


def test_leaves_the_normal_end_of_line_advance_to_the_tail_match(loc):
    """Barely into the next slide is the tail matcher's job — the localizer
    stepping in here would fire ahead of the tuned pre-trigger delay."""
    long_line = "Was blind but now I see the light of His amazing grace"
    norm, is_cn = fa.normalize_slide_text(long_line)
    entry = fa.SlideEntry(index=1, group="V2", raw=long_line, norm=norm, is_chinese=is_cn)
    assert fa.score_slide(entry, "was blind but").progress < fa.NEXT_MIN_PROGRESS

    localizer = fa.SongLocalizer()
    localizer.build([
        {"text": "I once was lost but now am found", "group": "V2"},
        {"text": long_line, "group": "V2"},
    ])
    decision = feed(localizer, en="was blind but", current=0, times=3)
    assert decision.confident
    assert decision.action == "none"
    assert "leaving it to tail match" in decision.reason


def test_staying_put_produces_no_action(loc):
    decision = feed(loc, en="i once was lost but now am found", current=2, times=3)
    assert decision.confident
    assert decision.action == "none"
    assert decision.reason == "on the right slide"


# ----- duplicate resolution -----------------------------------------------
def test_identical_chorus_resolves_to_the_nearest_copy_forward(loc):
    decision = feed(loc, en="my chains are gone ive been set free", current=5, times=2)
    assert decision.target_index == 6


def test_identical_chorus_resolves_backward_when_the_band_repeats(loc):
    decision = feed(loc, en="my chains are gone ive been set free", current=7, times=2)
    assert decision.target_index == 6
    assert "English Chorus 2" in decision.reason


def test_identical_copies_do_not_count_as_rivals_for_the_margin(loc):
    """Slides 4 and 6 are the same lyric. If 6 counted as a rival to 4 the margin
    would be ~0 and the localizer could never act on a repeated chorus at all."""
    decision = feed(loc, en="my chains are gone ive been set free", current=5)
    assert decision.confident
    assert decision.margin > fa.LOCATE_MARGIN


# ----- lead time vs the operator ------------------------------------------
@pytest.fixture(autouse=True)
def _clear_lead_times():
    fa.perf_metrics['lead_times'].clear()
    yield
    fa.perf_metrics['lead_times'].clear()


def test_records_how_far_ahead_of_the_operator_it_was(loc):
    feed(loc, en="my chains are gone ive been set free", current=1, t0=1000.0)
    lead = loc.note_movement(4, now=1003.0, source="operator")
    assert lead == pytest.approx(3.0)
    assert fa.perf_metrics['lead_times'] == [pytest.approx(3.0)]


def test_the_scripts_own_jumps_are_not_counted_as_lead(loc):
    feed(loc, en="my chains are gone ive been set free", current=1, t0=1000.0)
    assert loc.note_movement(4, now=1003.0, source="script") is None
    assert fa.perf_metrics['lead_times'] == []


def test_a_stale_sighting_is_not_a_lead(loc):
    feed(loc, en="my chains are gone ive been set free", current=1, t0=1000.0)
    lead = loc.note_movement(4, now=1000.0 + fa.MAX_LEAD_SEC + 60, source="operator")
    assert lead is None
    assert fa.perf_metrics['lead_times'] == []


def test_movement_starts_a_cooldown_and_clears_votes(loc):
    feed(loc, en="my chains are gone ive been set free", current=1, t0=1000.0)
    assert loc.votes
    loc.note_movement(4, now=1003.0, source="operator")
    assert not loc.votes
    assert loc.cooldown_until == pytest.approx(1003.0 + fa.LOCATE_COOLDOWN)
