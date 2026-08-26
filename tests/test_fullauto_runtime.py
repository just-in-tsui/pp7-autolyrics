"""Tests for the full-auto runtime wiring: how a localizer decision turns into a
ProPresenter call, how operator toggles override it, and how the dashboard renders.

The dashboard test matters more than it looks: the main audio loop swallows
exceptions on purpose (see CLAUDE.md), so a rendering bug wouldn't crash the
script — it would silently freeze the operator's only display.
"""

import pytest
import responses
from rich.console import Console

from tests._fullauto_stubs import SONG, fa


@pytest.fixture
def app(tmp_path):
    """A full-auto runtime pinned to slide 1 of SONG, with globals restored after."""
    saved = {name: getattr(fa, name) for name in
             ("poller", "localizer", "auto_locate", "shadow_mode",
              "stop_at_section_end", "cued_slide_index", "advance_guard")}
    saved_metrics = dict(fa.perf_metrics)

    poller = fa.PP7SmartPoller()
    poller.slide_cache = list(SONG)
    poller.current_index = 1
    poller.current_full_text = SONG[1]["text"]
    poller.connected = True

    fa.poller = poller
    fa.localizer = fa.SongLocalizer()
    fa.localizer.build(SONG)
    fa.auto_locate = True
    fa.shadow_mode = False
    fa.stop_at_section_end = False
    fa.cued_slide_index = None
    fa.advance_guard = True
    fa.perf_metrics['lead_times'] = []

    yield fa

    for name, value in saved.items():
        setattr(fa, name, value)
    fa.perf_metrics.clear()
    fa.perf_metrics.update(saved_metrics)


def confident_jump(app, current=1):
    """Drive the localizer to a settled 'jump to slide 4' decision."""
    app.localizer.observe("", "my chains are gone ive been set free", current, now=2000.0)
    return app.localizer.observe("", "my chains are gone ive been set free", current, now=2000.5)


# ----- decision -> ProPresenter -------------------------------------------
def test_confident_localizer_jumps_to_the_section_being_sung(app):
    decision = confident_jump(app)
    assert decision.action == "jump" and decision.target_index == 4

    with responses.RequestsMock() as mock:
        mock.add(responses.GET,
                 "http://127.0.0.1:1025/v1/presentation/active/4/trigger", status=200)
        assert app.execute_locate_action(decision, 1) is True
    assert app.perf_metrics['auto_jumps'] == 1


def test_shadow_mode_logs_the_decision_but_never_calls_pp7(app):
    app.shadow_mode = True
    decision = confident_jump(app)

    # RequestsMock with no registered URLs fails the test on any outbound call.
    with responses.RequestsMock():
        assert app.execute_locate_action(decision, 1) is False
    assert app.perf_metrics['shadow_jumps'] == 1
    assert app.perf_metrics['auto_jumps'] == 0
    assert "shadow" in app.ui.last_action.lower()


def test_hold_at_section_end_downgrades_a_jump_to_a_cue(app):
    """The operator asked to stop at section ends, so full auto tells them where
    the band went instead of overriding them — which is still the whole point."""
    app.stop_at_section_end = True
    app.poller.current_index = 1          # last slide of English Verse 1
    decision = confident_jump(app)

    with responses.RequestsMock():
        assert app.execute_locate_action(decision, 1) is False
    assert app.cued_slide_index == 4
    assert "cued" in app.ui.last_action.lower()


def test_rescue_goes_through_the_normal_advance_path(app):
    """A missed-advance rescue must honour a cued section jump, so it routes
    through handle_lyric_trigger rather than calling /next itself."""
    app.poller.current_index = 2
    app.localizer.observe("", "was blind but now i see", 2, now=2000.0)
    decision = app.localizer.observe("", "was blind but now i see", 2, now=2000.5)
    assert decision.action == "next"

    with responses.RequestsMock() as mock:
        mock.add(responses.GET,
                 "http://127.0.0.1:1025/v1/presentation/active/next/trigger", status=200)
        assert app.execute_locate_action(decision, 2) is True
    assert app.perf_metrics['auto_rescues'] == 1
    assert app.perf_metrics['tail_advances'] == 0


def test_cued_jump_wins_over_a_rescue_at_a_section_boundary(app):
    app.poller.current_index = 3          # last slide of English Verse 2
    app.cued_slide_index = 8
    app.localizer.observe("", "my chains are gone ive been set free", 3, now=2000.0)
    decision = app.localizer.observe("", "my chains are gone ive been set free", 3, now=2000.5)
    assert decision.action == "next" and decision.target_index == 4

    with responses.RequestsMock() as mock:
        mock.add(responses.GET,
                 "http://127.0.0.1:1025/v1/presentation/active/8/trigger", status=200)
        app.execute_locate_action(decision, 3)
    assert app.cued_slide_index is None


# ----- engine scheduling ---------------------------------------------------
def test_single_language_deck_never_pays_for_a_second_engine(app):
    app.localizer.build([{"text": "Amazing grace", "group": "V1"}])
    assert app.plan_languages('en', app.PROBE_EVERY, lost=True) == ['en']


def test_all_chinese_deck_uses_the_yue_engine_on_an_untexted_slide(app):
    """A blank slide has no CJK, so slide_language() reports 'en'. Following that
    in an all-Chinese deck would decode with the wrong engine and never recover."""
    app.localizer.build([
        {"text": "祢的信實極其廣大", "group": "V1"},
        {"text": "", "group": "V1"},
    ])
    assert app.plan_languages('en', 1, lost=False) == ['cn']


def test_an_empty_deck_falls_back_to_the_current_slide_language(app):
    app.localizer.build([])
    assert app.plan_languages('cn', 1, lost=True) == ['cn']


def test_bilingual_deck_probes_the_other_engine_periodically(app):
    assert app.plan_languages('en', 1, lost=False) == ['en']
    assert app.plan_languages('en', app.PROBE_EVERY, lost=False) == ['en', 'cn']


def test_a_lost_localizer_listens_in_both_languages(app):
    assert sorted(app.plan_languages('en', 1, lost=True)) == ['cn', 'en']


def test_localizer_off_means_current_slide_language_only(app):
    app.auto_locate = False
    assert app.plan_languages('cn', app.PROBE_EVERY, lost=True) == ['cn']


# ----- movement attribution ------------------------------------------------
def test_operator_move_records_how_far_ahead_the_script_was(app):
    app.localizer.observe("", "my chains are gone ive been set free", 1, now=3000.0)
    app.poller.movements.append((4, 3002.5))
    app.drain_movements()
    assert app.perf_metrics['lead_times'] == [pytest.approx(2.5)]


def test_the_scripts_own_jump_is_not_counted_as_lead(app):
    app.localizer.observe("", "my chains are gone ive been set free", 1, now=3000.0)
    app.mark_self_action(4)
    app.poller.movements.append((4, app.time.time()))
    app.drain_movements()
    assert app.perf_metrics['lead_times'] == []


# ----- dashboard -----------------------------------------------------------
@pytest.mark.parametrize("auto,shadow,lost", [
    (True, False, False),   # full auto
    (True, True, False),    # shadow
    (True, False, True),    # searching
    (False, False, False),  # localizer off
])
def test_dashboard_renders_in_every_mode(app, auto, shadow, lost):
    app.auto_locate, app.shadow_mode = auto, shadow
    app.localizer.unconfident_streak = app.LOST_CYCLES if lost else 0
    app.ui.candidates = app.localizer.rank("", "my chains are gone ive been set free", 1)[:3]
    app.ui.progress = 0.62
    app.ui.score = 71
    app.ui.heard = "my chains are gone"

    console = Console(file=open("/dev/null", "w"), width=120)
    console.print(app.render_dashboard())   # raises on any markup/API error


def test_dashboard_renders_before_any_audio_has_arrived(app):
    app.ui.candidates = []
    app.ui.progress = None
    app.ui.score = None
    console = Console(file=open("/dev/null", "w"), width=120)
    console.print(app.render_dashboard())
