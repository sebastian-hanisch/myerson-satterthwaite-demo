"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Lügen-Widget, Permalink-Grenzen, Extremwerte, drei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ms_constants as C
import ms_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_shows_the_conflict_and_five_mechanisms():
    at = _run()
    _ok(at)
    assert at.metric and any("Kein Mechanismus erfüllt alle vier Wünsche" in w.value for w in at.warning)
    assert any("Verletzt wird je Verfahren" in i.value for i in at.info)


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]
    assert at.get("plotly_chart")


def test_disjoint_supports_show_the_no_conflict_message():
    at = _run()
    next(b for b in at.button if b.key == "preset_Getrennte Bereiche").click().run()
    _ok(at)
    assert any("Hier gibt es keinen Konflikt" in s.value for s in at.success)


def test_lying_section_honest_is_optimal_for_second_best_and_lying_pays_for_the_midpoint_price():
    at = _run(mech_select="second_best")
    _ok(at)
    assert any("Ehrlich melden ist optimal" in s.value for s in at.success)
    at = _run(mech_select="midpoint")
    _ok(at)
    assert any("Lügen lohnt sich" in w.value and "seines Werts" in w.value for w in at.warning)


@pytest.mark.parametrize("mech", C.MECHANISMS)
def test_every_mechanism_view_runs(mech):
    _ok(_run(mech_select=mech))


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["buyer"] = "37-183"
    at.query_params["seller"] = "50-50"
    at.query_params["k"] = "9999"
    at.query_params["family"] = "bell"
    at.query_params["mech"] = "nonsense"
    at.run()
    _ok(at)
    assert at.session_state["buyer_range"] == (35, 185) and at.session_state["seller_range"] == (50, 55) and at.session_state["k_slider"] == C.K_MAX
    assert at.session_state["family_select"] == "bell" and at.session_state["mech_select"] == "second_best"


@pytest.mark.parametrize("kw", [dict(k_slider=C.K_MIN), dict(k_slider=C.K_MAX), dict(family_select="bell"), dict(buyer_range=(0, 200), seller_range=(0, 200)), dict(buyer_range=(0, 50), seller_range=(150, 200)),
                                dict(buyer_range=(100, 105), seller_range=(100, 105), k_slider=6), dict(buyer_range=(50, 50))])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_overlap_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "OVERLAP_SHIFTS", (0, 50, 100, 100))
    monkeypatch.setattr(C, "OVERLAP_K", 6)
    at = _run()
    next(b for b in at.button if b.key == "overlap_start").click().run()
    _ok(at)
    assert at.session_state["overlap_on"] and any("Bei voller Überlappung erreicht das zweitbeste Verfahren" in w.value for w in at.warning)


def test_grid_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "GRID_KS", (4, 8, 12))
    at = _run()
    next(b for b in at.button if b.key == "grid_start").click().run()
    _ok(at)
    assert at.session_state["grid_on"] and any("Bei nur 4 Typen je Seite" in w.value for w in at.warning)


def test_family_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "COMPARE_K", 8)
    at = _run()
    next(b for b in at.button if b.key == "family_start").click().run()
    _ok(at)
    assert at.session_state["family_on"] and any("Auch bei glockenförmigen Typen" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present_and_no_unresolved_f_strings():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value and "{yn(" not in el.value


def test_midpoint_lying_numbers_in_the_widget_match_the_preset_help():
    at = _run()
    next(b for b in at.button if b.key == "preset_Lügen beim Mittelpunkt-Preis").click().run()
    _ok(at)
    assert any("meldet der Käufer 34,4 € statt seines wahren Werts 53,1 €" in w.value and "Gewinn 2,93 €" in w.value for w in at.warning)
