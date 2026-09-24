"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Klemmen und Einrasten der Bereiche - reine Datenprüfungen ohne Streamlit-Session."""

import pytest

import ms_constants as C
import ms_evaluation as E
import ms_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for p in P.PRESETS.values():
        assert p["family"] in C.FAMILIES and p["mech"] in C.MECHANISMS and C.K_MIN <= p["k"] <= C.K_MAX
        for key in ("buyer", "seller"):
            lo, hi = p[key]
            assert C.RANGE_MIN <= lo < hi <= C.RANGE_MAX and lo % C.RANGE_STEP == 0 and hi % C.RANGE_STEP == 0
            assert P.clamp_range(lo, hi) == (lo, hi)


def test_default_preset_equals_the_default_settings():
    p = P.PRESETS["Klassiker: gleiche Bereiche"]
    assert E.Settings(p["family"], p["buyer"], p["seller"], p["k"]) == E.Settings()
    assert p["mech"] == P.SETTING_SPECS["mech_select"].default


@pytest.mark.parametrize("lo,hi,expected", [(37, 183, (35, 185)), (-20, 500, (0, 200)), (50, 50, (50, 55)), (200, 200, (195, 200)), (90, 30, (90, 95)), (0, 0, (0, 5))])
def test_clamp_range_snaps_clamps_and_keeps_a_minimum_width(lo, hi, expected):
    assert P.clamp_range(lo, hi) == expected
    a, b = P.clamp_range(lo, hi)
    assert C.RANGE_MIN <= a < b <= C.RANGE_MAX


def test_range_caster_parses_and_rejects_garbage():
    assert P._range("40-100") == (40, 100) and P._range("41.9-99") == (40, 100)
    for bad in ("abc", "1", "nan-5", "5-inf"):
        with pytest.raises(ValueError):
            P._range(bad)


def test_url_params_are_unique():
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_k_caster_clamps():
    cast = P.SETTING_SPECS["k_slider"].caster
    assert cast("9999") == C.K_MAX and cast("-3") == C.K_MIN and cast("12") == 12
