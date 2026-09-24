"""Analyse einer Einstellung und die drei Experimente (Form, Invarianten, Reproduzierbarkeit)."""

import numpy as np
import pytest

import ms_constants as C
import ms_evaluation as E


def test_analyse_is_consistent_and_flags_match_the_theory():
    a = E.analyse(E.Settings())
    assert a.first_best == pytest.approx(E.M.first_best_gains(a.v, a.p, a.c, a.q))
    assert a.flags("vcg") == (True, True, True, False)                    # gibt den Ausgleich auf
    assert a.flags("agv") == (True, True, False, True)                    # gibt die Freiwilligkeit auf
    assert a.flags("second_best") == (False, True, True, True) and a.flags("posted") == (False, True, True, True)      # geben die Effizienz auf
    assert a.flags("midpoint") == (True, False, True, True)               # gibt die Ehrlichkeit auf
    assert a.share("vcg") == pytest.approx(1.0) and a.share("second_best") < 1 and a.share("posted") < a.share("second_best")
    assert a.budget("vcg") == pytest.approx(-a.first_best) and a.budget("second_best") == pytest.approx(0.0, abs=1e-6)
    assert a.min_utility("agv") < 0 <= a.min_utility("second_best") + 1e-7


def test_no_conflict_when_supports_are_disjoint():
    a = E.analyse(E.Settings("uniform", (100, 200), (0, 100), 10))
    assert a.flags("second_best") == (True, True, True, True) and a.flags("posted") == (True, True, True, True)
    assert a.share("second_best") == pytest.approx(1.0)


def test_midpoint_best_bid_is_below_the_value_and_lying_pays():
    a = E.analyse(E.Settings())
    assert (a.midpoint_bids <= a.v + 1e-9).all() and a.midpoint_lie_gain.max() > 0
    assert a.midpoint_lie_gain[np.argmax(a.v)] > a.midpoint_lie_gain[0]                      # hohe Werte haben mehr zu gewinnen


def test_settings_cache_and_extremes_run():
    for st in (E.Settings("bell", (0, 5), (195, 200), C.K_MIN), E.Settings("uniform", (0, 200), (0, 200), C.K_MAX), E.Settings("uniform", (150, 200), (0, 50), 8)):
        a = E.analyse(st)
        assert a.share("second_best") <= 1 + 1e-9 and a.first_best >= 0


def test_no_trade_case_has_zero_gains_without_division_errors():
    a = E.analyse(E.Settings("uniform", (0, 50), (150, 200), 8))                                  # Kosten immer über dem Wert
    assert a.first_best == 0.0 and a.share("second_best") == 1.0 and a.share("posted") == 1.0


def test_overlap_experiment_shape_and_monotone_trend():
    rows = E.overlap_experiment(shifts=(0, 50, 100), k=10)
    assert [r["shift"] for r in rows] == [0, 50, 100]
    assert rows[0]["share_second"] < rows[1]["share_second"] < rows[2]["share_second"] == pytest.approx(1.0)
    assert all(r["share_posted"] <= r["share_second"] + 1e-9 and r["vcg_deficit"] == pytest.approx(r["first_best"]) and r["agv_min_utility"] < 0 for r in rows)


def test_grid_experiment_shape_and_monotone_trend():
    rows = E.grid_experiment(ks=(4, 8, 16))
    assert [r["k"] for r in rows] == [4, 8, 16]
    shares = [r["second_best"] / r["first_best"] for r in rows]
    assert shares[0] > shares[1] > shares[2] and shares[0] == pytest.approx(1.0)


def test_family_experiment_shape():
    rows = E.family_experiment(k=8)
    assert [r["family"] for r in rows] == list(C.FAMILIES)
    assert all(0 < r["share_posted"] < r["share_second"] < 1 and r["vcg_deficit_share"] == pytest.approx(1.0) and 0 < r["midpoint_best_bid_ratio"] < 1 for r in rows)
