"""Jede Zahl aus PRESET_HELP und README: Presets auf die gezeigte Rundung (deterministische Rechnung mit exakter LP-Lösung), Experimente mit Bändern (CI installiert die neueste scipy)."""

import numpy as np
import pytest

import ms_constants as C
import ms_evaluation as E
import ms_mechanisms as M
import ms_presets as P


def analyse_preset(name):
    p = P.PRESETS[name]
    return E.analyse(E.Settings(p["family"], p["buyer"], p["seller"], p["k"]))


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_preset_classic():
    a = analyse_preset("Klassiker: gleiche Bereiche")
    assert a.first_best == pytest.approx(16.6, abs=0.05) and a.gains("second_best") == pytest.approx(14.8, abs=0.05) and a.share("second_best") == pytest.approx(0.892, abs=0.0005)
    assert a.price == pytest.approx(46.9, abs=0.05) and a.gains("posted") == pytest.approx(13.2, abs=0.05) and a.share("posted") == pytest.approx(0.794, abs=0.0005)
    assert -a.budget("vcg") == pytest.approx(a.first_best) and -a.budget("vcg") == pytest.approx(16.6, abs=0.05)
    assert int((a.mechs["agv"].buyer_util(a.v, a.q) < -1e-7).sum()) == 13 and len(a.v) == 16


def test_preset_little_overlap():
    a = analyse_preset("Wenig Überlappung")
    assert a.first_best == pytest.approx(61.1, abs=0.05) and a.gains("second_best") == pytest.approx(58.5, abs=0.05) and a.share("second_best") == pytest.approx(0.958, abs=0.0005)
    assert a.price == pytest.approx(78.1, abs=0.05) and a.gains("posted") == pytest.approx(52.0, abs=0.05) and a.share("posted") == pytest.approx(0.851, abs=0.0005)


def test_preset_disjoint():
    a = analyse_preset("Getrennte Bereiche")
    assert a.share("second_best") == pytest.approx(1.0) and a.share("posted") == pytest.approx(1.0) and a.price == pytest.approx(96.9, abs=0.05)
    assert a.flags("second_best") == (True, True, True, True)


def test_preset_bell():
    a = analyse_preset("Glockenförmige Werte")
    assert a.first_best == pytest.approx(12.9, abs=0.05) and a.gains("second_best") == pytest.approx(11.7, abs=0.05) and a.share("second_best") == pytest.approx(0.909, abs=0.0005)
    assert a.gains("posted") == pytest.approx(10.2, abs=0.05) and a.share("posted") == pytest.approx(0.790, abs=0.0005)


def test_preset_coarse_grid():
    a = analyse_preset("Grobes Gitter (4 Typen)")
    assert a.share("second_best") == pytest.approx(1.0) and a.share("posted") == pytest.approx(0.90, abs=0.0005) and len(a.v) == 4


def test_preset_midpoint_lying():
    a = analyse_preset("Lügen beim Mittelpunkt-Preis")
    ratio = float(np.mean((a.midpoint_bids / a.v)[a.v > 0]))
    assert ratio == pytest.approx(0.62, abs=0.005)
    i = len(a.v) // 2
    assert a.v[i] == pytest.approx(53.1, abs=0.05) and a.midpoint_bids[i] == pytest.approx(34.4, abs=0.05) and a.midpoint_lie_gain[i] == pytest.approx(2.93, abs=0.005)
    assert a.lie_gain("second_best") <= 1e-7


def test_second_best_refuses_trade_where_the_gain_is_small():
    """Klassiker: das zweitbeste Verfahren verzichtet auf 33 % der Handelswahrscheinlichkeit, aber nur dort, wo der Gewinn im Mittel 11,7 € beträgt (gegen 35,4 € im Durchschnitt aller effizienten Handel)."""
    a = analyse_preset("Klassiker: gleiche Bereiche")
    x = a.mechs["second_best"].x
    d = a.v[:, None] - a.c[None, :]
    w = np.outer(a.p, a.q)
    eff = d > 0
    lost = (1 - x) * eff * w
    assert lost.sum() / (w * eff).sum() == pytest.approx(0.326, abs=0.005)
    assert (lost * d).sum() / lost.sum() == pytest.approx(11.7, abs=0.05) and (w * eff * d).sum() / (w * eff).sum() == pytest.approx(35.4, abs=0.05)
    assert np.all(np.diff(x @ a.q) >= -1e-8) and np.all(np.diff(a.p @ x) <= 1e-8)                # Handelswahrscheinlichkeit steigt im Wert, fällt in den Kosten
    assert (x * (~eff) * w).sum() < 1e-9                                                           # nie Handel bei Wert <= Kosten


# --- Experimente --------------------------------------------------------------------------------------------------------------------------------


def test_overlap_experiment_numbers():
    rows = E.overlap_experiment()
    assert [r["shift"] for r in rows] == [0, 20, 40, 60, 80, 100]
    assert rows[0]["share_second"] == pytest.approx(0.892, abs=0.005) and rows[0]["share_posted"] == pytest.approx(0.794, abs=0.005)            # gemessen 89,2 % und 79,4 %
    assert rows[3]["share_second"] == pytest.approx(0.958, abs=0.005) and rows[-1]["share_second"] == pytest.approx(1.0)
    shares = [r["share_second"] for r in rows]
    assert all(shares[i] <= shares[i + 1] + 1e-9 for i in range(1, len(shares) - 1)) and shares[-1] > shares[0] + 0.10        # zunächst kaum (89,2 -> 89,1 %), dann schnell
    assert rows[0]["vcg_deficit"] == pytest.approx(16.6, abs=0.05) and rows[-1]["vcg_deficit"] == pytest.approx(100.0, abs=0.05)
    assert all(r["vcg_deficit"] == pytest.approx(r["first_best"]) for r in rows)


def test_grid_experiment_numbers():
    rows = E.grid_experiment()
    shares = [r["second_best"] / r["first_best"] for r in rows]
    assert [r["k"] for r in rows] == [6, 10, 16, 24, 32, 40]
    assert shares == sorted(shares, reverse=True)                                                  # fällt monoton mit feinerem Gitter
    assert shares[0] == pytest.approx(0.952, abs=0.005) and shares[-1] == pytest.approx(0.864, abs=0.005)
    assert rows[-1]["first_best"] == pytest.approx(1 / 6, abs=3e-4) and rows[-1]["second_best"] == pytest.approx(0.1439, abs=0.0005) and all(r["second_best"] > 9 / 64 for r in rows)


def test_family_experiment_numbers():
    rows = {r["family"]: r for r in E.family_experiment()}
    assert rows["uniform"]["share_second"] == pytest.approx(0.892, abs=0.005) and rows["bell"]["share_second"] == pytest.approx(0.909, abs=0.005)
    assert rows["uniform"]["share_posted"] == pytest.approx(0.794, abs=0.005) and rows["bell"]["share_posted"] == pytest.approx(0.790, abs=0.005)
    assert rows["uniform"]["agv_share_types_below_zero"] == pytest.approx(0.8125) and rows["bell"]["agv_share_types_below_zero"] == pytest.approx(0.8125)
    assert rows["uniform"]["midpoint_best_bid_ratio"] == pytest.approx(0.62, abs=0.01) and rows["bell"]["midpoint_best_bid_ratio"] == pytest.approx(0.70, abs=0.01)


def test_readme_analytic_values():
    """9/64 : 1/6 = 84,375 % und die Handrechnungen zum Festpreis (1/8) und AGV (v^2/2 - 1/3)."""
    assert (9 / 64) / (1 / 6) == pytest.approx(0.84375)
    v, p = M.make_types(0, 1, 400, "uniform")
    price, m = M.best_posted_price(v, p, v, p)
    assert M.gains_of(m.x, v, p, v, p) == pytest.approx(0.125, abs=2e-3) and price == pytest.approx(0.5, abs=0.01)
