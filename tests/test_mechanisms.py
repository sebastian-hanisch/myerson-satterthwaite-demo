"""Mechanismen für den Handel eines Slots: Handrechnungen (2 x 2 Typen), Eigenschaften (ehrlich, freiwillig, ausgeglichen), zweitbestes Verfahren gegen eine unabhängige dichte LP-Formulierung,
Konvergenz zu den bekannten stetigen Werten 1/6 und 9/64."""

import numpy as np
import pytest
from scipy.optimize import linprog

import ms_mechanisms as M


def small_game():
    """Käufer-Werte {2, 4}, Verkäufer-Kosten {1, 3}, je mit Wahrscheinlichkeit 1/2."""
    return np.array([2.0, 4.0]), np.array([0.5, 0.5]), np.array([1.0, 3.0]), np.array([0.5, 0.5])


def dense_second_best(v, p, c, q):
    """Unabhängige Formulierung: Variablen x_ij sowie Zahlungen tb_ij und ts_ij je Typenpaar (nicht nur interimistisch); Bedingungen für die Erwartungswerte einzeln ausgeschrieben."""
    K, L = len(v), len(c)
    n = K * L
    ix = lambda i, j: i * L + j
    nv = 3 * n
    A, cost = [], np.zeros(nv)
    for i in range(K):
        for j in range(L):
            cost[ix(i, j)] = -p[i] * q[j] * (v[i] - c[j])
    def buyer_u(i, k):          # Nutzen von Wertetyp i bei Meldung k, als Zeilenvektor
        row = np.zeros(nv)
        for j in range(L):
            row[ix(k, j)] += q[j] * v[i]
            row[n + ix(k, j)] -= q[j]
        return row
    def seller_u(j, m):
        row = np.zeros(nv)
        for i in range(K):
            row[2 * n + ix(i, m)] += p[i]
            row[ix(i, m)] -= p[i] * c[j]
        return row
    for i in range(K):
        for k in range(K):
            if i != k:
                A.append(buyer_u(i, k) - buyer_u(i, i))
        A.append(-buyer_u(i, i))
    for j in range(L):
        for m in range(L):
            if j != m:
                A.append(seller_u(j, m) - seller_u(j, j))
        A.append(-seller_u(j, j))
    row = np.zeros(nv)
    for i in range(K):
        for j in range(L):
            row[n + ix(i, j)] = -p[i] * q[j]
            row[2 * n + ix(i, j)] = p[i] * q[j]
    A.append(row)
    bounds = [(0, 1)] * n + [(None, None)] * (2 * n)
    res = linprog(cost, A_ub=np.array(A), b_ub=np.zeros(len(A)), bounds=bounds, method="highs")
    return -res.fun


def test_vcg_by_hand():
    v, p, c, q = small_game()
    m = M.vcg(v, p, c, q)
    assert m.x.tolist() == [[1, 0], [1, 1]]                                                   # Handel bei (2,1), (4,1), (4,3)
    assert M.first_best_gains(v, p, c, q) == pytest.approx(1.25) and M.gains_of(m.x, v, p, c, q) == pytest.approx(1.25)
    assert m.TB == pytest.approx([0.5, 2.0]) and m.TS == pytest.approx([3.0, 2.0])
    assert m.budget(p, q) == pytest.approx(-1.25)                                              # Defizit = ganzer Gewinn
    assert m.buyer_util(v, q) == pytest.approx([2 * 0.5 - 0.5, 4 - 2.0]) and m.seller_util(c, p) == pytest.approx([3.0 - 1.0, 2.0 - 3 * 0.5])


def test_agv_by_hand():
    v, p, c, q = small_game()
    m = M.agv(v, p, c, q)
    assert m.TB == pytest.approx([3.0, 4.5]) and m.TS == pytest.approx([4.25, 3.25])            # A + E[B] bzw. B + E[A]
    assert m.budget(p, q) == pytest.approx(0.0)
    assert m.buyer_util(v, q) == pytest.approx([2 * 0.5 - 3.0, 4.0 - 4.5]) and (m.buyer_util(v, q) < 0).all()           # nicht freiwillig
    assert m.buyer_gain_from_lying(v, q).max() == pytest.approx(0.0, abs=1e-9) and m.seller_gain_from_lying(c, p).max() == pytest.approx(0.0, abs=1e-9)


def test_posted_price_by_hand():
    v, p, c, q = small_game()
    m = M.posted_price(v, p, c, q, 2.0)                                                       # Käufer 2 und 4 nehmen an, Verkäufer 1 (c <= 2), nicht 3
    assert m.x.tolist() == [[1, 0], [1, 0]] and M.gains_of(m.x, v, p, c, q) == pytest.approx(0.25 * (1 + 3))
    assert m.budget(p, q) == pytest.approx(0.0) and (m.buyer_util(v, q) >= 0).all() and (m.seller_util(c, p) >= 0).all()
    price, best = M.best_posted_price(v, p, c, q)
    assert M.gains_of(best.x, v, p, c, q) == pytest.approx(1.0)


def test_midpoint_best_bid_by_hand():
    """Verkäufer ehrlich mit Kosten {1, 3} (je 1/2), Käufer-Wert 4: Gebot 3 -> handelt mit beiden zum Mittelpunkt 2 bzw. 3: Nutzen 0,5 (4 - 2) + 0,5 (4 - 3) = 1,5;
    Gebot 4 -> 0,5 (4 - 2,5) + 0,5 (4 - 3,5) = 1,0; Gebot 1 -> nur Kosten 1: 0,5 (4 - 1) = 1,5; Gebot 2 -> 0,5 (4 - 1,5) = 1,25."""
    c, q = np.array([1.0, 3.0]), np.array([0.5, 0.5])
    u = M.midpoint_utility_curve(4.0, [1.0, 2.0, 3.0, 4.0], c, q)
    assert u == pytest.approx([1.5, 1.25, 1.5, 1.0])
    assert M.midpoint_best_bid(4.0, c, q, np.array([1.0, 2.0, 3.0, 4.0]))[1] == pytest.approx(1.5)             # ehrliches Gebot 4 ist schlechter als 3 oder 1


def test_vcg_and_posted_price_are_dominant_strategy_honest_for_every_partner_type():
    v, p = M.make_types(0, 100, 8, "uniform")
    c, q = M.make_types(20, 120, 8, "bell")
    for mech in (M.vcg(v, p, c, q), M.posted_price(v, p, c, q, 55.0)):
        pay = (mech.TB, mech.TS)
        # Käufer Typ i meldet k gegen jeden festen Verkäufertyp j: Nutzen v_i x[k,j] - Zahlung; Zahlung je Paar bei VCG: c_j x[k,j], beim Festpreis: Preis x[k,j]
        for j in range(len(c)):
            for i in range(len(v)):
                unit = c[j] if mech.name == "vcg" else 55.0
                util = [v[i] * mech.x[k, j] - unit * mech.x[k, j] for k in range(len(v))]
                assert max(util) <= util[i] + 1e-9
        assert mech.buyer_gain_from_lying(v, q).max() == pytest.approx(0.0, abs=1e-9)
        assert mech.seller_gain_from_lying(c, p).max() == pytest.approx(0.0, abs=1e-9)
        assert (mech.buyer_util(v, q) >= -1e-9).all() and (mech.seller_util(c, p) >= -1e-9).all()


@pytest.mark.parametrize("family,blo,bhi,slo,shi,k", [("uniform", 0, 1, 0, 1, 5), ("bell", 0, 100, 20, 120, 6), ("uniform", 30, 130, 0, 100, 4)])
def test_second_best_satisfies_all_constraints_and_matches_an_independent_lp(family, blo, bhi, slo, shi, k):
    v, p = M.make_types(blo, bhi, k, family)
    c, q = M.make_types(slo, shi, k, family)
    sb = M.second_best(v, p, c, q)
    assert sb.buyer_gain_from_lying(v, q).max() == pytest.approx(0.0, abs=1e-7) and sb.seller_gain_from_lying(c, p).max() == pytest.approx(0.0, abs=1e-7)
    assert sb.buyer_util(v, q).min() >= -1e-7 and sb.seller_util(c, p).min() >= -1e-7 and sb.budget(p, q) >= -1e-7
    assert (sb.x >= -1e-9).all() and (sb.x <= 1 + 1e-9).all()
    gains = M.gains_of(sb.x, v, p, c, q)
    assert gains == pytest.approx(dense_second_best(v, p, c, q), rel=1e-6, abs=1e-9)
    assert gains <= M.first_best_gains(v, p, c, q) + 1e-9
    _, posted = M.best_posted_price(v, p, c, q)
    assert gains >= M.gains_of(posted.x, v, p, c, q) - 1e-9                                   # das Festpreis-Verfahren ist zulässig, also nie besser


def test_second_best_never_trades_when_the_gain_is_negative():
    v, p = M.make_types(0, 100, 10, "uniform")
    sb = M.second_best(v, p, v, p)
    assert (sb.x[v[:, None] <= v[None, :]] < 1e-7).all()


def test_second_best_by_hand_when_supports_are_disjoint():
    """Alle Werte über allen Kosten: Handel immer effizient, festes Preisfenster genügt; das zweitbeste Verfahren erreicht den erstbesten Gewinn."""
    v, p = M.make_types(10, 20, 6, "uniform")
    c, q = M.make_types(0, 8, 6, "uniform")
    sb = M.second_best(v, p, c, q)
    assert M.gains_of(sb.x, v, p, c, q) == pytest.approx(M.first_best_gains(v, p, c, q)) and (sb.x > 1 - 1e-7).all()


def test_efficiency_is_lost_for_overlapping_types_and_the_loss_shrinks_with_less_overlap():
    shares = []
    for shift in (0, 0.4, 0.8, 1.0):
        v, p = M.make_types(shift, shift + 1, 20, "uniform")
        c, q = M.make_types(0, 1, 20, "uniform")
        sb = M.second_best(v, p, c, q)
        shares.append(M.gains_of(sb.x, v, p, c, q) / M.first_best_gains(v, p, c, q))
    assert shares[0] < 0.9 and shares[1] > shares[0] and shares[2] > shares[1] and shares[3] == pytest.approx(1.0)


def test_grid_converges_to_the_continuous_values_one_sixth_and_nine_sixtyfourths():
    """Erstbest -> 1/6 und zweitbest -> 9/64; mit Richardson-Extrapolation (Fehler ~ 1/k) aus k = 32 und 40."""
    vals = {}
    for k in (16, 32, 40):
        v, p = M.make_types(0, 1, k, "uniform")
        vals[k] = (M.first_best_gains(v, p, v, p), M.gains_of(M.second_best(v, p, v, p).x, v, p, v, p))
    assert abs(vals[40][0] - 1 / 6) < 2e-4 and vals[16][1] > vals[32][1] > vals[40][1] > 9 / 64
    extrap = (40 * vals[40][1] - 32 * vals[32][1]) / 8
    assert extrap == pytest.approx(9 / 64, abs=1.5e-3)


def test_vcg_deficit_equals_the_first_best_gains():
    for fam in ("uniform", "bell"):
        v, p = M.make_types(0, 100, 12, fam)
        c, q = M.make_types(10, 90, 12, fam)
        assert M.vcg(v, p, c, q).budget(p, q) == pytest.approx(-M.first_best_gains(v, p, c, q))


def test_agv_buyer_utility_matches_the_analytic_form_for_uniform_types():
    """Für gleichverteilte Typen auf [0,1]^2: erwarteter Nutzen des Käufers v^2/2 - 1/3 (negativ für v < 0,816)."""
    v, p = M.make_types(0, 1, 200, "uniform")
    u = M.agv(v, p, v, p).buyer_util(v, p)
    assert np.max(np.abs(u - (v ** 2 / 2 - 1 / 3))) < 3e-3 and u[v < 0.8].max() < 0 and u[v > 0.9].min() > -0.01


def test_posted_price_and_midpoint_shading_match_the_continuous_values():
    v, p = M.make_types(0, 1, 100, "uniform")
    price, m = M.best_posted_price(v, p, v, p)
    assert price == pytest.approx(0.5, abs=0.02) and M.gains_of(m.x, v, p, v, p) == pytest.approx(1 / 8, abs=3e-3)          # Festpreis 1/2, Gewinn 1/8
    grid = np.linspace(0, 1, 400)
    for value in (0.3, 0.6, 0.9):
        b, _ = M.midpoint_best_bid(value, v, p, grid)
        assert b == pytest.approx(2 * value / 3, abs=0.03)                                     # bestes Gebot gegen ehrlichen Verkäufer: 2v/3


def test_cell_probs_and_types():
    assert M.cell_probs("uniform", 8) == pytest.approx(np.full(8, 0.125))
    b = M.cell_probs("bell", 10)
    assert b.sum() == pytest.approx(1.0) and b[0] < b[4] and b == pytest.approx(b[::-1])                # symmetrisch, Mitte häufiger
    v, p = M.make_types(20, 120, 5, "uniform")
    assert v == pytest.approx([30, 50, 70, 90, 110]) and p.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        M.cell_probs("bogus", 3)
