"""Unabhängiges Orakel auf zufälligen kleinen Instanzen (auch mit Gleichständen):
zweitbestes Verfahren gegen eine zweite LP-Formulierung (Zahlungen je Typenpaar, GLOP aus OR-Tools statt HiGHS),
VCG/AGV direkt aus der Definition, bester Festpreis per feiner Preissuche, Mittelpunkt-Gebot per Schleife."""

import numpy as np
import pytest

import ms_mechanisms as M

pywraplp = pytest.importorskip("ortools.linear_solver.pywraplp")


def random_instance(seed):
    rng = np.random.default_rng(seed)
    K, L = int(rng.integers(2, 6)), int(rng.integers(2, 6))
    if seed % 3 == 1:                                    # ganzzahlig: Gleichstände zwischen Werten und Kosten
        v, c = np.unique(rng.integers(1, 8, K).astype(float)), np.unique(rng.integers(0, 7, L).astype(float))
    else:
        v, c = np.sort(rng.uniform(0, 100, K)), np.sort(rng.uniform(0, 100, L))
    return v, rng.dirichlet(np.ones(len(v))), c, rng.dirichlet(np.ones(len(c)))


def dense_second_best_gain(v, p, c, q):
    K, L = len(v), len(c)
    s = pywraplp.Solver.CreateSolver("GLOP")
    inf = s.infinity()
    x = [[s.NumVar(0, 1, "") for _ in range(L)] for _ in range(K)]
    tb = [[s.NumVar(-inf, inf, "") for _ in range(L)] for _ in range(K)]
    ts = [[s.NumVar(-inf, inf, "") for _ in range(L)] for _ in range(K)]

    def buyer_u(i, k):
        return sum(q[j] * (v[i] * x[k][j] - tb[k][j]) for j in range(L))

    def seller_u(j, m):
        return sum(p[i] * (ts[i][m] - c[j] * x[i][m]) for i in range(K))
    for i in range(K):
        s.Add(buyer_u(i, i) >= 0)
        for k in range(K):
            if k != i:
                s.Add(buyer_u(i, i) >= buyer_u(i, k))
    for j in range(L):
        s.Add(seller_u(j, j) >= 0)
        for m in range(L):
            if m != j:
                s.Add(seller_u(j, j) >= seller_u(j, m))
    s.Add(sum(p[i] * q[j] * (tb[i][j] - ts[i][j]) for i in range(K) for j in range(L)) >= 0)
    s.Maximize(sum(p[i] * q[j] * x[i][j] * (v[i] - c[j]) for i in range(K) for j in range(L)))
    assert s.Solve() == pywraplp.Solver.OPTIMAL
    return s.Objective().Value()


@pytest.mark.parametrize("seed", range(30))
def test_second_best_value_agrees_with_an_independent_lp_and_the_solution_is_feasible(seed):
    v, p, c, q = random_instance(seed)
    K, L = len(v), len(c)
    sb = M.second_best(v, p, c, q)
    assert M.gains_of(sb.x, v, p, c, q) == pytest.approx(dense_second_best_gain(v, p, c, q), abs=1e-6)
    for i in range(K):
        u = [v[i] * sum(q[j] * sb.x[k, j] for j in range(L)) - sb.TB[k] for k in range(K)]
        assert u[i] >= -1e-6 and max(u) <= u[i] + 1e-6
    for j in range(L):
        u = [sb.TS[m] - c[j] * sum(p[i] * sb.x[i, m] for i in range(K)) for m in range(L)]
        assert u[j] >= -1e-6 and max(u) <= u[j] + 1e-6
    assert sum(p[i] * sb.TB[i] for i in range(K)) - sum(q[j] * sb.TS[j] for j in range(L)) >= -1e-6


@pytest.mark.parametrize("seed", range(40))
def test_vcg_agv_and_posted_price_from_their_definitions(seed):
    v, p, c, q = random_instance(100 + seed)
    K, L = len(v), len(c)
    trade = np.array([[1.0 if v[i] > c[j] else 0.0 for j in range(L)] for i in range(K)])
    vc, ag = M.vcg(v, p, c, q), M.agv(v, p, c, q)
    assert np.array_equal(vc.x, trade) and np.array_equal(ag.x, trade)
    a = np.array([sum(q[j] * c[j] * trade[i, j] for j in range(L)) for i in range(K)])      # Externalität des Käufers
    b = np.array([sum(p[i] * v[i] * trade[i, j] for i in range(K)) for j in range(L)])      # Externalität des Verkäufers
    assert vc.TB == pytest.approx(a) and vc.TS == pytest.approx(b)
    assert ag.TB == pytest.approx([a[i] + sum(q[j] * b[j] for j in range(L)) for i in range(K)])
    assert ag.TS == pytest.approx([b[j] + sum(p[i] * a[i] for i in range(K)) for j in range(L)])
    assert vc.budget(p, q) == pytest.approx(-M.first_best_gains(v, p, c, q))
    assert abs(ag.budget(p, q)) < 1e-9
    assert vc.buyer_gain_from_lying(v, q).max() < 1e-9 and ag.buyer_gain_from_lying(v, q).max() < 1e-9      # ehrlich
    pts = sorted(set(v) | set(c))
    fine = np.concatenate([np.linspace(pts[0] - 1, pts[-1] + 1, 300), pts, [x + 1e-6 for x in pts], [x - 1e-6 for x in pts]])
    best = max(sum(p[i] * q[j] * (v[i] - c[j]) for i in range(K) for j in range(L) if v[i] >= pr and c[j] <= pr) for pr in fine)
    price, pm = M.best_posted_price(v, p, c, q)
    assert M.gains_of(pm.x, v, p, c, q) == pytest.approx(best, abs=1e-9)
    assert pm.buyer_gain_from_lying(v, q).max() < 1e-9 and pm.seller_gain_from_lying(c, p).max() < 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_midpoint_best_bid_by_loop(seed):
    v, p, c, q = random_instance(200 + seed)
    grid = np.linspace(min(v.min(), c.min()), max(v.max(), c.max()), 200)
    for vi in v:
        us = [sum(q[j] * (vi - (bd + c[j]) / 2) for j in range(len(c)) if c[j] <= bd + 1e-9) for bd in grid]
        assert M.midpoint_best_bid(vi, c, q, grid)[1] == pytest.approx(max(us), abs=1e-9)
