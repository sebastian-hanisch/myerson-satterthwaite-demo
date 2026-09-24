"""Zweiseitiger Handel eines Slots: Käufer mit privatem Wert v, Verkäufer mit privaten Kosten c, beide unabhängig verteilt.

Typen sind auf einem Gitter diskretisiert (K Werte, L Kosten mit Wahrscheinlichkeiten p, q). Ein direkter Mechanismus besteht aus der Handelswahrscheinlichkeit x[i, j] für (Wert v_i, Kosten c_j) und den erwarteten (interimistischen)
Zahlungen TB[i] (Käufer zahlt) und TS[j] (Verkäufer erhält). Eigenschaften (alle exakt geprüft):
- ehrlich (Bayes-Nash, BIC): kein Typ verbessert seinen erwarteten Nutzen durch eine falsche Meldung;
- freiwillig (interim IR): jeder Typ hat erwarteten Nutzen >= 0;
- ausgeglichen (BB): erwartete Zahlungen des Käufers >= erwartete Zahlungen an den Verkäufer;
- effizient: es wird genau gehandelt, wenn v > c."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import ms_constants as C


def cell_probs(family, k):
    """Wahrscheinlichkeiten der k gleich breiten Zellen auf [0, 1]."""
    edges = np.linspace(0.0, 1.0, k + 1)
    if family == "uniform":
        return np.full(k, 1.0 / k)
    if family == "bell":
        F = 3 * edges ** 2 - 2 * edges ** 3                      # Verteilungsfunktion von Beta(2,2)
        return np.diff(F)
    raise ValueError(family)


def make_types(lo, hi, k, family):
    """Zellmitten in [lo, hi] und ihre Wahrscheinlichkeiten."""
    mids = (np.arange(k) + 0.5) / k
    return lo + mids * (hi - lo), cell_probs(family, k)


def efficient_trade(v, c):
    return (v[:, None] > c[None, :]).astype(float)


def first_best_gains(v, p, c, q):
    return float(np.einsum("i,j,ij->", p, q, np.maximum(v[:, None] - c[None, :], 0.0)))


def gains_of(x, v, p, c, q):
    return float(np.einsum("i,j,ij->", p, q, x * (v[:, None] - c[None, :])))


@dataclass
class Mechanism:
    name: str
    x: np.ndarray          # Handelswahrscheinlichkeit (K x L)
    TB: np.ndarray         # erwartete Zahlung des Käufers je Wertetyp
    TS: np.ndarray         # erwartete Zahlung an den Verkäufer je Kostentyp

    def buyer_util(self, v, q):
        return v * (self.x @ q) - self.TB

    def seller_util(self, c, p):
        return self.TS - c * (p @ self.x)

    def buyer_gain_from_lying(self, v, q):
        """Für jeden Wertetyp: bester Nutzen einer falschen Meldung minus ehrlicher Nutzen (>= 0 heißt: Lügen lohnt)."""
        XB = self.x @ q
        util = np.outer(v, XB) - self.TB[None, :]                 # util[i, k] = Nutzen von Typ i bei Meldung k
        return util.max(axis=1) - np.diag(util)

    def seller_gain_from_lying(self, c, p):
        XS = p @ self.x
        util = self.TS[None, :] - np.outer(c, XS)                 # util[j, m]
        return util.max(axis=1) - np.diag(util)

    def budget(self, p, q):
        return float(p @ self.TB - q @ self.TS)


def vcg(v, p, c, q):
    """Effizienter Handel; der Käufer zahlt die Kosten des Verkäufers, der Verkäufer erhält den Wert des Käufers (Clarke-Pivot): dominant ehrlich und freiwillig, aber nie ausgeglichen (Defizit)."""
    x = efficient_trade(v, c)
    TB = x @ (q * c)
    TS = (p * v) @ x
    return Mechanism("vcg", x, TB, TS)


def agv(v, p, c, q):
    """d'Aspremont/Gérard-Varet: effizienter Handel mit Erwartungs-Externalitäts-Zahlungen, die sich genau ausgleichen; ehrlich im Bayes-Nash-Sinn, aber nicht freiwillig."""
    x = efficient_trade(v, c)
    A = x @ (q * c)                                              # A_i = E_c[c * 1{trade}] : Kosten, die der Verkäufer beim Wert v_i im Mittel trägt
    B = (p * v) @ x                                              # B_j = E_v[v * 1{trade}]
    TB = A + q @ B
    TS = B + p @ A
    return Mechanism("agv", x, TB, TS)


def posted_price(v, p, c, q, price):
    """Handel zum Festpreis: der Käufer nimmt an, wenn v >= Preis, der Verkäufer, wenn c <= Preis; dominant ehrlich, freiwillig, ausgeglichen - aber nicht effizient."""
    x = (v[:, None] >= price - C.EPS).astype(float) * (c[None, :] <= price + C.EPS).astype(float)
    return Mechanism("posted", x, price * (x @ q), price * (p @ x))


def best_posted_price(v, p, c, q):
    """Bester Festpreis unter allen Typwerten (der Erwartungsgewinn ist stückweise linear im Preis, das Optimum liegt auf einem Gitterwert)."""
    cands = np.unique(np.concatenate([v, c]))
    gains = [gains_of(posted_price(v, p, c, q, pr).x, v, p, c, q) for pr in cands]
    k = int(np.argmax(gains))
    return float(cands[k]), posted_price(v, p, c, q, float(cands[k]))


def second_best(v, p, c, q):
    """Das beste erreichbare Verfahren, das ehrlich, freiwillig und im Erwartungswert ausgeglichen ist: lineares Programm über Handelswahrscheinlichkeiten x[i, j] und interimistische Zahlungen TB[i], TS[j]
    (Myerson/Satterthwaite 1983). Zielfunktion: erwartete Gewinne aus Handel."""
    K, L = len(v), len(c)
    nx = K * L
    nv = nx + K + L
    rows, cols, vals = [], [], []
    state = {"r": 0}
    rhs_len = []

    def block(n_rows, entries):
        """entries: Liste (Zeilenindex-Array relativ, Spaltenindex-Array, Wert-Array)."""
        base = state["r"]
        for rel, col, val in entries:
            rows.append(np.asarray(rel) + base)
            cols.append(np.asarray(col))
            vals.append(np.asarray(val, dtype=float))
        state["r"] += n_rows
        rhs_len.append(n_rows)

    def xi(i, j):
        return np.asarray(i) * L + np.asarray(j)

    # Käufer-IC (i meldet k statt i): v_i XB(k) - TB_k - v_i XB(i) + TB_i <= 0
    ii, kk = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    m = ii != kk
    ii, kk = ii[m], kk[m]
    nb = len(ii)
    rel = np.repeat(np.arange(nb), L)
    jj = np.tile(np.arange(L), nb)
    block(nb, [(rel, xi(np.repeat(kk, L), jj), np.repeat(v[ii], L) * q[jj]),
               (rel, xi(np.repeat(ii, L), jj), -np.repeat(v[ii], L) * q[jj]),
               (np.arange(nb), nx + kk, -np.ones(nb)),
               (np.arange(nb), nx + ii, np.ones(nb))])
    # Käufer-IR: -v_i XB(i) + TB_i <= 0
    rel = np.repeat(np.arange(K), L)
    jj = np.tile(np.arange(L), K)
    block(K, [(rel, xi(rel, jj), -v[rel] * q[jj]), (np.arange(K), nx + np.arange(K), np.ones(K))])
    # Verkäufer-IC (j meldet m statt j): TS_m - c_j XS(m) - TS_j + c_j XS(j) <= 0
    jj_, mm = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    m = jj_ != mm
    jj_, mm = jj_[m], mm[m]
    ns = len(jj_)
    rel = np.repeat(np.arange(ns), K)
    ik = np.tile(np.arange(K), ns)
    block(ns, [(rel, xi(ik, np.repeat(mm, K)), -c[np.repeat(jj_, K)] * p[ik]),
               (rel, xi(ik, np.repeat(jj_, K)), c[np.repeat(jj_, K)] * p[ik]),
               (np.arange(ns), nx + K + mm, np.ones(ns)),
               (np.arange(ns), nx + K + jj_, -np.ones(ns))])
    # Verkäufer-IR: -TS_j + c_j XS(j) <= 0
    rel = np.repeat(np.arange(L), K)
    ik = np.tile(np.arange(K), L)
    block(L, [(rel, xi(ik, rel), c[rel] * p[ik]), (np.arange(L), nx + K + np.arange(L), -np.ones(L))])
    # Budget (im Erwartungswert): -sum p_i TB_i + sum q_j TS_j <= 0
    block(1, [(np.zeros(K, dtype=int), nx + np.arange(K), -p), (np.zeros(L, dtype=int), nx + K + np.arange(L), q)])
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(state["r"], nv)).tocsr()
    cost = np.zeros(nv)
    cost[:nx] = -(np.outer(p, q) * (v[:, None] - c[None, :])).ravel()
    bounds = [(0, 1)] * nx + [(None, None)] * (K + L)
    res = linprog(cost, A_ub=A, b_ub=np.zeros(state["r"]), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return Mechanism("second_best", res.x[:nx].reshape(K, L), res.x[nx:nx + K], res.x[nx + K:])


# --- Mittelpunkt-Preis (nicht ehrlich) -----------------------------------------------------------------------------------------------------------


def midpoint_utility_curve(value, bids, c, q):
    """Nutzen eines Käufers mit Wert `value`, wenn er `bids` bietet und der Verkäufer ehrlich seine Kosten nennt: gehandelt wird zum Mittelpunkt der Gebote, wenn Gebot >= Kosten."""
    bids = np.asarray(bids, dtype=float)
    ok = (c[None, :] <= bids[:, None] + C.EPS)
    return (ok * (value - (bids[:, None] + c[None, :]) / 2) * q[None, :]).sum(axis=1)


def midpoint_best_bid(value, c, q, grid):
    u = midpoint_utility_curve(value, grid, c, q)
    k = int(np.argmax(u))
    return float(grid[k]), float(u[k])


def midpoint_shading_gains(v, p, c, q, grid):
    """Erwartete Gewinne aus Handel, wenn jeder Käufer sein bestes Gebot gegen einen ehrlichen Verkäufer abgibt (kein Gleichgewicht: der Verkäufer würde ebenfalls abweichen)."""
    total = 0.0
    for i, vi in enumerate(v):
        b, _ = midpoint_best_bid(vi, c, q, grid)
        total += p[i] * float(np.sum(q * (c <= b + C.EPS) * (vi - c)))
    return total
