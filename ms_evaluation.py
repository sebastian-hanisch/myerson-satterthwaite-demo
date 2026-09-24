"""Auswertung: fünf Mechanismen für den Handel eines Slots und drei Experimente (Überlappung der Bereiche, Feinheit des Gitters, Verteilungsfamilien)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import ms_constants as C
import ms_mechanisms as M


@dataclass(frozen=True)
class Settings:
    family: str = "uniform"
    buyer: tuple = C.DEFAULT_BUYER
    seller: tuple = C.DEFAULT_SELLER
    k: int = C.DEFAULT_K


def bid_grid(v, c, n=400):
    lo, hi = min(v.min(), c.min()), max(v.max(), c.max())
    return np.linspace(lo, hi, n)


@dataclass
class Analysis:
    settings: Settings
    v: np.ndarray
    p: np.ndarray
    c: np.ndarray
    q: np.ndarray
    first_best: float
    mechs: dict                # vcg, agv, second_best, posted -> Mechanism
    price: float               # bester Festpreis
    midpoint_bids: np.ndarray  # bestes Gebot je Wertetyp gegen einen ehrlichen Verkäufer
    midpoint_lie_gain: np.ndarray

    def gains(self, name):
        if name == "midpoint":
            return self.first_best
        return M.gains_of(self.mechs[name].x, self.v, self.p, self.c, self.q)

    def share(self, name):
        return self.gains(name) / self.first_best if self.first_best > 1e-12 else 1.0

    def budget(self, name):
        return 0.0 if name == "midpoint" else self.mechs[name].budget(self.p, self.q)

    def min_utility(self, name):
        if name == "midpoint":
            return 0.0
        m = self.mechs[name]
        return float(min(m.buyer_util(self.v, self.q).min(), m.seller_util(self.c, self.p).min()))

    def lie_gain(self, name):
        if name == "midpoint":
            return float(self.midpoint_lie_gain.max())
        m = self.mechs[name]
        return float(max(m.buyer_gain_from_lying(self.v, self.q).max(), m.seller_gain_from_lying(self.c, self.p).max()))

    def flags(self, name):
        """(effizient, ehrlich, freiwillig, ausgeglichen)"""
        return (self.share(name) >= 1 - 1e-9, self.lie_gain(name) <= C.TOL, self.min_utility(name) >= -C.TOL, self.budget(name) >= -C.TOL)

    def buyer_util(self, name):
        if name == "midpoint":
            return np.array([M.midpoint_utility_curve(vi, [vi], self.c, self.q)[0] for vi in self.v])
        return self.mechs[name].buyer_util(self.v, self.q)


@lru_cache(maxsize=64)
def analyse(settings):
    v, p = M.make_types(settings.buyer[0], settings.buyer[1], settings.k, settings.family)
    c, q = M.make_types(settings.seller[0], settings.seller[1], settings.k, settings.family)
    mechs = {"vcg": M.vcg(v, p, c, q), "agv": M.agv(v, p, c, q), "second_best": M.second_best(v, p, c, q)}
    price, posted = M.best_posted_price(v, p, c, q)
    mechs["posted"] = posted
    grid = bid_grid(v, c)
    bids, gain = [], []
    for vi in v:
        b, u = M.midpoint_best_bid(vi, c, q, grid)
        bids.append(b)
        gain.append(max(u - M.midpoint_utility_curve(vi, [vi], c, q)[0], 0.0))
    return Analysis(settings, v, p, c, q, M.first_best_gains(v, p, c, q), mechs, price, np.array(bids), np.array(gain))


# --- Experiment 1: Überlappung ---------------------------------------------------------------------------------------------------------------


def overlap_experiment(shifts=None, k=None, family="uniform"):
    shifts = C.OVERLAP_SHIFTS if shifts is None else shifts
    k = C.OVERLAP_K if k is None else k
    rows = []
    for s in shifts:
        a = analyse(Settings(family, (s, s + 100), (0, 100), k))
        rows.append({"shift": s, "first_best": a.first_best, "second_best": a.gains("second_best"), "share_second": a.share("second_best"), "share_posted": a.share("posted"),
                     "vcg_deficit": -a.budget("vcg"), "agv_min_utility": a.min_utility("agv")})
    return rows


# --- Experiment 2: Feinheit des Gitters ------------------------------------------------------------------------------------------------------


def grid_experiment(ks=None):
    """Gleichverteilt auf [0, 1] gegen [0, 1]: erwartete Gewinne (in Einheiten der Breite) gegen die bekannten Werte 1/6 (erstbest) und 9/64 (zweitbest im Kontinuum)."""
    ks = C.GRID_KS if ks is None else ks
    rows = []
    for k in ks:
        v, p = M.make_types(0.0, 1.0, k, "uniform")
        sb = M.second_best(v, p, v, p)
        rows.append({"k": k, "first_best": M.first_best_gains(v, p, v, p), "second_best": M.gains_of(sb.x, v, p, v, p)})
    return rows


# --- Experiment 3: Verteilungsfamilien -------------------------------------------------------------------------------------------------------


def family_experiment(k=None):
    k = C.COMPARE_K if k is None else k
    rows = []
    for fam in C.FAMILIES:
        a = analyse(Settings(fam, (0, 100), (0, 100), k))
        rows.append({"family": fam, "first_best": a.first_best, "share_second": a.share("second_best"), "share_posted": a.share("posted"), "vcg_deficit_share": -a.budget("vcg") / a.first_best,
                     "agv_share_types_below_zero": float(np.mean(a.mechs["agv"].buyer_util(a.v, a.q) < -C.TOL)), "midpoint_best_bid_ratio": float(np.mean((a.midpoint_bids / np.maximum(a.v, 1e-9))[a.v > 0]))})
    return rows
