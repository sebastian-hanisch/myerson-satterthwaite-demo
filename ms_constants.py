"""Konstanten der Myerson-Satterthwaite-Demo: Vehikel C "Slot-Handel" (ein freier Laderaum-Slot, Verkäufer mit privaten Kosten, Käufer mit privatem Wert), Regler, Experimente."""

EPS = 1e-9
TOL = 1e-7
SEED_MAX = 999999

FAMILIES = ("uniform", "bell")
FAMILY_LABELS = {"uniform": "Gleichverteilt", "bell": "Glockenförmig (Beta(2,2))"}

# Bereiche in Euro je Slot; Schrittweite der Regler
RANGE_MIN, RANGE_MAX, RANGE_STEP = 0, 200, 5
DEFAULT_BUYER = (0, 100)
DEFAULT_SELLER = (0, 100)
K_MIN, K_MAX, K_STEP, DEFAULT_K = 4, 30, 1, 16

MECHANISMS = ("vcg", "agv", "second_best", "posted", "midpoint")
MECHANISM_LABELS = {
    "vcg": "Effizient + ehrlich + freiwillig (VCG)",
    "agv": "Effizient + ehrlich + ausgeglichen (AGV)",
    "second_best": "Ehrlich + freiwillig + ausgeglichen (zweitbestes)",
    "posted": "Festpreis",
    "midpoint": "Mittelpunkt-Preis",
}

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

OVERLAP_SHIFTS = (0, 20, 40, 60, 80, 100)          # Käufer-Bereich [s, s+100] gegen Verkäufer-Bereich [0, 100]
OVERLAP_K = 16
GRID_KS = (6, 10, 16, 24, 32, 40)
COMPARE_K = 16
