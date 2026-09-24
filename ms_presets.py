"""SETTING_SPECS-Permalink-Muster und Presets (Standardmuster des Portfolios, vgl. mo_presets.py). Bereichsregler tragen ein Paar (von, bis)."""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import ms_constants as C


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def clamp_range(lo, hi):
    """Auf das Raster der Regler einrasten, in den erlaubten Bereich klemmen und eine Mindestbreite von einer Schrittweite sichern."""
    step = C.RANGE_STEP
    lo = C.RANGE_MIN + round((lo - C.RANGE_MIN) / step) * step
    hi = C.RANGE_MIN + round((hi - C.RANGE_MIN) / step) * step
    lo = int(min(max(lo, C.RANGE_MIN), C.RANGE_MAX - step))
    hi = int(min(max(hi, C.RANGE_MIN + step), C.RANGE_MAX))
    if hi <= lo:
        hi = lo + step
    return lo, hi


def _range(value):
    """'40-100' -> (40, 100), geklemmt."""
    a, b = str(value).split("-")
    lo, hi = float(a), float(b)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(value)
    return clamp_range(lo, hi)


def _int_clamped(lo_, hi_):
    def cast(value):
        return int(min(max(int(float(value)), lo_), hi_))
    return cast


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "family_select": SettingSpec("family", _choice(C.FAMILIES), "uniform"),
    "buyer_range": SettingSpec("buyer", _range, C.DEFAULT_BUYER),
    "seller_range": SettingSpec("seller", _range, C.DEFAULT_SELLER),
    "k_slider": SettingSpec("k", _int_clamped(C.K_MIN, C.K_MAX), C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "mech_select": SettingSpec("mech", _choice(C.MECHANISMS), "second_best"),
}
PRESET_KEYS = {"family": "family_select", "buyer": "buyer_range", "seller": "seller_range", "k": "k_slider", "mech": "mech_select"}

PRESETS = {
    "Klassiker: gleiche Bereiche": {"family": "uniform", "buyer": (0, 100), "seller": (0, 100), "k": 16, "mech": "second_best"},
    "Wenig Überlappung": {"family": "uniform", "buyer": (60, 160), "seller": (0, 100), "k": 16, "mech": "second_best"},
    "Getrennte Bereiche": {"family": "uniform", "buyer": (100, 200), "seller": (0, 100), "k": 16, "mech": "second_best"},
    "Glockenförmige Werte": {"family": "bell", "buyer": (0, 100), "seller": (0, 100), "k": 16, "mech": "second_best"},
    "Grobes Gitter (4 Typen)": {"family": "uniform", "buyer": (0, 100), "seller": (0, 100), "k": 4, "mech": "second_best"},
    "Lügen beim Mittelpunkt-Preis": {"family": "uniform", "buyer": (0, 100), "seller": (0, 100), "k": 16, "mech": "midpoint"},
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                st.session_state[state_key] = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            if isinstance(value, tuple):
                value = f"{value[0]}-{value[1]}"
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


PRESET_HELP = {
    "Klassiker: gleiche Bereiche": "Wert und Kosten gleichverteilt in 0 bis 100 €, 16 Typen je Seite: erstbester Gewinn 16,6 €, zweitbestes Verfahren 14,8 € (89,2 %), bester Festpreis (46,9 €) 13,2 € (79,4 %). VCG hat ein Defizit von 16,6 € - so viel wie der ganze Gewinn; AGV lässt 13 von 16 Wertetypen mit negativem erwarteten Nutzen zurück.",
    "Wenig Überlappung": "Käufer-Werte 60 bis 160 €, Verkäufer-Kosten 0 bis 100 €: nur noch 40 € Überlappung. Erstbester Gewinn 61,1 €, zweitbestes Verfahren 58,5 € (95,8 %), bester Festpreis (78,1 €) 52,0 € (85,1 %).",
    "Getrennte Bereiche": "Käufer-Werte 100 bis 200 €, Verkäufer-Kosten 0 bis 100 €: Handel ist immer effizient, und alle vier Wünsche sind zugleich erfüllbar - das zweitbeste Verfahren und sogar der Festpreis (96,9 €) erreichen 100 %. Der Konflikt entsteht erst durch Überlappung.",
    "Glockenförmige Werte": "Beta(2,2)-verteilte Typen in 0 bis 100 €: erstbester Gewinn 12,9 €, zweitbestes Verfahren 11,7 € (90,9 %), bester Festpreis 10,2 € (79,0 %). Auch ohne Gleichverteilung gibt es keinen Ausweg.",
    "Grobes Gitter (4 Typen)": "Nur 4 Typen je Seite: das zweitbeste Verfahren erreicht 100 % - der Konflikt verschwindet, weil so grobe Typenräume Anreize leichter zulassen (ein Gitter-Effekt, kein Widerspruch zur Theorie: sie gilt für stetige Typen). Der Festpreis erreicht 90 %.",
    "Lügen beim Mittelpunkt-Preis": "Mechanismus-Detail: Mittelpunkt-Preis. Gegen einen ehrlichen Verkäufer bietet der Käufer im Mittel nur 62 % seines Werts; beim mittleren Wert 53,1 € bietet er 34,4 € und gewinnt 2,93 € erwarteten Nutzen. Das zweitbeste Verfahren lohnt Lügen nie.",
}
