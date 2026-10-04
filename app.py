"""Myerson-Satterthwaite - warum zwei Spediteure keinen effizienten, ehrlichen und freiwilligen Slot-Handel bekommen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes und letztes Stück der Linie "Spieltheorie & Mechanism Design" der "Konzepte"-Reihe (Nachfolger von moulin-demo): Ein Spediteur verkauft einem anderen einen freien Laderaum-Slot; beide kennen
nur ihren eigenen Wert bzw. ihre eigenen Kosten. Kein Mechanismus ist zugleich effizient, ehrlich, freiwillig und ausgeglichen.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import ms_constants as C
import ms_evaluation as E
import ms_mechanisms as M
from ms_evaluation import Settings, analyse, family_experiment, grid_experiment, overlap_experiment
from ms_presets import PRESET_HELP, PRESETS, apply_preset, clamp_range, init_session_state_defaults, load_permalink_settings, sync_query_params
from ms_visualization import SHORT, build_family, build_grid, build_lying, build_mechanisms, build_overlap, build_trade_maps, build_utility_by_type

st.set_page_config(page_title="Myerson-Satterthwaite – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0                                                   # kein "-0,0"
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


def yn(flag):
    return "ja" if flag else "nein"


@st.cache_data(show_spinner=False)
def _overlap(shifts, k):
    return overlap_experiment(shifts=shifts, k=k)


@st.cache_data(show_spinner=False)
def _grid(ks):
    return grid_experiment(ks=ks)


@st.cache_data(show_spinner=False)
def _family(k):
    return family_experiment(k=k)


st.title("🤝 Myerson-Satterthwaite – warum ehrlicher Handel nie ganz effizient ist")
st.markdown(
    """
Ein Spediteur hat auf seiner Tour einen freien Laderaum-**Slot**, ein anderer könnte ihn gebrauchen. Der Verkäufer kennt seine **Kosten** $c$ (was ihn der Slot sonst einbrächte), der Käufer seinen **Wert** $v$ - beide
nur für sich. Effizient wäre der Handel genau dann, wenn $v > c$. Myerson und Satterthwaite (1983) zeigen: **kein Mechanismus** kann zugleich **effizient**, **ehrlich** (Bayes-Nash), **freiwillig** und
**ausgeglichen** (kein Zuschuss von außen) sein, sobald Wert und Kosten sich überschneiden können. Die Demo rechnet das exakt - mit einem linearen Programm für das beste erreichbare Verfahren - und zeigt, welche
Eigenschaft jeder bekannte Mechanismus opfert und wie viel Gewinn der Handel dabei verliert.
"""
)
st.caption(
    "Zehntes und letztes Stück der Linie \"Spieltheorie & Mechanism Design\" der \"Konzepte\"-Reihe, Nachfolger von **moulin-demo** (dort: Kostenteilung mit Zahlungsbereitschaft; hier: zwei Seiten mit privaten Werten). "
    "Werte und Kosten in Euro je Slot; alle Verteilungen auf einem Typengitter diskretisiert."
)

with st.expander("So funktioniert der Handel", expanded=True):
    st.markdown(
        """
1. **Typen.** Wert $v$ des Käufers und Kosten $c$ des Verkäufers sind unabhängig verteilt (Bereiche und Form stellen Sie ein). Beide melden dem Mechanismus einen Typ - ehrlich oder nicht.
2. **Mechanismus.** Aus den Meldungen bestimmt er, ob gehandelt wird und was der Käufer zahlt und der Verkäufer erhält.
3. **Vier Wünsche:** **effizient** (handeln, wenn $v > c$), **ehrlich** (keine Meldung lohnt mehr als die wahre), **freiwillig** (jeder Typ steht sich mindestens so gut wie ohne Handel), **ausgeglichen** (der Käufer
   zahlt im Mittel mindestens so viel, wie der Verkäufer erhält).
4. **Unmöglichkeit.** Überschneiden sich die Bereiche, gibt es keinen Mechanismus mit allen vier. Jeder gibt eine auf: **VCG** den Ausgleich (Defizit), **AGV** die Freiwilligkeit, das **zweitbeste** Verfahren und
   der **Festpreis** die Effizienz, der **Mittelpunkt-Preis** die Ehrlichkeit.
5. **Zweitbestes Verfahren.** Unter allen ehrlichen, freiwilligen, ausgeglichenen Mechanismen liefert das lineare Programm den größten erwarteten Gewinn aus Handel.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    family = st.selectbox("Verteilung der Typen", C.FAMILIES, key="family_select", format_func=lambda k: C.FAMILY_LABELS[k], help="Gleichverteilt über den ganzen Bereich oder glockenförmig (Beta(2,2)), das heißt Werte in der Mitte sind häufiger.")
    buyer_raw = st.slider("Wert des Käufers (€)", C.RANGE_MIN, C.RANGE_MAX, key="buyer_range", step=C.RANGE_STEP, help="Bereich, in dem der private Wert des Käufers liegt.")
    seller_raw = st.slider("Kosten des Verkäufers (€)", C.RANGE_MIN, C.RANGE_MAX, key="seller_range", step=C.RANGE_STEP, help="Bereich, in dem die privaten Kosten des Verkäufers liegen.")
    k = st.slider("Feinheit des Typengitters", C.K_MIN, C.K_MAX, key="k_slider", step=C.K_STEP, help="Zahl der Typen je Seite. Je feiner, desto näher an stetigen Verteilungen; das lineare Programm wächst mit k².")
    mech = st.selectbox("Mechanismus im Detail", C.MECHANISMS, key="mech_select", format_func=lambda m: C.MECHANISM_LABELS[m], help="Welcher Mechanismus im Abschnitt \"Lohnt sich eine Lüge?\" untersucht wird.")

buyer = clamp_range(*buyer_raw)
seller = clamp_range(*seller_raw)
sync_query_params({"family_select": family, "buyer_range": buyer, "seller_range": seller, "k_slider": int(k), "mech_select": mech})

settings = Settings(family, buyer, seller, int(k))
with st.spinner("Löse das lineare Programm..."):
    a = analyse(settings)
v, c = a.v, a.c
sb_share = a.share("second_best")
overlap = max(0.0, min(buyer[1], seller[1]) - max(buyer[0], seller[0]))

# --- Wie viel ist möglich -----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wie viel Handelsgewinn ist möglich?")
m1, m2, m3 = st.columns(3)
m1.metric("Erstbester Gewinn", f"{de(a.first_best)} €", help="Erwarteter Gewinn, wenn immer gehandelt wird, sobald Wert > Kosten (alle Typen bekannt).")
m2.metric("Zweitbester Gewinn", f"{de(a.gains('second_best'))} €", delta=f"{pct(sb_share - 1, 1)} gegenüber erstbest", delta_color="off",
          help="Bester erwarteter Gewinn eines ehrlichen, freiwilligen, ausgeglichenen Mechanismus (lineares Programm).")
m3.metric("Beste Festpreis-Regel", f"{de(a.gains('posted'))} €", delta=f"Preis {de(a.price)} €", delta_color="off", help="Bester fester Preis, zu dem beide handeln dürfen (Käufer nimmt an bei v ≥ Preis, Verkäufer bei c ≤ Preis).")
st.plotly_chart(build_trade_maps(a), width="stretch", key="trade_maps")
st.caption(
    f"Die Bereiche überlappen sich um {de(overlap, 0)} €. Links handelt der erstbeste Mechanismus überall oberhalb der Diagonale (Wert > Kosten). Rechts: das zweitbeste Verfahren handelt in einem Teil davon nicht - "
    "und zwar dort, wo der Gewinn klein ist (nahe der Diagonale): wer dort handeln ließe, müsste ehrliche Meldungen mit Zuschüssen erkaufen."
)
if a.flags("second_best") == (True, True, True, True):
    st.success("✅ Hier gibt es keinen Konflikt: das zweitbeste Verfahren handelt genauso viel wie das erstbeste - alle vier Eigenschaften sind gleichzeitig erfüllbar. Der Handel ist entweder immer oder nie effizient, "
               "oder das Gitter ist so grob, dass die Anreize aufgehen.")
else:
    st.warning(f"⚠️ Kein Mechanismus erfüllt alle vier Wünsche: das beste ehrliche, freiwillige und ausgeglichene Verfahren erreicht {pct(sb_share, 1)} des erstbesten Gewinns und verliert {de(a.first_best - a.gains('second_best'))} € "
               "je Handelschance im Mittel.")

st.markdown("---")

# --- Fünf Mechanismen ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Fünf Mechanismen, jeder gibt etwas auf")
rows_m = []
for m in C.MECHANISMS:
    eff, honest, vol, bal = a.flags(m)
    rows_m.append({"Mechanismus": C.MECHANISM_LABELS[m], "Gewinn (€)": de(a.gains(m)), "Anteil am Erstbesten": pct(a.share(m)), "Budgetsaldo (€)": de(a.budget(m)),
                   "Niedrigster erwarteter Nutzen (€)": de(a.min_utility(m)), "Größter Lügen-Gewinn (€)": de(a.lie_gain(m), 2), "Effizient": yn(eff), "Ehrlich": yn(honest), "Freiwillig": yn(vol), "Ausgeglichen": yn(bal)})
st.dataframe(rows_m, hide_index=True)
st.plotly_chart(build_mechanisms(a), width="stretch", key="mechanisms_chart")
prop_names = ("effizient", "ehrlich", "freiwillig", "ausgeglichen")
verletzt = "; ".join(f"{SHORT[m]}: " + (", ".join(n for n, ok in zip(prop_names, a.flags(m)) if not ok) or "nichts") for m in C.MECHANISMS)
st.info(f"Verletzt wird je Verfahren - {verletzt}. Der Mittelpunkt-Preis ist nur effizient, solange alle ehrlich bieten; das zweitbeste Verfahren verzichtet vor allem auf Handel mit kleinem Gewinn (im Klassiker im Mittel 11,7 € gegen 35,4 € bei allen effizienten Handeln); der Festpreis verzichtet auf jeden Handel, bei dem der Preis nicht zwischen Kosten und Wert liegt - auch bei großem Gewinn.")
st.markdown("##### Erwarteter Nutzen des Käufers je Wert")
st.plotly_chart(build_utility_by_type(a), width="stretch", key="utility_chart")
low = int(np.sum(a.mechs["agv"].buyer_util(v, a.q) < -C.TOL))
st.caption(
    f"Unter der Null-Linie würde der Käufer lieber gar nicht teilnehmen. Beim AGV-Verfahren (effizient und ausgeglichen, aber nicht freiwillig) betrifft das {low} von {len(v)} Wertetypen; sein niedrigster erwarteter Nutzen liegt bei "
    f"{de(a.min_utility('agv'))} €. Beim VCG-Verfahren steht sich jeder Typ mindestens so gut wie ohne Handel, aber die Zahlungen decken die Kosten nicht: Defizit {de(-a.budget('vcg'))} € im Mittel je Handelschance."
)

st.markdown("---")

# --- Lohnt sich eine Lüge? --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Lohnt sich eine Lüge?")
st.caption(f"Wählen Sie den wahren Wert des Käufers. Die Kurve zeigt seinen erwarteten Nutzen für jede mögliche Meldung (bzw. jedes Gebot beim Mittelpunkt-Preis); der Verkäufer meldet ehrlich. Mechanismus: {C.MECHANISM_LABELS[mech]}.")
sig = f"{family}_{buyer}_{seller}_{int(k)}"
options = [round(float(x), 1) for x in v]
truth = st.select_slider("Wahrer Wert des Käufers (€)", options=options, value=options[len(options) // 2], key=f"lie_value_{sig}")
ti = options.index(truth)
if mech == "midpoint":
    reports = E.bid_grid(v, c)
    utils = M.midpoint_utility_curve(float(v[ti]), reports, c, a.q)
    truthful_u = float(M.midpoint_utility_curve(float(v[ti]), [float(v[ti])], c, a.q)[0])
else:
    m_obj = a.mechs[mech]
    reports = v
    utils = float(v[ti]) * (m_obj.x @ a.q) - m_obj.TB
    truthful_u = float(utils[ti])
best_k = int(np.argmax(utils))
best_report = float(reports[best_k])
gain = float(utils[best_k]) - truthful_u
st.plotly_chart(build_lying(a, mech, float(v[ti]), reports, utils, best_report), width="stretch", key="lying_chart")
if gain > C.TOL:
    st.warning(f"⚠️ Lügen lohnt sich: meldet der Käufer {de(best_report)} € statt seines wahren Werts {de(v[ti])} €, steigt sein erwarteter Nutzen von {de(truthful_u, 2)} € auf {de(utils[best_k], 2)} € (Gewinn {de(gain, 2)} €)"
               + (f" - er bietet {pct(best_report / v[ti])} seines Werts." if mech == "midpoint" and v[ti] > 0 else "."))
else:
    st.success(f"✅ Ehrlich melden ist optimal: der erwartete Nutzen ist {de(truthful_u, 2)} €, keine andere Meldung ist besser.")

st.markdown("---")

# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie viel kostet die Ehrlichkeit bei wachsender Überlappung?")
st.caption(f"Gleichverteilt, Käufer-Bereich [s, s+100] gegen Verkäufer-Bereich [0, 100] für s = {', '.join(str(s) for s in C.OVERLAP_SHIFTS)}; Gitter mit {C.OVERLAP_K} Typen je Seite.")
if st.button("Überlappung durchrechnen", key="overlap_start"):
    st.session_state["overlap_on"] = True
if st.session_state.get("overlap_on"):
    rows_o = _overlap(C.OVERLAP_SHIFTS, C.OVERLAP_K)
    st.plotly_chart(build_overlap(rows_o), width="stretch", key="overlap_chart")
    r0, rl = rows_o[0], rows_o[-1]
    st.warning(
        f"**Befund:** Bei voller Überlappung erreicht das zweitbeste Verfahren {pct(r0['share_second'], 1)} des erstbesten Gewinns, der beste Festpreis {pct(r0['share_posted'], 1)}. Mit weniger Überlappung schrumpft der Verlust - zunächst kaum, dann schnell "
        f"({pct(rows_o[3]['share_second'], 1)} bei 40 € Überlappung) und verschwindet bei getrennten Bereichen ({pct(rl['share_second'], 1)}). Das VCG-Defizit ist dabei jedes Mal so groß wie der gesamte Gewinn "
        f"({de(r0['vcg_deficit'])} € bis {de(rl['vcg_deficit'])} €)."
    )

st.markdown("---")

# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Stimmt das mit der Theorie überein? Feineres Typengitter")
st.caption(f"Gleichverteilt auf [0, 1] gegen [0, 1] mit {', '.join(str(x) for x in C.GRID_KS)} Typen je Seite. Im stetigen Grenzfall sind die bekannten Werte 1/6 (erstbest) und 9/64 (zweitbest).")
if st.button("Gitter verfeinern", key="grid_start"):
    st.session_state["grid_on"] = True
if st.session_state.get("grid_on"):
    rows_g = _grid(C.GRID_KS)
    st.plotly_chart(build_grid(rows_g), width="stretch", key="grid_chart")
    g0, gl = rows_g[0], rows_g[-1]
    st.warning(
        f"**Befund:** Bei nur {g0['k']} Typen je Seite liegt das zweitbeste Verfahren bei {pct(g0['second_best'] / g0['first_best'], 1)} des erstbesten Gewinns - der Konflikt ist fast verschwunden, weil so grobe Typenräume Anreize "
        f"leichter zulassen. Mit {gl['k']} Typen sind es {pct(gl['second_best'] / gl['first_best'], 1)}, der zweitbeste Gewinn liegt bei {de(gl['second_best'], 4)} statt der stetigen 9/64 = {de(9 / 64, 4)}, der erstbeste bei {de(gl['first_best'], 4)} "
        f"statt 1/6 = {de(1 / 6, 4)}. Die Werte fallen mit feinerem Gitter monoton auf 84,4 % zu."
    )

st.markdown("---")

# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Gilt das nur für gleichverteilte Typen?")
st.caption(f"Bereiche [0, 100] gegen [0, 100], {C.COMPARE_K} Typen je Seite; gleichverteilt gegen glockenförmig.")
if st.button("Verteilungen vergleichen", key="family_start"):
    st.session_state["family_on"] = True
if st.session_state.get("family_on"):
    rows_f = _family(C.COMPARE_K)
    st.plotly_chart(build_family(rows_f), width="stretch", key="family_chart")
    fu, fb = rows_f
    st.warning(
        f"**Befund:** Auch bei glockenförmigen Typen gibt es keinen Ausweg: das zweitbeste Verfahren erreicht {pct(fb['share_second'], 1)} (gleichverteilt: {pct(fu['share_second'], 1)}), der Festpreis {pct(fb['share_posted'], 1)} "
        f"({pct(fu['share_posted'], 1)}). Das AGV-Verfahren lässt bei gleichverteilten Typen {pct(fu['agv_share_types_below_zero'])}, bei glockenförmigen {pct(fb['agv_share_types_below_zero'])} der Käufertypen mit negativem erwartetem Nutzen zurück; "
        f"beim Mittelpunkt-Preis bietet der Käufer im Mittel nur {pct(fu['midpoint_best_bid_ratio'])} (gleichverteilt) bzw. {pct(fb['midpoint_best_bid_ratio'])} (glockenförmig) seines Werts."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genau ein Käufer und ein Verkäufer** | In großen Märkten verschwindet der Verlust: McAfee (1992) erreicht mit dem Weglassen des am wenigsten lohnenden Handels dominante Ehrlichkeit, Rustichini/Satterthwaite/Williams (1994) zeigen Konvergenz zur Effizienz. | Doppelte Auktionen |
| **Werte und Kosten sind unabhängig verteilt** | Die Unmöglichkeit gilt für unabhängige Typen; bei geeignet korrelierten Typen kann sie entfallen. | – |
| **Beide kennen die Verteilung des anderen** | Der Mechanismus braucht das gemeinsame Vorwissen; ohne es sind nur dominante Strategien (z. B. Festpreis) verlässlich. | Festpreis (hier) |
| **Nutzen quasilinear, ein unteilbarer Slot** | Bei teilbaren Gütern oder Budgetgrenzen ändert sich die Lage; hier ist es ein einzelner Slot. | – |
| **Diskretisierung** | Das Typengitter ist eine Näherung; grobe Gitter zeigen den Konflikt kaum (Experiment oben). Das stetige Ergebnis 9/64 : 1/6 = 84,4 % ist Literatur, hier gemessen wird die Annäherung. | – |
"""
)
st.caption(
    "Verwandt: [moulin-demo](https://sebastianhanisch-moulin-demo.streamlit.app/) (Vorgänger: Kostenteilung mit privaten Werten), [maut-demo](https://sebastianhanisch-maut-demo.streamlit.app/) "
    "(Preise, die Externalitäten einpreisen), auction-demo (Auktionen, VCG)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Wert $v \sim F_B$ auf $[\underline v, \overline v]$, Kosten $c \sim F_S$ auf $[\underline c, \overline c]$, unabhängig; diskretisiert auf $k$ Typen je Seite mit Wahrscheinlichkeiten $p_i$, $q_j$. Direkter Mechanismus:
Handelswahrscheinlichkeit $x_{ij} \in [0,1]$ und erwartete Zahlungen $T^B_i$ (Käufer) und $T^S_j$ (Verkäufer erhält). Mit $X^B_i = \sum_j q_j x_{ij}$ und $X^S_j = \sum_i p_i x_{ij}$:

- **Ehrlich (BIC):** $v_i X^B_i - T^B_i \ge v_i X^B_{i'} - T^B_{i'}$ für alle $i, i'$; $T^S_j - c_j X^S_j \ge T^S_{j'} - c_j X^S_{j'}$ für alle $j, j'$.
- **Freiwillig (IR):** $v_i X^B_i - T^B_i \ge 0$ und $T^S_j - c_j X^S_j \ge 0$.
- **Ausgeglichen:** $\sum_i p_i T^B_i \ge \sum_j q_j T^S_j$.
- **Zweitbestes Verfahren:** $\max \sum_{ij} p_i q_j x_{ij} (v_i - c_j)$ unter diesen Bedingungen - ein lineares Programm mit $k^2 + 2k$ Variablen.

**Myerson/Satterthwaite (1983).** Überlappen die Träger (es gilt $\underline v < \overline c$ und $\underline c < \overline v$), ist der Maximalwert strikt kleiner als der erstbeste Gewinn $E[(v-c)^+]$. Für Gleichverteilung auf $[0,1]^2$:
erstbest $1/6$, zweitbest $9/64$ (84,4 %; gleich dem linearen Gleichgewicht der doppelten Auktion von Chatterjee/Samuelson 1983).

**VCG.** $x = \mathbb 1[v > c]$; der Käufer zahlt $c$, der Verkäufer erhält $v$: dominant ehrlich und freiwillig, Defizit $v - c$ je Handel, also im Mittel der ganze erstbeste Gewinn.

**AGV** (d'Aspremont/Gérard-Varet 1979). $x = \mathbb 1[v > c]$; der Käufer zahlt $A_i + \sum_j q_j B_j$, der Verkäufer erhält $B_j + \sum_i p_i A_i$ mit $A_i = E_c[c\,x]$, $B_j = E_v[v\,x]$: Bayes-Nash-ehrlich und ausgeglichen,
aber der Käufer mit kleinem Wert erwartet einen negativen Nutzen (Gleichverteilung: $v^2/2 - 1/3$ je Einheitsbereich).

**Festpreis.** Handel, wenn $c \le P \le v$; dominant ehrlich, freiwillig, ausgeglichen. Für $[0,1]^2$ ist $P = 1/2$ optimal mit Gewinn $1/8$.

**Mittelpunkt-Preis.** Handel zu $(b + s)/2$, wenn Gebot $b \ge$ Verkäuferforderung $s$: bei ehrlichen Angaben effizient, aber der Käufer bietet gegen einen ehrlichen Verkäufer $b = 2v/3$ (Gleichverteilung).

Implementiert in `ms_mechanisms.py` (Mechanismen, LP), `ms_evaluation.py` (Analyse, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Spieltheorie: von Nash bis Myerson-Satterthwaite](https://sebastianhanisch.net/konzepte-spieltheorie.html)."
)
