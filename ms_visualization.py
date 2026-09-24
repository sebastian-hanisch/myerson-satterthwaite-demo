"""Plotly-Abbildungen der Myerson-Satterthwaite-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ms_constants as C

LINE_COLOR = "#4c78a8"
REF_COLOR = "#7f7f7f"
GOOD = "#54a24b"
BAD = "#e45756"
WARN = "#f58518"
PURPLE = "#b279a2"
MECH_COLORS = {"vcg": WARN, "agv": BAD, "second_best": PURPLE, "posted": LINE_COLOR, "midpoint": GOOD}
SHORT = {"vcg": "VCG", "agv": "AGV", "second_best": "Zweitbestes", "posted": "Festpreis", "midpoint": "Mittelpunkt"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_trade_maps(a):
    """Links: wo effizienter Handel stattfindet (Wert v > Kosten c); rechts: Handelswahrscheinlichkeit des zweitbesten Mechanismus."""
    efficient = (a.v[:, None] > a.c[None, :]).astype(float)
    sb = a.mechs["second_best"].x
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Erstbest: handeln, wenn Wert > Kosten", "Zweitbest: Handelswahrscheinlichkeit"), horizontal_spacing=0.12)
    for col, z, name in ((1, efficient, "Erstbest"), (2, sb, "Zweitbest")):
        fig.add_trace(go.Heatmap(x=a.c, y=a.v, z=z, zmin=0, zmax=1, colorscale=[[0, "#f4f4f4"], [1, PURPLE if col == 2 else LINE_COLOR]], showscale=False, name=name,
                                 hovertemplate="Kosten %{x:.0f} €, Wert %{y:.0f} €: %{z:.2f}<extra></extra>"), row=1, col=col)
        lo, hi = min(a.c.min(), a.v.min()), max(a.c.max(), a.v.max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=REF_COLOR, dash="dot"), showlegend=False, hoverinfo="skip"), row=1, col=col)
        fig.update_xaxes(title_text="Kosten des Verkäufers (€)", row=1, col=col)
        fig.update_yaxes(title_text="Wert des Käufers (€)", row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_mechanisms(a):
    """Anteil am erstbesten Gewinn je Mechanismus."""
    names = list(C.MECHANISMS)
    shares = [100 * a.share(m) for m in names]
    fig = go.Figure(go.Bar(x=[SHORT[m] for m in names], y=shares, marker_color=[MECH_COLORS[m] for m in names], text=[f"{s:.0f} %" for s in shares], textposition="outside", showlegend=False))
    fig.update_yaxes(title_text="Anteil am erstbesten Gewinn (%)", range=[0, 118])
    return _base(fig, 300)


def build_utility_by_type(a):
    """Erwarteter Nutzen eines Käufers je Wertetyp (unter der Null-Linie: er würde lieber gar nicht teilnehmen)."""
    fig = go.Figure()
    for m in C.MECHANISMS:
        fig.add_trace(go.Scatter(x=a.v, y=a.buyer_util(m), mode="lines+markers", name=SHORT[m], line=dict(color=MECH_COLORS[m], width=2.2), marker=dict(size=5)))
    fig.add_hline(y=0, line=dict(color=REF_COLOR, dash="dot"), annotation_text="Teilnahme lohnt gerade nicht", annotation_position="bottom right")
    fig.update_xaxes(title_text="Wert des Käufers (€)")
    fig.update_yaxes(title_text="Erwarteter Nutzen (€)")
    return _base(fig, 340)


def build_lying(a, name, value, reports, utils, best_report):
    """Erwarteter Nutzen eines Käufers mit dem Wert `value` in Abhängigkeit von seiner Meldung (bzw. seinem Gebot beim Mittelpunkt-Preis)."""
    fig = go.Figure()
    shape = "linear" if name == "midpoint" else "hv"
    fig.add_trace(go.Scatter(x=list(reports), y=list(utils), mode="lines", line=dict(color=MECH_COLORS[name], width=2.5, shape=shape), name="Nutzen", hovertemplate="Meldung %{x:.0f} €: Nutzen %{y:.1f} €<extra></extra>"))
    truth_u = float(np.interp(value, reports, utils))
    fig.add_trace(go.Scatter(x=[value], y=[truth_u], mode="markers", marker=dict(size=13, color=GOOD, symbol="circle", line=dict(color="white", width=1)), name="ehrlich melden"))
    best_u = float(np.interp(best_report, reports, utils))
    fig.add_trace(go.Scatter(x=[best_report], y=[best_u], mode="markers", marker=dict(size=13, color=WARN, symbol="diamond", line=dict(color="white", width=1)), name="beste Meldung"))
    fig.update_xaxes(title_text="Gemeldeter Wert bzw. Gebot (€)")
    fig.update_yaxes(title_text="Erwarteter Nutzen (€)")
    return _base(fig, 320)


def build_overlap(rows):
    xs = [100 - r["shift"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["share_second"] for r in rows], mode="lines+markers", name="Zweitbestes", line=dict(color=PURPLE, width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["share_posted"] for r in rows], mode="lines+markers", name="bester Festpreis", line=dict(color=LINE_COLOR, width=2.5)))
    fig.add_hline(y=100, line=dict(color=REF_COLOR, dash="dot"))
    fig.update_xaxes(title_text="Überlappung der Wert- und Kostenbereiche (€)", autorange="reversed")
    fig.update_yaxes(title_text="Anteil am erstbesten Gewinn (%)", range=[70, 105])
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def build_grid(rows):
    xs = [r["k"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["second_best"] / r["first_best"] for r in rows], mode="lines+markers", name="zweitbester Gewinn / erstbester Gewinn", line=dict(color=PURPLE, width=2.5)))
    fig.add_hline(y=100 * 9 / 64 / (1 / 6), line=dict(color=BAD, dash="dash"), annotation_text="Kontinuum: 9/64 : 1/6 = 84,4 %", annotation_position="bottom right")
    fig.update_xaxes(title_text="Feinheit des Typengitters (Typen je Seite)")
    fig.update_yaxes(title_text="Prozent", range=[75, 102])
    return _base(fig, 340)


def build_family(rows):
    labels = [C.FAMILY_LABELS[r["family"]] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * r["share_second"] for r in rows], name="Zweitbestes", marker_color=PURPLE, text=[f"{100 * r['share_second']:.0f}" for r in rows], textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["share_posted"] for r in rows], name="bester Festpreis", marker_color=LINE_COLOR, text=[f"{100 * r['share_posted']:.0f}" for r in rows], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Anteil am erstbesten Gewinn (%)", range=[0, 118])
    return _base(fig, 320)
