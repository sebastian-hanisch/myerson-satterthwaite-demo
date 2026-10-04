# 🤝 Myerson-Satterthwaite – warum ehrlicher Handel nie ganz effizient ist

**[→ Demo live ausprobieren](https://sebastianhanisch-myerson-satterthwaite-demo.streamlit.app/)**

Zehntes und letztes Stück der **Spieltheorie-&-Mechanism-Design-Linie** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning, und Nachfolger von [moulin-demo](https://sebastianhanisch-moulin-demo.streamlit.app/) im zweiten Ast
(kooperative Spieltheorie / Mechanism Design). Dort hatten mehrere Spediteure private Werte für eine gemeinsame Tour; hier stehen sich **zwei Seiten** gegenüber, und beide haben private Information.

## Warum dieses Problem

Ein Spediteur hat auf seiner Tour einen freien Laderaum-**Slot**, ein anderer könnte ihn gebrauchen. Der Verkäufer kennt seine **Kosten** $c$, der Käufer seinen **Wert** $v$. Effizient wäre der Handel genau dann,
wenn $v > c$. **Myerson und Satterthwaite (1983):** sobald sich die möglichen Werte und Kosten überschneiden, gibt es **keinen Mechanismus**, der zugleich **effizient**, **ehrlich** (Bayes-Nash),
**freiwillig** (interim) und **im Erwartungswert ausgeglichen** ist. Die Demo rechnet das exakt aus und zeigt, welche Eigenschaft jeder bekannte Mechanismus aufgibt.

## Modell

**Vehikel C "Slot-Handel"** (`ms_mechanisms.py`): Wert des Käufers und Kosten des Verkäufers unabhängig, gleichverteilt oder glockenförmig (Beta(2,2)) auf einstellbaren Bereichen in Euro, diskretisiert auf $k$ Typen je Seite.
Ein direkter Mechanismus besteht aus Handelswahrscheinlichkeiten $x_{ij}$ und erwarteten Zahlungen $T^B_i$ (Käufer) und $T^S_j$ (Verkäufer erhält).

Fünf Mechanismen, jeder gibt genau eine Eigenschaft auf:
- **VCG** (effizient, ehrlich, freiwillig): gibt den **Ausgleich** auf – Defizit im Mittel gleich dem ganzen Gewinn.
- **AGV** (d'Aspremont/Gérard-Varet 1979; effizient, ehrlich, ausgeglichen): gibt die **Freiwilligkeit** auf.
- **Zweitbestes Verfahren** (ehrlich, freiwillig, ausgeglichen): gibt die **Effizienz** teilweise auf – das **lineare Programm** über $x_{ij}, T^B_i, T^S_j$ liefert den größten erreichbaren Gewinn.
- **Bester Festpreis** (ehrlich, freiwillig, ausgeglichen): gibt die Effizienz stärker auf, ist aber dominant ehrlich.
- **Mittelpunkt-Preis** (bei ehrlichen Geboten effizient, freiwillig, ausgeglichen): gibt die **Ehrlichkeit** auf; der Käufer bietet gegen einen ehrlichen Verkäufer nur einen Teil seines Werts.

Alle Eigenschaften werden **exakt** geprüft (Lügen-Gewinn je Typ, niedrigster erwarteter Nutzen, Budgetsaldo), nicht behauptet.

## Methodik

- **Handrechnungen (2 × 2 Typen, Werte {2, 4}, Kosten {1, 3}):** erstbester Gewinn 1,25; VCG-Zahlungen (0,5 / 2 und 3 / 2), Budget −1,25; AGV-Zahlungen (3 / 4,5 und 4,25 / 3,25), Budget 0, erwartete Käufer-Nutzen −2 und −0,5;
  Festpreis (Gewinn 1,0); Mittelpunkt-Preis (Nutzen für Gebot 1/2/3/4: 1,5 / 1,25 / 1,5 / 1,0).
- **Gegenprobe des linearen Programms:** das dünnbesetzte LP (interimistische Zahlungen) stimmt in drei Instanzen mit einer unabhängig geschriebenen dichten Formulierung (Zahlungen je Typenpaar) auf 1e-6 überein;
  die Lösung erfüllt alle Bedingungen exakt (kein Lügen-Gewinn, kein negativer Nutzen, Budget ≥ 0).
- **Gegenprobe zur Literatur:** bei gleichverteilten Typen auf $[0,1]^2$ erwartet die Theorie erstbest $1/6$, zweitbest $9/64$ (84,4 %). Das Gitter nähert sich von oben: bei 40 Typen 0,1439; Richardson-Extrapolation (Fehler ~ 1/k)
  aus 32 und 40 Typen ergibt 0,1409 (Test: innerhalb 1,5e-3 von 9/64). Der AGV-Nutzen des Käufers stimmt mit der Formel $v^2/2 - 1/3$ überein (Abweichung < 3e-3 bei 200 Typen); der beste Festpreis liegt bei 1/2 mit Gewinn 1/8;
  das beste Gebot beim Mittelpunkt-Preis bei $2v/3$.
- **Literatur** (per Recherche geprüft, nicht nachgebaut): Myerson/Satterthwaite 1983 ("Efficient mechanisms for bilateral trading", JET 29), Chatterjee/Samuelson 1983 (doppelte Auktion; lineares Gleichgewicht erreicht im
  gleichverteilten Fall das Zweitbeste), d'Aspremont/Gérard-Varet 1979 ("Incentives and incomplete information", J. Public Economics 11), McAfee 1992 ("A dominant strategy double auction", JET 56),
  Rustichini/Satterthwaite/Williams 1994 ("Convergence to efficiency in a simple market with incomplete information", Econometrica 62).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Wie viel Gewinn geht verloren? (Klassiker: Bereiche 0 bis 100 €, 16 Typen je Seite) | Erstbest 16,6 €; zweitbestes Verfahren 14,8 € (89,2 %); bester Festpreis (46,9 €) 13,2 € (79,4 %). VCG-Defizit 16,6 € – so viel wie der ganze Gewinn; AGV lässt 13 von 16 Wertetypen mit negativem erwarteten Nutzen zurück. | `test_preset_classic` |
| Wo verzichtet das zweitbeste Verfahren auf Handel? | Auf 32,6 % der effizienten Handelswahrscheinlichkeit, aber nur dort, wo der Gewinn klein ist: im Mittel 11,7 € gegen 35,4 € im Durchschnitt aller effizienten Handel. Die Handelswahrscheinlichkeit steigt im Wert und fällt in den Kosten; bei Wert ≤ Kosten wird nie gehandelt. | `test_second_best_refuses_trade_where_the_gain_is_small` |
| Wie hängt der Verlust an der Überlappung? | Gleichverteilt, Käufer-Bereich $[s, s+100]$ gegen $[0, 100]$: zweitbestes Verfahren 89,2 % bei voller Überlappung (Festpreis 79,4 %), 95,8 % bei 40 € Überlappung, 100 % bei getrennten Bereichen – zunächst kaum Veränderung (89,2 → 89,1 %), dann schnell. Das VCG-Defizit ist jedes Mal so groß wie der ganze Gewinn. | `test_overlap_experiment_numbers`, `test_preset_little_overlap`, `test_preset_disjoint` |
| Ist das ein Gitter-Artefakt? | Teilweise: bei 4 Typen je Seite verschwindet der Konflikt ganz (100 %), bei 6 sind es 95,2 %, bei 40 Typen 86,4 %; die Werte fallen monoton auf den stetigen Grenzwert 84,4 % zu (0,1439 statt 9/64 = 0,1406 bei 40 Typen). Der Konflikt gilt für stetige Typen; grobe Gitter lassen Anreize leichter zu. | `test_grid_experiment_numbers`, `test_grid_converges_to_the_continuous_values_one_sixth_and_nine_sixtyfourths`, `test_preset_coarse_grid` |
| Gilt das nur für gleichverteilte Typen? | Nein: bei Beta(2,2)-Typen 90,9 % (Festpreis 79,0 %; erstbest 12,9 €, zweitbest 11,7 €). Das AGV-Verfahren lässt bei beiden Verteilungen 81 % der Käufertypen mit negativem erwarteten Nutzen zurück. | `test_family_experiment_numbers`, `test_preset_bell` |
| Was passiert, wenn man einfach zum Mittelpunkt handelt? | Ehrlich melden lohnt nicht: gegen einen ehrlichen Verkäufer bietet der Käufer im Mittel nur 62 % (gleichverteilt) bzw. 70 % (Beta(2,2)) seines Werts; beim Wert 53,1 € bietet er 34,4 € und gewinnt 2,93 € erwarteten Nutzen. Beim zweitbesten Verfahren lohnt keine Lüge. | `test_preset_midpoint_lying`, `test_midpoint_best_bid_by_hand` |
| Wenig Überlappung | Käufer-Werte 60 bis 160 €: erstbest 61,1 €, zweitbest 58,5 € (95,8 %), Festpreis (78,1 €) 52,0 € (85,1 %). | `test_preset_little_overlap` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genau ein Käufer und ein Verkäufer** | In großen Märkten verschwindet der Verlust: McAfee (1992) erreicht dominante Ehrlichkeit durch Weglassen des am wenigsten lohnenden Handels; Rustichini/Satterthwaite/Williams (1994) zeigen Konvergenz zur Effizienz. | Doppelte Auktionen |
| **Werte und Kosten sind unabhängig** | Die Unmöglichkeit gilt für unabhängige Typen; bei geeignet korrelierten Typen kann sie entfallen. | – |
| **Beide kennen die Verteilung des anderen** | Der Mechanismus braucht gemeinsames Vorwissen; ohne es sind nur dominante Strategien (Festpreis) verlässlich. | Festpreis |
| **Quasilineare Nutzen, ein unteilbarer Slot** | Bei teilbaren Gütern oder Budgetgrenzen ändert sich die Lage. | – |
| **Diskretisierung** | Das Typengitter ist eine Näherung; grobe Gitter zeigen den Konflikt kaum (Befund oben). Der stetige Wert 9/64 : 1/6 ist Literatur, gemessen wird die Annäherung. | – |

Die Ausgleichsbedingung ist im **Erwartungswert** formuliert (die schwächste Form); die Unmöglichkeit gilt erst recht ex post. Der Mittelpunkt-Preis ist nur die Anreizrechnung gegen einen ehrlichen Verkäufer,
kein Gleichgewicht (dort würde auch der Verkäufer abweichen; das Gleichgewicht der doppelten Auktion nach Chatterjee/Samuelson ist nicht nachgebaut).

Verwandt: [moulin-demo](https://sebastianhanisch-moulin-demo.streamlit.app/) (Vorgänger: Kostenteilung mit privaten Werten), [maut-demo](https://sebastianhanisch-maut-demo.streamlit.app/) (Preise, die Externalitäten einpreisen),
auction-demo (Auktionen, VCG).

## Tests

Pytest-Suite (`pytest tests/ -v`): Mechanismen per Handrechnung (2 × 2 Typen), Eigenschaften (dominant ehrlich bei VCG und Festpreis für jeden Partnertyp, Bayes-Nash-ehrlich bei AGV und zweitbestem Verfahren),
zweitbestes LP gegen eine unabhängige dichte Formulierung, Konvergenz zu 1/6 und 9/64, Auswertung und Experimente (Form, Trends), Preset- und Permalink-Klemmen, AppTest-Rauchtests (jedes Preset,
Lügen-Widget, Extremwerte, Experimente auf Abruf) und `test_claims.py` (jede Zahl aus diesem README).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `ms_constants.py` | Bereiche, Regler-Grenzen, Experiment-Parameter |
| `ms_presets.py` | Permalink/Presets-Mechanik (Bereichsregler tragen Paare) |
| `ms_mechanisms.py` | Typengitter, VCG, AGV, Festpreis, zweitbestes LP, Mittelpunkt-Preis, Ehrlichkeits-Prüfungen |
| `ms_evaluation.py` | Analyse einer Einstellung, drei Experimente |
| `ms_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Mehrere Käufer und Verkäufer (doppelte Auktion, Handelsreduktion): eigenes Thema.
- Das Bayes-Nash-Gleichgewicht der doppelten Auktion (Chatterjee/Samuelson) und Verhandlungsmodelle.
- Ein PDF-Export – wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly, numpy und scipy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Spieltheorie: von Nash bis Myerson-Satterthwaite](https://sebastianhanisch.net/konzepte-spieltheorie.html).
