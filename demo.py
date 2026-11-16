import pandas as pd
import plotly.graph_objects as go

# 1. charger : la prédiction (moyenne des modèles), leur désaccord, et la réalité
pred = pd.read_csv("pred_sd.csv").set_index("ut_ms")
sigma = pd.read_csv("sigma_sd.csv").set_index("ut_ms")
vrai = pd.read_csv("validation/verite_horaire.csv").set_index("ut_ms")

# 2. garder les heures où la réalité existe, et mettre de vraies dates
communes = vrai.index.intersection(pred.index)
pred, sigma, vrai = pred.loc[communes], sigma.loc[communes], vrai.loc[communes, pred.columns]
for d in (pred, sigma, vrai):
    d.index = pd.to_datetime(d.index, unit="ms")

# 3. les 8 circuits les plus difficiles (plus grosse erreur)
erreur = ((pred - vrai) ** 2).mean() ** 0.5
circuits = list(erreur.sort_values(ascending=False).index[:8])


def lisse(df, c):
    """Moyenne sur 3 heures, pour garder un fichier léger et lisible."""
    return df[c].resample("3h").mean()


# 4. pour chaque circuit : 4 courbes (borne haute, borne basse, réalité, rêve)
fig = go.Figure()
for i, c in enumerate(circuits):
    p, s, v = lisse(pred, c), lisse(sigma, c), lisse(vrai, c)
    visible = (i == 0)
    bas = (p - 2 * s).clip(lower=0)  # une consommation n'est jamais négative
    fig.add_trace(go.Scatter(x=p.index, y=p + 2 * s, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip", visible=visible))
    fig.add_trace(go.Scatter(x=p.index, y=bas, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(0,100,255,0.2)",
                             name="désaccord des modèles (±2σ)", visible=visible))
    # la réalité d'abord (derrière), puis le rêve par-dessus
    fig.add_trace(go.Scatter(x=v.index, y=v, mode="lines", line=dict(color="black", width=1),
                             name="réalité", visible=visible))
    fig.add_trace(go.Scatter(x=p.index, y=p, mode="lines", line=dict(color="blue", width=2),
                             name="rêve (moyenne des modèles)", visible=visible))

# 5. un menu pour changer de circuit
boutons = []
for i, c in enumerate(circuits):
    visibles = [False] * (4 * len(circuits))
    for k in range(4):
        visibles[4 * i + k] = True
    boutons.append(dict(label=f"{c} (RMSE {erreur[c]:.3f})", method="update",
                        args=[{"visible": visibles}, {"title": f"Rêve contre réalité : {c}"}]))

# 6. un menu pour changer la durée affichée (on part du début de la validation)
debut = pred.index.min()
fin = pred.index.max()
durees = [("2 semaines", pd.Timedelta(days=14)), ("1 mois", pd.Timedelta(days=30)),
          ("2 mois", pd.Timedelta(days=60)), ("tout", fin - debut)]
boutons_duree = [dict(label=nom, method="relayout",
                      args=[{"xaxis.range": [debut, debut + d]}]) for nom, d in durees]

fig.update_layout(
    title=f"Rêve contre réalité : {circuits[0]}",
    updatemenus=[dict(buttons=boutons, x=0, y=1.15),
                 dict(buttons=boutons_duree, x=0.45, y=1.15)],
    xaxis_title="date (période de validation)",
    yaxis_title="consommation (moyenne sur 3 h)",
    xaxis_range=[debut, debut + pd.Timedelta(days=30)],  # au départ : 1 mois
    yaxis_rangemode="tozero",
    xaxis_rangeslider_visible=True,
)
fig.write_html("demo.html", include_plotlyjs=True)
print("demo.html écrit")