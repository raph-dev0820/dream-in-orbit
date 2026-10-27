import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gru_model import Modele
from prepare_data import H, entrees, power_horaire

ap = argparse.ArgumentParser()
ap.add_argument("--annees", nargs="+", default=["annee1", "annee2"])  # dossiers du kit utilisés
ap.add_argument("--n_modeles", type=int, default=6)
ap.add_argument("--etapes", type=int, default=2000)
ap.add_argument("--cache", type=int, default=128)
ap.add_argument("--bruit", type=float, default=0.05)
ap.add_argument("--p_max", type=float, default=1.0)
ap.add_argument("--graine_debut", type=int, default=1)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--out_dir", default="poids")
a = ap.parse_args()

if a.threads:
    torch.set_num_threads(a.threads)

L = 168  # longueur d'un morceau d'entraînement : 7 jours en heures

# 1. charger chaque année : les entrées et la consommation, une ligne par heure
donnees = []
for d in a.annees:
    p = power_horaire(d)
    grille = pd.Index(range(int(p.index.min()), int(p.index.max()) + H, H), name="ut_ms")
    donnees.append((entrees(d, grille), p.reindex(grille)))

colonnes = sorted(set().union(*[set(X.columns) for X, _ in donnees]))
circuits = list(donnees[0][1].columns)
donnees = [(X.reindex(columns=colonnes, fill_value=0), Y[circuits]) for X, Y in donnees]

# 2. mettre les nombres à une échelle comparable (sur toutes les années ensemble)
X_tout = pd.concat([X for X, _ in donnees])
Y_tout = pd.concat([Y for _, Y in donnees])
x_mean = X_tout.mean()
x_std = X_tout.std()
x_std[x_std < 1e-6] = 1.0
y_mean = Y_tout.mean()

# 3. les tenseurs de chaque année : entrées, consommation, masque (1 = vraie mesure)
annees_t = []
for X, Y in donnees:
    Xt = torch.tensor(((X - x_mean) / x_std).to_numpy(dtype=np.float32))
    Yt = torch.tensor((Y - y_mean).fillna(0).to_numpy(dtype=np.float32))
    Mt = torch.tensor(Y.notna().all(axis=1).to_numpy(dtype=np.float32))
    annees_t.append((Xt, Yt, Mt))
longueurs = np.array([len(Xt) for Xt, _, _ in annees_t], dtype=float)
proba = longueurs / longueurs.sum()


def lot(taille=32):
    """Tire au hasard 32 morceaux de 7 jours, chacun dans une seule année."""
    xs, ys, ms, y0 = [], [], [], []
    for k in np.random.choice(len(annees_t), size=taille, p=proba):
        Xk, Yk, Mk = annees_t[k]
        d = np.random.randint(1, len(Xk) - L)
        xs.append(Xk[d:d + L])
        ys.append(Yk[d:d + L])
        ms.append(Mk[d:d + L])
        y0.append(Yk[d - 1])
    return torch.stack(xs), torch.stack(ys), torch.stack(ms), torch.stack(y0)


def entrainer(graine):
    torch.manual_seed(graine)
    np.random.seed(graine)
    modele = Modele(len(colonnes), len(circuits), cache=a.cache)
    opt = torch.optim.Adam(modele.parameters(), lr=2e-3)
    for etape in range(1, a.etapes + 1):
        # p monte doucement de 0 à p_max pendant les 70 % premiers pas
        p = a.p_max * min(1.0, etape / (0.7 * a.etapes))
        xs, cible, m, y_prec = lot()
        y_prec = y_prec + a.bruit * torch.randn_like(y_prec)
        h = None
        sorties = []
        for t in range(L):
            entree = torch.cat([xs[:, t], y_prec], dim=1).unsqueeze(1)
            s, h = modele(entree, h)
            s = s[:, 0]
            sorties.append(s)
            # pour l'heure suivante : la vraie valeur, ou (avec probabilité p) ma prédiction
            vraie = cible[:, t] + a.bruit * torch.randn_like(s)
            prend_pred = (torch.rand(s.shape[0], 1) < p).float()
            y_prec = prend_pred * s.detach() + (1 - prend_pred) * vraie
        sortie = torch.stack(sorties, dim=1)
        erreur = (((sortie - cible) ** 2).mean(dim=2) * m).sum() / m.sum()
        opt.zero_grad()
        erreur.backward()
        torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
        opt.step()
        if etape % 500 == 0:
            print("modèle", graine, "étape", etape, "RMSE d'entraînement :", round(erreur.sqrt().item(), 4))
    return modele


# 4. entraîner les modèles, un par graine, et sauvegarder les poids de chacun
dossier_sortie = Path(a.out_dir)
dossier_sortie.mkdir(exist_ok=True)
for k in range(a.n_modeles):
    graine = a.graine_debut + k
    modele = entrainer(graine)
    torch.save({
        "etat": modele.state_dict(),
        "colonnes": colonnes,
        "circuits": circuits,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "cache": a.cache,
    }, dossier_sortie / f"gru_{graine}.pt")
    print("modèle", graine, "sauvegardé")