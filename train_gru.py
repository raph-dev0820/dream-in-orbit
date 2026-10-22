import numpy as np
import pandas as pd
import torch

from gru_model import Modele
from prepare_data import H, entrees, power_horaire

torch.manual_seed(0)
np.random.seed(0)

L = 168        # longueur d'un morceau d'entraînement : 7 jours en heures
BRUIT = 0.05   # petit bruit sur la "consommation de l'heure d'avant" (voir plus bas)
ETAPES = 2000

# 1. charger annee1 : la consommation et les entrées, une ligne par heure
dossier = "annee1"
p = power_horaire(dossier)
grille = pd.Index(range(int(p.index.min()), int(p.index.max()) + H, H), name="ut_ms")
Y = p.reindex(grille)
X = entrees(dossier, grille)
colonnes, circuits = list(X.columns), list(Y.columns)

# 2. mettre les nombres à une échelle comparable
x_mean = X.mean()
x_std = X.std()
x_std[x_std < 1e-6] = 1.0           # évite de diviser par presque zéro
y_mean = Y.mean()
Xn = ((X - x_mean) / x_std).to_numpy(dtype=np.float32)
Yn = (Y - y_mean).fillna(0).to_numpy(dtype=np.float32)   # trous de mesure = 0
masque = Y.notna().all(axis=1).to_numpy(dtype=np.float32)  # 1 = vraie mesure

# 3. les tableaux de nombres (des "tenseurs") pour PyTorch
Xt, Yt, Mt = torch.tensor(Xn), torch.tensor(Yn), torch.tensor(masque)


def lot(taille=32):
    """Tire au hasard 32 morceaux de 7 jours."""
    debuts = np.random.randint(1, len(Xt) - L, size=taille)
    idx = torch.tensor(debuts[:, None] + np.arange(L)[None, :])   # les heures choisies
    y_avant = Yt[idx - 1]                                          # consommation de l'heure d'avant
    y_avant = y_avant + BRUIT * torch.randn_like(y_avant)          # + un peu de bruit
    entree = torch.cat([Xt[idx], y_avant], dim=2)
    return entree, Yt[idx], Mt[idx]


# 4. le modèle et l'optimiseur (celui qui corrige le modèle)
modele = Modele(Xt.shape[1], Yt.shape[1])
opt = torch.optim.Adam(modele.parameters(), lr=2e-3)

# 5. la boucle d'entraînement : essayer, mesurer l'erreur, corriger, recommencer
for etape in range(1, ETAPES + 1):
    entree, cible, m = lot()
    sortie, _ = modele(entree)
    erreur = (((sortie - cible) ** 2).mean(dim=2) * m).sum() / m.sum()
    opt.zero_grad()
    erreur.backward()
    torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
    opt.step()
    if etape % 200 == 0:
        print(etape, "RMSE d'entraînement :", round(float(erreur.sqrt()), 4))

# 6. sauvegarder le modèle (poids) et ce qu'il faut pour l'utiliser
torch.save({
    "etat": modele.state_dict(),
    "colonnes": colonnes,
    "circuits": circuits,
    "x_mean": x_mean.tolist(),
    "x_std": x_std.tolist(),
    "y_mean": y_mean.tolist(),
}, "gru.pt")
print("modèle sauvegardé dans gru.pt")