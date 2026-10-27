import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gru_model import Modele
from prepare_data import H, entrees, power_horaire


def rever(modele, xt, yt, n_init):
    """Échauffement sur les heures réelles, puis le modèle continue seul."""
    with torch.no_grad():
        entree = torch.cat([xt[1:n_init], yt[: n_init - 1]], dim=1).unsqueeze(0)
        _, h = modele(entree)
        y_prec = yt[n_init - 1].view(1, 1, -1)
        sorties = []
        for t in range(n_init, len(xt)):
            entree = torch.cat([xt[t].view(1, 1, -1), y_prec], dim=2)
            y_prec, h = modele(entree, h)
            sorties.append(y_prec[0, 0].numpy())
    return np.array(sorties)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poids", default="poids")          # dossier des modèles
    ap.add_argument("--incertitude", default=None)       # fichier optionnel : désaccord entre modèles
    a = ap.parse_args()
    dossier = Path(a.test_dir)

    # 1. charger tous les modèles du dossier
    fichiers = sorted((Path(__file__).parent / a.poids).glob("*.pt"))
    assert fichiers, f"aucun modèle trouvé dans {a.poids}"
    sauvs = [torch.load(f) for f in fichiers]
    colonnes, circuits = sauvs[0]["colonnes"], sauvs[0]["circuits"]
    x_mean = pd.Series(sauvs[0]["x_mean"], index=colonnes)
    x_std = pd.Series(sauvs[0]["x_std"], index=colonnes)
    y_mean = pd.Series(sauvs[0]["y_mean"], index=circuits)
    modeles = []
    for s in sauvs:
        m = Modele(len(colonnes), len(circuits), cache=s["cache"])
        m.load_state_dict(s["etat"])
        m.eval()
        modeles.append(m)

    # 2. la grille d'heures : les 7 jours réels, puis les heures à prédire
    heures = pd.read_csv(dossier / "heures_a_predire.csv")["ut_ms"].to_numpy()
    p_init = power_horaire(dossier, "power_init.csv")
    debut = int(p_init.index.min())
    grille = pd.Index(range(debut, int(heures.max()) + H, H), name="ut_ms")
    n_init = int((grille < heures.min()).sum())

    # 3. les entrées normalisées, et la vraie consommation (seulement au début)
    X = entrees(dossier, grille).reindex(columns=colonnes, fill_value=0)
    xt = torch.tensor(((X - x_mean) / x_std).to_numpy(dtype=np.float32))
    Y = p_init.reindex(grille).reindex(columns=circuits) - y_mean
    yt = torch.tensor(Y.fillna(0).to_numpy(dtype=np.float32))

    # 4. chaque modèle rêve, puis on fait la moyenne et on mesure leur désaccord
    toutes = np.stack([rever(m, xt, yt, n_init) for m in modeles])
    moyenne = toutes.mean(axis=0) + y_mean.to_numpy()
    desaccord = toutes.std(axis=0)

    # 5. écrire le fichier de réponses
    pred = pd.DataFrame(moyenne, columns=circuits)
    assert len(pred) == len(heures), "le nombre d'heures prédites ne correspond pas"
    pred.insert(0, "ut_ms", heures)
    pred.to_csv(a.out, index=False)
    print(f"{len(pred)} lignes écrites dans {a.out} ({len(modeles)} modèles)")
    if a.incertitude:
        inc = pd.DataFrame(desaccord, columns=circuits)
        inc.insert(0, "ut_ms", heures)
        inc.to_csv(a.incertitude, index=False)


if __name__ == "__main__":
    main()