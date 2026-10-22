import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gru_model import Modele
from prepare_data import H, entrees, power_horaire


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dossier = Path(a.test_dir)

    # 1. charger le modèle entraîné
    sauv = torch.load(Path(__file__).parent / "gru.pt")
    colonnes, circuits = sauv["colonnes"], sauv["circuits"]
    x_mean = pd.Series(sauv["x_mean"], index=colonnes)
    x_std = pd.Series(sauv["x_std"], index=colonnes)
    y_mean = pd.Series(sauv["y_mean"], index=circuits)
    modele = Modele(len(colonnes), len(circuits))
    modele.load_state_dict(sauv["etat"])
    modele.eval()

    # 2. la grille d'heures : les 7 jours réels, puis les heures à prédire
    heures = pd.read_csv(dossier / "heures_a_predire.csv")["ut_ms"].to_numpy()
    p_init = power_horaire(dossier, "power_init.csv")
    debut = int(p_init.index.min())
    grille = pd.Index(range(debut, int(heures.max()) + H, H), name="ut_ms")
    n_init = int((grille < heures.min()).sum())   # nombre d'heures réelles connues

    # 3. les entrées normalisées, et la vraie consommation (seulement au début)
    X = entrees(dossier, grille).reindex(columns=colonnes, fill_value=0)
    xt = torch.tensor(((X - x_mean) / x_std).to_numpy(dtype=np.float32))
    Y = p_init.reindex(grille).reindex(columns=circuits) - y_mean
    yt = torch.tensor(Y.fillna(0).to_numpy(dtype=np.float32))

    with torch.no_grad():
        # 4. l'échauffement : le GRU lit les 7 jours réels pour remplir ses notes
        entree = torch.cat([xt[1:n_init], yt[: n_init - 1]], dim=1).unsqueeze(0)
        _, h = modele(entree)

        # 5. le rêve : il continue seul, en réutilisant ses propres prédictions
        y_prec = yt[n_init - 1].view(1, 1, -1)
        sorties = []
        for t in range(n_init, len(grille)):
            entree = torch.cat([xt[t].view(1, 1, -1), y_prec], dim=2)
            y_prec, h = modele(entree, h)
            sorties.append(y_prec[0, 0].numpy())

    # 6. remettre la consommation à sa vraie échelle et écrire le fichier
    valeurs = np.array(sorties) + y_mean.to_numpy()
    pred = pd.DataFrame(valeurs, columns=circuits)
    assert len(pred) == len(heures), "le nombre d'heures prédites ne correspond pas"
    pred.insert(0, "ut_ms", heures)
    pred.to_csv(a.out, index=False)
    print(f"{len(pred)} lignes écrites dans {a.out}")


if __name__ == "__main__":
    main()