import argparse
from pathlib import Path

import pandas as pd


def main():
    # 1. lire les arguments donnés dans la commande
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)  # dossier de test
    ap.add_argument("--out", required=True)       # fichier de sortie
    a = ap.parse_args()

    # 2. charger les 7 jours réels et la liste des heures à prédire
    dossier = Path(a.test_dir)
    init = pd.read_csv(dossier / "power_init.csv")
    heures = pd.read_csv(dossier / "heures_a_predire.csv")

    # 3. les circuits = toutes les colonnes sauf l'heure
    circuits = [c for c in init.columns if c != "ut_ms"]

    # 4. la moyenne de chaque circuit sur les 7 jours (0 si aucune mesure)
    moyennes = init[circuits].mean().fillna(0)

    # 5. un tableau : une ligne par heure, la même valeur pour chaque circuit
    pred = pd.DataFrame({"ut_ms": heures["ut_ms"]})
    for c in circuits:
        pred[c] = moyennes[c]

    # 6. écrire le fichier de réponses
    pred.to_csv(a.out, index=False)
    print(f"{len(pred)} lignes écrites dans {a.out}")


if __name__ == "__main__":
    main()