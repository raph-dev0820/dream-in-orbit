import argparse
from pathlib import Path

import joblib
import pandas as pd

from prepare_data import entrees


def main():
    # 1. lire les arguments de la commande
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dossier = Path(a.test_dir)

    # 2. charger le modèle entraîné
    modele, colonnes, circuits = joblib.load(Path(__file__).parent / "baseline.joblib")

    # 3. les heures à prédire, et les entrées de ces heures
    heures = pd.read_csv(dossier / "heures_a_predire.csv")["ut_ms"]
    grille = pd.Index(heures, name="ut_ms")
    X = entrees(dossier, grille).reindex(columns=colonnes, fill_value=0)

    # 4. prédire, puis écrire le fichier de réponses
    pred = pd.DataFrame(modele.predict(X), columns=circuits)
    pred.insert(0, "ut_ms", heures.values)
    pred.to_csv(a.out, index=False)
    print(f"{len(pred)} lignes écrites dans {a.out}")


if __name__ == "__main__":
    main()