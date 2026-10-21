import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from prepare_data import H, entrees, power_horaire

dossier = "annee1"

# 1. la consommation (ce qu'il faut prédire) et la grille d'heures
p = power_horaire(dossier)
grille = pd.Index(range(int(p.index.min()), int(p.index.max()) + H, H), name="ut_ms")
p = p.reindex(grille)

# 2. les entrées (le planning et le contexte)
X = entrees(dossier, grille)

# 3. on enlève les heures sans mesure de consommation
ok = p.notna().all(axis=1)
print(ok.sum(), "heures pour s'entraîner,", (~ok).sum(), "heures écartées")

# 4. entraînement
modele = ExtraTreesRegressor(n_estimators=50, min_samples_leaf=30, n_jobs=-1, random_state=0)
modele.fit(X[ok], p[ok])

# 5. sauvegarde du modèle, avec les noms des colonnes
joblib.dump((modele, list(X.columns), list(p.columns)), "baseline.joblib")
print("modèle sauvegardé")