# Dream in Orbit : une IA qui simule un satellite

Simulation, heure par heure, de la consommation électrique de 33 circuits d'un satellite, à partir de son seul planning et de ses 7 premiers jours réels. Le modèle est un ensemble de 6 réseaux GRU (PyTorch).

## Installation

- Python 3.13
- `pip install -r requirements.txt`
- Place les dossiers `annee1/` et `annee2/` du kit à la racine du projet, à côté de `train.py`.

## Utilisation

Entraîner (6 modèles, poids écrits dans `poids/`) :

    python train.py

Prédire :

    python predict.py --test_dir DOSSIER --out predictions.csv

Options de `predict.py` : `--poids DOSSIER` (modèles utilisés, `poids` par défaut) et `--incertitude fichier.csv` (désaccord entre les modèles).

## Valider (même procédure que l'évaluation)

    python make_validation.py --annee annee2 --out validation
    python predict.py --test_dir validation/test --out pred.csv
    python evaluate.py --pred pred.csv --verite validation/verite_horaire.csv --heures validation/test/heures_a_predire.csv

Les poids de `poids/` ont été entraînés sur `annee1` **et** `annee2` : ce score serait trop optimiste. Pour un score honnête (entraînement sur `annee1` seule) :

    python train.py --annees annee1 --out_dir poids_val_sd
    python predict.py --test_dir validation/test --out pred.csv --poids poids_val_sd
    python evaluate.py --pred pred.csv --verite validation/verite_horaire.csv --heures validation/test/heures_a_predire.csv

## Résultats (RMSE sur annee2, modèles entraînés sur annee1)

| Modèle | RMSE |
|---|---|
| Constante (moyenne des 7 jours réels) | 0,1116 |
| Forêt d'arbres, sans mémoire (baseline) | 0,0936 |
| Un seul GRU avec rêve | de 0,096 à 0,110 selon la graine |
| Ensemble de 6 GRU, toutes les entrées | 0,0909 |
| **Ensemble de 6 GRU, sans les commandes (modèle final)** | **0,0869** |

Les réglages ont été choisis en regardant `annee2` : le score sur une année nouvelle sera probablement moins bon.

## Choix techniques

- Les 6 fichiers du kit sont ramenés à une grille horaire : consommation moyenne par heure, part de l'heure passée dans chaque mode de pointage, en éclipse ou en communication, angles du Soleil, données journalières.
- Modèle : GRU de 128 unités. À chaque heure, il reçoit les entrées de l'heure et la consommation de l'heure précédente.
- Entraînement au « rêve » (scheduled sampling) : pendant l'entraînement, la part de prédictions du modèle utilisées à la place des vraies valeurs monte de 0 à 100 %.
- Au test : le modèle lit d'abord les 7 jours réels, puis simule seul le reste.
- Ensemble de 6 modèles (graines différentes) : on moyenne leurs prédictions, et leur désaccord sert d'incertitude (corrélation de 0,69 avec l'erreur sur annee2).
- Les commandes (`dmop`) sont retirées : l'ablation montre qu'elles n'apportent rien de mesurable.

## Ablations (écart de RMSE quand on retire un groupe d'entrées, moyenne de 2 séries de 3 modèles)

| Entrées retirées | RMSE moyen | Écart |
|---|---|---|
| aucune | 0,0928 | |
| éclipses | 0,1037 | +0,0109 |
| pointage et communications | 0,0990 | +0,0062 |
| Soleil | 0,0938 | +0,0010 |
| commandes | 0,0896 | -0,0032 |

Reproduction : `python ablations.py` puis `python ablations.py 4` (graines 1 à 3, puis 4 à 6).

## Démo

    python predict.py --test_dir validation/test --out pred_sd.csv --poids poids_val_sd --incertitude sigma_sd.csv
    python demo.py

Ouvre ensuite `demo.html` dans un navigateur : réalité, rêve et désaccord des modèles, avec un menu par circuit.

## Contenu du dépôt

- `prepare_data.py` : construit le tableau d'entrées horaire.
- `gru_model.py` : le réseau.
- `train.py`, `predict.py` : entraînement et prédiction.
- `evaluate.py`, `make_validation.py` : scripts du kit, non modifiés.
- `ablations.py`, `ablations.md`, `ablations_g4.md` : ablations.
- `demo.py` : démo.
- `poids/` : poids du modèle final.
- `journal.md` : journal des expériences.
- `train_baseline.py`, `predict_constante.py`, `train_gru.py`, `gru.pt`, `poids_val/` : anciens essais, gardés pour la trace (`poids_val/` : modèles avec commandes, RMSE 0,0909).

## Reproductibilité

Les graines sont fixées. Sur ma machine, relancer avec les mêmes graines redonne exactement les mêmes résultats ; entre machines, de petits écarts sont possibles. Durée de `python train.py` : [à compléter].

## Limites

- Le modèle lisse les pics de consommation : il les sous-estime.
- Les réglages ont été choisis sur `annee2`.
- Le désaccord entre modèles est un signal d'alerte, pas un intervalle de confiance calibré.
