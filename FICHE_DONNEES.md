# Fiche des données : Dream in Orbit

## Source
- Fournies par l'organisation dans le kit du sujet (dossiers `annee1/` et `annee2/`). Aucune donnée extérieure n'a été utilisée.
- Nature : télémétrie réelle d'un satellite en opération, rendue publique par une agence spatiale (d'après l'énoncé). D'après les distances et les noms de colonnes (`sunmars_km`, `MAR_UMBRA`), le satellite orbite autour de Mars.
- Agence et provenance exacte : [à confirmer avec l'organisation].
- Licence : [à confirmer avec l'organisation].

## Composition (`annee1`)
Période : du 22 août 2008 au 9 juillet 2010 (687 jours).

| Fichier | Contenu | Taille | Rôle |
|---|---|---|---|
| `power.csv` | courant de 33 circuits (`NPWDxxxx`), 1 à 2 mesures par minute | 1 830 121 lignes | ce qu'on prédit |
| `dmop.csv` | commandes envoyées aux sous-systèmes | 187 126 lignes, 9 280 codes différents | actions (retirées du modèle final) |
| `ftl.csv` | modes de pointage et communications | 19 571 périodes, 14 types | actions |
| `saaf.csv` | angles du Soleil par rapport au satellite | 1 261 290 lignes | contexte |
| `evtf.csv` | événements (éclipses, passages au plus près de la planète, contacts radio) | 76 776 lignes, 118 familles | contexte |
| `ltdata.csv` | chiffres journaliers (distances, durée d'éclipse) | 687 lignes | contexte |

`annee2` a la même structure. Sa période de validation va du 17 juillet 2010 au 26 mai 2012 : 16 319 heures à prédire après les 7 jours réels donnés au départ.

## Découpage entraînement / validation / test
- Entraînement : `annee1`. Validation : `annee2`. Test : une troisième période, cachée par l'organisation.
- Le découpage suit le temps, jamais au hasard : la validation vient après l'entraînement.
- Modèle final : entraîné sur `annee1` et `annee2`, avec les réglages choisis sur la validation.
- Risque de fuite : aucun recouvrement temporel entre entraînement et validation. Les moyennes et écarts-types de normalisation sont calculés sur l'entraînement uniquement. Les réglages ont en revanche été choisis en regardant `annee2` : son score est donc optimiste.

## Prétraitement
- Tout est ramené à une grille horaire : consommation moyenne par heure et par circuit, part de l'heure passée dans chaque mode de pointage, en éclipse ou en communication, moyenne horaire des angles du Soleil, valeur journalière répétée sur 24 heures.
- Les commandes (9 280 codes) étaient regroupées par leurs 4 premières lettres (26 familles) ; elles sont retirées du modèle final.

## Valeurs manquantes
- `annee1` : 32 heures sans aucune mesure de consommation sur 16 488 (0,2 %), écartées de l'entraînement.
- `annee2` : sur 16 319 heures à prédire, 246 n'ont pas de vérité (1,5 %), ignorées par le script de notation.

## Biais et limites
- Un seul satellite et environ 3,8 ans de données : le modèle ne généralise pas à un autre engin.
- Le satellite évolue avec le temps (vieillissement, changements d'opérations) : un écart entre les années est possible.
- 10 circuits sur 33 portent environ 97 % de l'erreur, dont deux (NPWD2551 et NPWD2451) environ 58 %.
- Les 17 circuits presque plats sont prédits avec une valeur quasi fixe.
- Le sens de beaucoup de codes (`dmop`, `evtf`) n'est pas documenté dans le kit : ils ont été traités par comptage et regroupement, pas par interprétation.
