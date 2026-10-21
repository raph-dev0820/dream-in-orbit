import pandas as pd

H = 3600 * 1000  # une heure, en millisecondes


def heure(ut_ms):
    """Arrondit un instant au début de son heure (14h35 devient 14h00)."""
    return (ut_ms // H) * H


def power_horaire(dossier):
    """Consommation : moyenne par heure de chaque circuit."""
    p = pd.read_csv(f"{dossier}/power.csv")
    p["ut_ms"] = heure(p["ut_ms"])
    return p.groupby("ut_ms").mean()


def saaf_horaire(dossier):
    """Angles du Soleil : moyenne par heure."""
    s = pd.read_csv(f"{dossier}/saaf.csv")
    s["ut_ms"] = heure(s["ut_ms"])
    return s.groupby("ut_ms").mean().add_prefix("saaf_")


def ltdata_horaire(dossier, grille):
    """Chiffres du jour : on répète la valeur du jour sur chaque heure."""
    lt = pd.read_csv(f"{dossier}/ltdata.csv").set_index("ut_ms").sort_index()
    return lt.reindex(grille, method="ffill").add_prefix("lt_")


def dmop_horaire(dossier, grille):
    """Ordres : combien d'ordres de chaque famille (4 premières lettres) par heure."""
    d = pd.read_csv(f"{dossier}/dmop.csv")
    d["ut_ms"] = heure(d["ut_ms"])
    d["debut"] = d["subsystem"].str[:4]
    t = d.groupby(["ut_ms", "debut"]).size().unstack(fill_value=0)
    return t.reindex(grille, fill_value=0).add_prefix("dmop_")


def parts_par_heure(debuts, fins, etiquettes, grille):
    """Pour chaque période et chaque heure : la part de l'heure couverte, par étiquette."""
    couvert = {}  # (heure, étiquette) -> millisecondes couvertes
    for b, f, nom in zip(debuts, fins, etiquettes):
        h = heure(b)
        while h < f:
            recouvrement = min(f, h + H) - max(b, h)
            if recouvrement > 0:
                couvert[(h, nom)] = couvert.get((h, nom), 0) + recouvrement
            h += H
    s = pd.Series(couvert)
    t = (s.unstack(fill_value=0) / H).clip(upper=1)
    return t.reindex(grille, fill_value=0)


def ftl_horaire(dossier, grille):
    """Pointage : part de l'heure dans chaque mode, et en communication."""
    f = pd.read_csv(f"{dossier}/ftl.csv")
    modes = parts_par_heure(f["utb_ms"], f["ute_ms"], f["type"], grille).add_prefix("ftl_")
    c = f[f["flagcomms"]]
    comms = parts_par_heure(c["utb_ms"], c["ute_ms"], ["comms"] * len(c), grille)
    return pd.concat([modes, comms.add_prefix("ftl_")], axis=1)


GENRES = {
    "umbre": r"^MAR_UMBRA",
    "penombre": r"^MAR_PENUMBRA",
    "occ_mars": r"^OCC_MARS_(?:START|END)",
    "occ_mars_km": r"^OCC_MARS_\d+KM_(?:START|END)",
}


def evtf_horaire(dossier, grille):
    """Événements : on associe chaque START à son END, puis part de l'heure couverte."""
    e = pd.read_csv(f"{dossier}/evtf.csv").sort_values("ut_ms")
    debuts, fins, noms = [], [], []
    for nom, motif in GENRES.items():
        x = e[e["description"].str.contains(motif, regex=True)]
        debut = None
        for t, desc in zip(x["ut_ms"], x["description"]):
            if "_START" in desc:
                debut = t
            elif "_END" in desc and debut is not None:
                debuts.append(debut)
                fins.append(t)
                noms.append(nom)
                debut = None
    return parts_par_heure(debuts, fins, noms, grille).add_prefix("evtf_")

def entrees(dossier, grille):
    """Assemble toutes les entrées : une ligne par heure, 55 colonnes."""
    x = pd.concat([
        dmop_horaire(dossier, grille),
        ftl_horaire(dossier, grille),
        evtf_horaire(dossier, grille),
        saaf_horaire(dossier).reindex(grille),
        ltdata_horaire(dossier, grille),
    ], axis=1)
    return x.ffill().fillna(0)  # s'il manque une valeur, on reprend la précédente

if __name__ == "__main__":
    dossier = "annee1"
    p = power_horaire(dossier)

    # la grille : toutes les heures, sans trou, du début à la fin
    grille = pd.Index(range(int(p.index.min()), int(p.index.max()) + H, H), name="ut_ms")
    p = p.reindex(grille)

    entrees = pd.concat([
        dmop_horaire(dossier, grille),
        ftl_horaire(dossier, grille),
        evtf_horaire(dossier, grille),
        saaf_horaire(dossier).reindex(grille),
        ltdata_horaire(dossier, grille),
    ], axis=1)

    print("consommation :", p.shape, "| entrées :", entrees.shape)
    print("cases vides dans les entrées :", int(entrees.isna().sum().sum()))
    print("heures avec ombre complète :", int((entrees["evtf_umbre"] > 0).sum()))
    modes = [c for c in entrees.columns if c.startswith("ftl_") and c != "ftl_comms"]
    print("somme des parts de modes par heure (doit être au plus ~1) :")
    print(entrees[modes].sum(axis=1).describe())
    vides = p.index[p.isna().all(axis=1)]
    print("premières heures sans consommation :", list(pd.to_datetime(vides[:3], unit="ms")))