import os
import re
import subprocess
import sys

PY = [sys.executable]  # le même Python que celui qui lance ce script
N_MODELES = 3          # modèles par essai
PARALLELE = 3          # essais lancés en même temps
GRAINE_DEBUT = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # graines : 1-3, 4-6...
VERITE = "validation/verite_horaire.csv"
HEURES = "validation/test/heures_a_predire.csv"

# chaque essai : les colonnes (ou débuts de noms) qu'on retire
ESSAIS = {
    "reference": [],
    "sans_eclipses": ["evtf_umbre", "evtf_penombre", "lt_eclipseduration_min"],
    "sans_soleil": ["saaf_"],
    "sans_commandes": ["dmop_"],
    "sans_pointage": ["ftl_"],
}

env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3")


def dossier_poids(nom):
    return f"poids_abl_{nom}_g{GRAINE_DEBUT}"


def lancer(nom, sans):
    cmd = PY + ["train.py", "--annees", "annee1", "--n_modeles", str(N_MODELES),
                "--graine_debut", str(GRAINE_DEBUT), "--out_dir", dossier_poids(nom)]
        cmd += ["--sans"] + sans
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL)


print("graines :", GRAINE_DEBUT, "à", GRAINE_DEBUT + N_MODELES - 1, flush=True)

# 1. entraîner chaque essai, quelques-uns à la fois
noms = list(ESSAIS)
for i in range(0, len(noms), PARALLELE):
    groupe = noms[i:i + PARALLELE]
    print("entraînement de :", groupe, flush=True)
    processus = [lancer(nom, ESSAIS[nom]) for nom in groupe]
    for p in processus:
        p.wait()

# 2. noter chaque essai avec le script officiel
resultats = {}
for nom in noms:
    sortie = f"pred_abl_{nom}_g{GRAINE_DEBUT}.csv"
    subprocess.run(PY + ["predict.py", "--test_dir", "validation/test", "--out", sortie,
                         "--poids", dossier_poids(nom)], check=True, stdout=subprocess.DEVNULL)
    res = subprocess.run(PY + ["evaluate.py", "--pred", sortie, "--verite", VERITE, "--heures", HEURES],
                         capture_output=True, text=True)
    resultats[nom] = float(re.search(r"RMSE = ([0-9.]+)", res.stdout).group(1))
    print(nom, resultats[nom], flush=True)

# 3. le tableau final
ref = resultats["reference"]
lignes = ["| Essai | RMSE (moyenne de 3 modèles) | Écart à la référence |", "|---|---|---|"]
for nom, r in resultats.items():
    lignes.append(f"| {nom} | {r:.4f} | {r - ref:+.4f} |")
texte = "\n".join(lignes)
print(texte)
open(f"ablations_g{GRAINE_DEBUT}.md", "w", encoding="utf-8").write(texte)