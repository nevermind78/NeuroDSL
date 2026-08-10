"""Reconstruit les zips de soumission arXiv pour math1 (2607.16568) et math2 (2607.18323).

IMPORTANT : utilise zipfile avec des arcname en SLASHS EXPLICITES -- ne PAS
utiliser Compress-Archive de PowerShell, qui ecrit des antislashs comme
separateurs de chemin dans l'archive, ce qu'arXiv refuse
("appears to use backslashes as path separators").

Le manifeste reproduit exactement celui des v1 deja acceptees : .tex + figures,
sans le PDF compile (arXiv construit le PDF depuis la source).
Ecrit vers des noms _v2 afin de laisser intacts les zips v1.
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

BUNDLES = {
    "math1_arxiv_submission_v2.zip": [
        "math1.tex",
        "figures/fig_e2_plasticity.pdf",
        "figures/fig_e3_locality.pdf",
        "figures/fig_e6_gate_dynamics.pdf",
        "figures/fig_e8_warmstart.pdf",
        "figures/fig_f3_continuity.pdf",
        "figures/fig_utility_paired.pdf",
    ],
    "math2_arxiv_submission_v2.zip": [
        "math2.tex",
    ],
}

for zip_name, arcnames in BUNDLES.items():
    zip_path = os.path.join(HERE, zip_name)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname in arcnames:
            src = os.path.join(HERE, arcname.replace("/", os.sep))
            if not os.path.isfile(src):
                raise SystemExit("MANQUANT : " + src)
            zf.write(src, arcname=arcname)  # arcname en slashs, jamais os.sep
    print("Ecrit :", zip_name)

# Verification : aucun antislash ni dans les arcnames ni dans les octets bruts
for zip_name in BUNDLES:
    zip_path = os.path.join(HERE, zip_name)
    with open(zip_path, "rb") as f:
        raw = f.read()
    names = zipfile.ZipFile(zip_path).namelist()
    bad = [n for n in names if "\\" in n]
    print("  %-34s entrees=%d  arcnames_avec_antislash=%d  octets_'figures\\\\'=%d"
          % (zip_name, len(names), len(bad), raw.count(b"figures\\")))
    for n in names:
        print("      -", n)
