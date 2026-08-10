"""Reconstruit les zips de soumission arXiv (v2, avec les citations ajoutees).

IMPORTANT : utilise zipfile avec des arcname en SLASHS EXPLICITES -- ne PAS
utiliser Compress-Archive de PowerShell, qui ecrit des antislashs comme
separateurs de chemin dans l'archive, ce qu'arXiv refuse
("appears to use backslashes as path separators"). Piege deja rencontre et
diagnostique lors de la premiere soumission.
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

BUNDLES = {
    "conditional_collapse_theory_submission.zip": [
        ("conditional_collapse_theory.tex", "conditional_collapse_theory.tex"),
        ("conditional_collapse_theory.pdf", "conditional_collapse_theory.pdf"),
        ("figures/fig_marker_seed44_polarity.pdf", "figures/fig_marker_seed44_polarity.pdf"),
        ("figures/fig_marker_interaction_oos.pdf", "figures/fig_marker_interaction_oos.pdf"),
    ],
    "crosslayer_interaction_qwen_submission.zip": [
        ("crosslayer_interaction_qwen.tex", "crosslayer_interaction_qwen.tex"),
        ("crosslayer_interaction_qwen.pdf", "crosslayer_interaction_qwen.pdf"),
        ("figures/fig_qwen_ioi_circuit_frequency.pdf", "figures/fig_qwen_ioi_circuit_frequency.pdf"),
        ("figures/fig_qwen_ioi_recovery_trajectory.pdf", "figures/fig_qwen_ioi_recovery_trajectory.pdf"),
    ],
}

for zip_name, entries in BUNDLES.items():
    zip_path = os.path.join(HERE, zip_name)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_rel, arcname in entries:
            src = os.path.join(HERE, src_rel.replace("/", os.sep))
            if not os.path.isfile(src):
                raise SystemExit("MANQUANT : " + src)
            zf.write(src, arcname=arcname)  # arcname en slashs, jamais os.sep
    print("Ecrit :", zip_name)

# Verification : aucun antislash residuel dans les octets bruts de l'archive
for zip_name in BUNDLES:
    zip_path = os.path.join(HERE, zip_name)
    with open(zip_path, "rb") as f:
        data = f.read()
    bad = [n for n in zipfile.ZipFile(zip_path).namelist() if "\\" in n]
    raw_bad = data.count(b"figures\\")
    print("  %-52s entrees=%d  arcnames_avec_antislash=%d  octets_'figures\\\\'=%d"
          % (zip_name, len(zipfile.ZipFile(zip_path).namelist()), len(bad), raw_bad))
    for n in zipfile.ZipFile(zip_path).namelist():
        print("      -", n)
