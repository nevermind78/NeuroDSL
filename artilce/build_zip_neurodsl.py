"""Construit le zip de soumission arXiv pour NeuroDSL.tex (article 1).

IMPORTANT -- LES SEPARATEURS DE CHEMIN : on utilise zipfile avec des `arcname`
en SLASHS EXPLICITES. Ne PAS utiliser Compress-Archive de PowerShell, qui ecrit
des ANTISLASHS comme separateurs dans l'archive : arXiv refuse alors le depot
avec "appears to use backslashes as path separators". Piege deja rencontre sur
une soumission precedente de ce projet.

PARTICULARITE DE CE PAPIER : `NeuroDSL.tex` n'a AUCUN \\graphicspath et ses deux
figures vivent a la RACINE du depot (`figures/`), pas dans `artilce/`. Il ne
compile donc que depuis la racine. Dans l'archive on remet le .tex a la racine et
les figures sous `figures/`, ce qui reproduit exactement la disposition attendue
et rend l'archive autonome.

MANIFESTE : .tex + figures, sans le PDF compile -- c'est ce qu'utilisaient les
soumissions math1/math2 acceptees ; arXiv construit le PDF depuis la source.
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))          # .../artilce
ROOT = os.path.dirname(HERE)                               # .../NeuroDSL

ZIP_NAME = "NeuroDSL_submission.zip"

# (chemin source absolu, arcname dans l'archive -- TOUJOURS en slashs)
ENTRIES = [
    (os.path.join(HERE, "NeuroDSL.tex"),                    "NeuroDSL.tex"),
    (os.path.join(ROOT, "figures", "sinus_tanh.pdf"),       "figures/sinus_tanh.pdf"),
    (os.path.join(ROOT, "figures", "transformer_time.pdf"), "figures/transformer_time.pdf"),
]

zip_path = os.path.join(HERE, ZIP_NAME)
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for src, arcname in ENTRIES:
        if not os.path.isfile(src):
            raise SystemExit("MANQUANT : " + src)
        assert "\\" not in arcname, "arcname avec antislash : " + arcname
        zf.write(src, arcname=arcname)

print("Ecrit :", ZIP_NAME)

# ── Verification : aucun antislash, ni dans les arcnames ni dans les octets ──
with open(zip_path, "rb") as f:
    raw = f.read()
names = zipfile.ZipFile(zip_path).namelist()
bad = [n for n in names if "\\" in n]
print("  entrees                        :", len(names))
print("  arcnames avec antislash        :", len(bad))
print("  octets 'figures\\\\' dans le zip :", raw.count(b"figures\\"))
for n in names:
    print("      -", n)
if bad or raw.count(b"figures\\"):
    raise SystemExit("ECHEC : des antislashs subsistent, arXiv refuserait l'archive.")
print("  -> separateurs conformes (slashs uniquement)")
