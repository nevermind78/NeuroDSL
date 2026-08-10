"""Compte les environnements flottants de NeuroDSL.tex et l'origine des figures.

Ecrit dans un fichier plutot que passe au shell : les echappements LaTeX
survivent mal a un heredoc ou a des guillemets imbriques (constate plusieurs
fois le 2026-08-08).
"""
import io
import os
import re

BS = chr(92)
HERE = os.path.dirname(os.path.abspath(__file__))
s = io.open(os.path.join(HERE, "NeuroDSL.tex"), encoding="utf-8").read()

print("=== environnements ===")
for env in ("figure", "table", "tabular", "lstlisting", "tikzpicture", "algorithm"):
    print("  %-13s : %d" % (env, s.count(BS + "begin{" + env + "}")))

print()
print("  includegraphics (fichiers externes) : %d" % s.count(BS + "includegraphics"))

print()
print("=== detail des figures ===")
pat = re.compile(re.escape(BS + "begin{figure}") + r"(.*?)" + re.escape(BS + "end{figure}"), re.S)
for i, m in enumerate(pat.finditer(s), 1):
    body = m.group(1)
    img = re.search(r"includegraphics[^{]*\{([^}]*)\}", body)
    tikz = "tikzpicture" in body
    cap = re.search(re.escape(BS + "caption") + r"\{(.{0,88})", body)
    src = img.group(1) if img else ("TikZ, dessine dans le .tex" if tikz else "?")
    txt = cap.group(1).replace(BS, "") if cap else "(sans legende)"
    print("  %d. [%s]" % (i, src))
    print("     %s..." % txt)
