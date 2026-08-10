"""Extrait le resume de NeuroDSL.tex en texte brut, pret a coller dans le
formulaire de metadonnees arXiv.

POURQUOI PAS pdftotext : l'extraction depuis le PDF perd les ligatures et
produit des mots mutiles -- "buer" pour "buffer", "inate" pour "inflate",
"conguration", "prex", "dierence" -- et casse les signes multiplier. Coller ca
dans arXiv publierait un resume abime. On repart donc de la SOURCE.

POURQUOI CE FICHIER EXISTE au lieu d'une ligne de shell : passer des
echappements LaTeX a travers un heredoc les detruit (constate deux fois
aujourd'hui, dont une corruption de fichier par un caractere BELL).
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "NeuroDSL.tex")
OUT = os.path.join(HERE, "NeuroDSL_abstract_arxiv.txt")

src = io.open(SRC, encoding="utf-8").read()

m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S)
if m is None:
    raise SystemExit("resume introuvable dans NeuroDSL.tex")
t = m.group(1)

# Deplier les commandes qui n'ont pas de sens en texte brut.
for pat in (r"\\textsc\{([^{}]*)\}", r"\\emph\{([^{}]*)\}", r"\\textbf\{([^{}]*)\}",
            r"\\texttt\{([^{}]*)\}", r"\\text\{([^{}]*)\}", r"\\mathrm\{([^{}]*)\}",
            r"\\mathcal\{([^{}]*)\}"):
    for _ in range(6):          # imbrications
        t = re.sub(pat, r"\1", t)

t = re.sub(r"\\cite\{[^}]*\}", "", t)
t = re.sub(r"\\label\{[^}]*\}", "", t)
t = t.replace(r"\times", "\u00d7").replace(r"\%", "%").replace(r"\&", "&")
t = t.replace(r"\,", " ").replace("~", " ")
t = t.replace("10^{-6}", "10^-6").replace("{+}", "+").replace("{-}", "-")
t = t.replace("$", "")
t = t.replace("---", "\u2014").replace("--", "\u2013")
t = re.sub(r"\\[a-zA-Z]+", "", t)       # commandes residuelles
t = t.replace("{", "").replace("}", "")
t = re.sub(r"\s+", " ", t).strip()

io.open(OUT, "w", encoding="utf-8").write(t + "\n")

print("Ecrit :", os.path.basename(OUT))
print("  mots       :", len(t.split()))
print("  caracteres :", len(t))
suspects = [w for w in ("buer", "inate", "conguration", "prex", "dierence",
                        "oers", "nal ", "\ufffd", "\\") if w in t]
print("  residus suspects :", suspects if suspects else "aucun")
