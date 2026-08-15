"""Encodeur BPE byte-level pour Qwen2.5, et jeu de prompts pour le balayage de p.

POURQUOI CE FICHIER EXISTE
--------------------------
`bench_eps_branch_order_qwen.jl` a mesuré la fraction de masse par branche
p ~= 0.16 sur UN seul prompt. La question ouverte est : p est-il une constante
d'architecture+poids (donc exploitable dans un theoreme), ou varie-t-il avec
l'entree ? Il fallait donc plus de prompts, et `adhoc_prompts.json` n'en
contient que 3. Le depot n'a pas de tokeniseur, mais le modele est livre avec
vocab.json + merges.txt : on reconstruit l'encodeur.

PORTE DE VALIDATION, AVANT TOUT USAGE
-------------------------------------
Cet encodeur DOIT reproduire exactement les 3 tokenisations deja presentes dans
adhoc_prompts.json (produites par le vrai tokeniseur HF). S'il en devie d'un
seul identifiant, il est faux et le script s'arrete : des ids errones
donneraient des activations plausibles mais fausses, l'erreur la plus
difficile a reperer en aval.

APPROXIMATION ASSUMEE
---------------------
Le pre-tokeniseur de Qwen2 utilise des classes Unicode (\\p{L}, \\p{N}) que le
module `re` de Python ne supporte pas. Si `regex` est installe on l'utilise tel
quel ; sinon on retombe sur une traduction ASCII (A-Za-z, 0-9), EXACTE pour de
l'anglais pur -- ce que sont tous les prompts ci-dessous. La porte de
validation tranche : si le repli etait insuffisant, les 3 references ne
passeraient pas.

USAGE
  python notebook/qwen_tokenize_prompts.py
Ecrit notebook/qwen_sweep_prompts.json
"""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "qwen2.5-1.5b-instruct")

PAT_UNICODE = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
               r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
PAT_ASCII = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]|"
             r" ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

try:
    import regex as _re
    PAT, ENGINE = PAT_UNICODE, "regex (Unicode)"
except ImportError:
    import re as _re
    PAT, ENGINE = PAT_ASCII, "re (repli ASCII)"
SPLIT = _re.compile(PAT)


def bytes_to_unicode():
    """Table byte -> caractere imprimable de GPT-2, reprise telle quelle par Qwen2."""
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


B2U = bytes_to_unicode()

vocab = json.load(io.open(os.path.join(MODEL_DIR, "vocab.json"), encoding="utf-8"))
ranks = {}
with io.open(os.path.join(MODEL_DIR, "merges.txt"), encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.rstrip("\n")
        if not line or line.startswith("#version"):
            continue
        a, _, b = line.partition(" ")
        ranks[(a, b)] = len(ranks)


def bpe(token):
    """Fusionne les paires par rang croissant, algorithme BPE standard."""
    word = list(token)
    if len(word) < 2:
        return word
    while True:
        best, best_rank = None, None
        for i in range(len(word) - 1):
            r = ranks.get((word[i], word[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best, best_rank = i, r
        if best is None:
            return word
        word[best:best + 2] = [word[best] + word[best + 1]]
        if len(word) == 1:
            return word


_cache = {}


def encode(text):
    out = []
    for piece in SPLIT.findall(text):
        if not piece:
            continue
        mapped = "".join(B2U[b] for b in piece.encode("utf-8"))
        if mapped not in _cache:
            _cache[mapped] = bpe(mapped)
        for sub in _cache[mapped]:
            if sub not in vocab:
                raise KeyError("sous-token absent du vocabulaire : %r" % sub)
            out.append(vocab[sub])
    return out


# ─── PORTE DE VALIDATION ───────────────────────────────────────────────────
ref = json.load(io.open(os.path.join(MODEL_DIR, "adhoc_prompts.json"), encoding="utf-8"))
print("Moteur regex :", ENGINE)
print("Vocabulaire  :", len(vocab), " fusions :", len(ranks))
print("\n=== PORTE : reproduction des tokenisations de reference ===")
all_ok = True
for r in ref:
    got = encode(r["prompt"])
    ok = (got == r["token_ids"])
    all_ok = all_ok and ok
    print("  %-52s %s" % (repr(r["prompt"])[:52], "ok" if ok else "ECHEC"))
    if not ok:
        print("     attendu :", r["token_ids"])
        print("     obtenu  :", got)
if not all_ok:
    raise SystemExit("\nPORTE ECHOUEE -- encodeur faux, aucun prompt genere.")
print("  -> %d/%d references reproduites exactement." % (len(ref), len(ref)))

# ─── Jeu de balayage ───────────────────────────────────────────────────────
# Choisi pour faire varier ce qui pourrait faire bouger p : le domaine
# (factuel, arithmetique, code, narratif, listes), la longueur, et la
# predictibilite du token suivant (une suite tres previsible concentre le
# gradient differemment d'une suite ambigue).
PROMPTS = [
    # factuel / rappel a un saut
    "The capital of France is",
    "The capital of Japan is",
    "The largest ocean on Earth is the",
    # deux sauts -- la forme meme de la figure R-lens
    "The capital of the country famous for ramen noodles is",
    "The capital of the country where the Eiffel Tower stands is",
    # arithmetique
    "If a train travels 60 miles in 2 hours, its speed is",
    "2 plus 2 equals",
    "The sum of 15 and 27 is",
    # code
    "def add(a, b):\n    return",
    "for i in range(10):\n    print(",
    "SELECT name FROM users WHERE id =",
    # narratif / long
    "It was a bright cold day in April, and the clocks were striking",
    "Once upon a time there lived a young girl who dreamed of",
    # listes / structure tres previsible
    "Monday, Tuesday, Wednesday,",
    "one, two, three, four,",
    "a b c d e f g",
    # question-reponse
    "Q: What is the largest planet in the solar system?\nA:",
    "Question: Who wrote Hamlet?\nAnswer:",
    # antonymes / analogie
    "The opposite of hot is",
    "big is to small as tall is to",
    # repetition (structure d'induction)
    "cat dog bird cat dog",
    "The key is under the mat. The key is under the",
]

out = []
for p in PROMPTS:
    ids = encode(p)
    out.append({"prompt": p, "token_ids": ids, "n": len(ids)})

dest = os.path.join(HERE, "qwen_sweep_prompts.json")
io.open(dest, "w", encoding="utf-8").write(
    json.dumps(out, ensure_ascii=False, indent=1) + "\n")

print("\n=== Jeu de balayage ===")
for i, d in enumerate(out):
    print("  %2d. %-56s %3d tokens" % (i + 1, repr(d["prompt"])[:56], d["n"]))
print("\nLongueurs : min=%d  median=%d  max=%d" % (
    min(d["n"] for d in out),
    sorted(d["n"] for d in out)[len(out) // 2],
    max(d["n"] for d in out)))
print("Ecrit :", dest)
