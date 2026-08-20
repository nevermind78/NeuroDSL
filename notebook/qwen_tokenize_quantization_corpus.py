"""Tokenize the fresh held-out corpus (quantization_signature_qwen_corpus.txt) and a frequency
reference corpus (TinyShakespeare, already in this repo, used ONLY to build an empirical BPE-token
frequency table -- never mixed into the held-out evaluation corpus itself) with the same
hand-built Qwen2.5 BPE encoder validated in qwen_tokenize_prompts.py (same vocab.json/merges.txt,
same validation gate against adhoc_prompts.json before trusting any output).

Writes:
  notebook/quantization_signature_qwen_corpus_tokens.json
    {"token_ids": [...], "n_tokens": N, "n_chars": M}
  notebook/quantization_signature_qwen_freqref_tokens.json
    {"token_counts": {token_id_str: count, ...}, "n_tokens": N}

Usage: python notebook/qwen_tokenize_quantization_corpus.py
"""
import io
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "qwen2.5-1.5b-instruct")
CORPUS_F = os.path.join(HERE, "quantization_signature_qwen_corpus_full.txt")
FREQREF_F = os.path.join(HERE, "data", "tinyshakespeare", "input.txt")

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
    for line in f:
        line = line.rstrip("\n")
        if not line or line.startswith("#version"):
            continue
        a, _, b = line.partition(" ")
        ranks[(a, b)] = len(ranks)


def bpe(token):
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


# ─── VALIDATION GATE (same as qwen_tokenize_prompts.py) ─────────────────────
ref = json.load(io.open(os.path.join(MODEL_DIR, "adhoc_prompts.json"), encoding="utf-8"))
print("Engine:", ENGINE)
print("Vocab size:", len(vocab), " merges:", len(ranks))
print("\n=== GATE: reproducing reference tokenizations ===")
all_ok = True
for r in ref:
    got = encode(r["prompt"])
    ok = (got == r["token_ids"])
    all_ok = all_ok and ok
    print("  %-52s %s" % (repr(r["prompt"])[:52], "ok" if ok else "FAIL"))
    if not ok:
        print("     expected:", r["token_ids"])
        print("     got     :", got)
if not all_ok:
    raise SystemExit("\nGATE FAILED -- encoder is wrong, nothing written.")
print("  -> %d/%d references reproduced exactly." % (len(ref), len(ref)))

# ─── Tokenize the held-out corpus ────────────────────────────────────────────
with io.open(CORPUS_F, encoding="utf-8") as f:
    corpus_text = f.read()
corpus_ids = encode(corpus_text)
print("\nHeld-out corpus: %d chars -> %d BPE tokens" % (len(corpus_text), len(corpus_ids)))
with io.open(os.path.join(HERE, "quantization_signature_qwen_corpus_tokens.json"), "w",
             encoding="utf-8") as f:
    json.dump({"token_ids": corpus_ids, "n_tokens": len(corpus_ids), "n_chars": len(corpus_text)}, f)

# ─── Tokenize the frequency-reference corpus (TinyShakespeare) ──────────────
with io.open(FREQREF_F, encoding="utf-8") as f:
    freqref_text = f.read()
freqref_ids = encode(freqref_text)
counts = Counter(freqref_ids)
print("Frequency-reference corpus (TinyShakespeare): %d chars -> %d BPE tokens, %d distinct token ids"
      % (len(freqref_text), len(freqref_ids), len(counts)))
with io.open(os.path.join(HERE, "quantization_signature_qwen_freqref_tokens.json"), "w",
             encoding="utf-8") as f:
    json.dump({"token_counts": {str(k): v for k, v in counts.items()}, "n_tokens": len(freqref_ids)}, f)

print("\nWrote:")
print(" ", os.path.join(HERE, "quantization_signature_qwen_corpus_tokens.json"))
print(" ", os.path.join(HERE, "quantization_signature_qwen_freqref_tokens.json"))
