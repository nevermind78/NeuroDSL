"""
P4 -- comparaison croisee NeuroDSL vs PyTorch du cout d'un balayage de patching
d'activation, sur le MEME modele (5 couches effectives : 4 originales + 1
greffee via insert_block!, poids exportes depuis Julia par save_graph!,
src/serialization.jl) et les MEMES 20 sites (5 couches x 4 tetes
d'attention, noeud `ao_h` = sortie post-attention d'une tete avant la
concatenation).

Reutilise les classes deja ecrites et validees dans real_llm_py.py
(RMSNorm, architecture LlamaBlock) -- seule MultiHeadAttention est etendue
avec une option d'override par tete (necessaire pour reproduire le patch
node-level de NeuroDSL, sans quoi il n'y a pas de granularite par tete en
PyTorch eager standard).

Trois methodes comparees pour chaque site :
  (a) "hook naif"        : forward complet des 5 couches, tete i patchee au
                            passage -- cout structurel de TransformerLens/
                            hooks PyTorch standard (un forward entier par
                            site, quelle que soit sa profondeur).
  (b) "partiel optimise" : reprend depuis l'entree EN CACHE de la couche
                            patchee (les couches amont ne sont jamais
                            rejouees), mais doit toujours recalculer TOUTE
                            l'attention de cette couche (les 4 tetes, pas
                            seulement celle patchee) faute de nœuds
                            individuellement adressables en PyTorch eager --
                            meilleur effort manuel raisonnable, pas une
                            reimplementation du mecanisme de NeuroDSL.
  (c) NeuroDSL (sweep_patch_sites!, deja mesure par
      notebook/p4_export_and_bench.jl, relu ici seulement pour le tableau
      final) : ne recalcule ni les couches amont NI les tetes soeurs de la
      couche patchee -- seul mecanisme des trois qui evite aussi (b).

Verification de correction AVANT toute mesure de vitesse (meme discipline
que tout le reste de la session) : le forward PyTorch (poids charges depuis
l'export Julia) doit reproduire `reference_output.bin` (le forward propre
calcule par NeuroDSL) a moins de 1e-3 d'erreur absolue max -- sinon on
mesurerait deux implementations qui ne calculent pas la meme fonction.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(2026)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

HERE = Path(__file__).resolve().parent
EXPORT_DIR = HERE / "p4_export"

# ── 1. Chargement du manifeste + des poids (format Julia save_graph!) ───────
with open(EXPORT_DIR / "meta.json", encoding="utf-8") as f:
    meta = json.load(f)
with open(EXPORT_DIR / "model.json", encoding="utf-8") as f:
    manifest = json.load(f)

DIM = meta["dim"]; N_HEADS = meta["n_heads"]; HIDDEN_DIM = meta["hidden_dim"]
SEQ_LEN = meta["seq_len"]; LAYER_PREFIXES = meta["layer_prefixes"]
D_HEAD = DIM // N_HEADS
print(f"dim={DIM} n_heads={N_HEADS} hidden_dim={HIDDEN_DIM} seq_len={SEQ_LEN} "
      f"couches={LAYER_PREFIXES}")

bin_path = EXPORT_DIR / "model.bin"
raw = bin_path.read_bytes()
params = {}   # nom (str) -> np.ndarray, forme logique = manifest (ordre Julia = colonne-majeur)
for nm in manifest["nodes"]:
    if "shape" not in nm:
        continue
    shape = tuple(nm["shape"])
    n = int(np.prod(shape)) if shape else 1
    offset = nm["blob_offset"]
    arr = np.frombuffer(raw, dtype=np.float32, count=n, offset=offset)
    # Julia ecrit les tableaux en ordre colonne-majeur -- reshape 'F' pour
    # retrouver la meme disposition logique (meme convention que le reste
    # de la session pour les echanges Julia<->Python de tableaux bruts).
    arr = arr.reshape(shape, order="F") if shape else arr.copy()
    params[nm["name"]] = arr.copy()

print(f"{len(params)} tenseurs charges depuis {bin_path.name}")


def load_linear_(lin: nn.Linear, prefix: str):
    """Charge le poids (et biais si present) d'un `Linear` NeuroDSL (meme
    convention (out_features, in_features), src/layers.jl:34-46) dans un
    `nn.Linear` -- aucune transposition necessaire."""
    W = params[f"{prefix}_W"]
    assert tuple(W.shape) == tuple(lin.weight.shape), (prefix, W.shape, lin.weight.shape)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(W))
        bname = f"{prefix}_b"
        if bname in params:
            lin.bias.copy_(torch.from_numpy(params[bname]))


def load_rmsnorm_gamma(prefix: str) -> torch.Tensor:
    return torch.from_numpy(params[f"{prefix}_gamma"]).to(DEVICE)


# ── 2. Architecture -- RMSNorm reprise telle quelle de real_llm_py.py ───────
class RMSNorm(nn.Module):
    def __init__(self, dim, gamma, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(gamma.clone())
        self.eps = eps

    def forward(self, x):
        rms_inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms_inv * self.gamma


class PatchableMHA(nn.Module):
    """Comme `MultiHeadAttention` de real_llm_py.py, mais expose chaque tete
    individuellement (ao_h avant concatenation) pour permettre l'override
    d'UNE seule tete -- necessaire pour reproduire le patch node-level de
    NeuroDSL (`{prefix}_mha_ao_h{h}`)."""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim; self.n_heads = n_heads; self.d_head = dim // n_heads
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, override_head=None, override_value=None, return_heads=False):
        seq = x.size(0)
        q = self.Wq(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        k = self.Wk(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        v = self.Wv(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ao = torch.matmul(attn, v)          # (n_heads, seq, d_head)
        heads = [ao[h] for h in range(self.n_heads)]
        if override_head is not None:
            heads[override_head] = override_value
        concat = torch.cat(heads, dim=-1)   # (seq, dim)
        out = self.Wo(concat)
        if return_heads:
            return out, heads
        return out


class PatchableLlamaBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, prefix):
        super().__init__()
        mha_prefix = f"{prefix}_mha"
        self.norm1 = RMSNorm(dim, load_rmsnorm_gamma(f"{prefix}_norm1"))
        self.attn = PatchableMHA(dim, n_heads)
        load_linear_(self.attn.Wq, f"{mha_prefix}_q")
        load_linear_(self.attn.Wk, f"{mha_prefix}_k")
        load_linear_(self.attn.Wv, f"{mha_prefix}_v")
        load_linear_(self.attn.Wo, f"{mha_prefix}_output")
        self.norm2 = RMSNorm(dim, load_rmsnorm_gamma(f"{prefix}_norm2"))
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.w1.weight.copy_(torch.from_numpy(params[f"{prefix}_mlp_w1"]))
            self.w2.weight.copy_(torch.from_numpy(params[f"{prefix}_mlp_w2"]))
            self.w3.weight.copy_(torch.from_numpy(params[f"{prefix}_mlp_w3"]))

    def mlp(self, xn2):
        gate = self.w1(xn2); up = self.w3(xn2)
        swiglu = gate * torch.sigmoid(gate) * up
        return self.w2(swiglu)

    def forward(self, x, override_head=None, override_value=None, return_heads=False):
        if return_heads:
            ao, heads = self.attn(self.norm1(x), override_head, override_value, return_heads=True)
        else:
            ao = self.attn(self.norm1(x), override_head, override_value)
            heads = None
        r1 = x + ao
        out = r1 + self.mlp(self.norm2(r1))
        if return_heads:
            return out, r1, heads
        return out, r1


blocks = [PatchableLlamaBlock(DIM, N_HEADS, HIDDEN_DIM, p).to(DEVICE) for p in LAYER_PREFIXES]
for b in blocks:
    b.eval()

INPUT = torch.from_numpy(params["input"]).to(DEVICE)
assert tuple(INPUT.shape) == (SEQ_LEN, DIM)

# ── 3. Verification de parite AVANT toute mesure de vitesse ─────────────────
ref_bytes = (EXPORT_DIR / "reference_output.bin").read_bytes()
ref_shape = tuple(meta["output_shape"])
reference_output = np.frombuffer(ref_bytes, dtype=np.float32).reshape(ref_shape, order="F")

with torch.no_grad():
    x = INPUT
    for b in blocks:
        x, _ = b(x)
    pytorch_output = x.cpu().numpy()

parity_max_abs_diff = float(np.max(np.abs(pytorch_output - reference_output)))
print(f"\nParite NeuroDSL vs PyTorch (max abs diff des logits) : {parity_max_abs_diff:.3e}")
assert parity_max_abs_diff < 1e-3, (
    "Parite insuffisante -- les deux implementations ne calculent pas la meme "
    "fonction, arret avant toute mesure de vitesse."
)

# ── 4. Corruption (un seul token, protocole standard, memes conventions que tout le reste de la session) ──
gen = torch.Generator(device=DEVICE).manual_seed(99)
CORRUPT_ROW = 4  # 0-indexe (Python) -- valeur precise sans incidence sur le cout, seulement sur la valeur
corrupted_input = INPUT.clone()
corrupted_input[CORRUPT_ROW] = (torch.rand(DIM, generator=gen, device=DEVICE) - 0.5)

with torch.no_grad():
    # Cache "propre" : sortie par tete de CHAQUE couche sur l'entree propre --
    # sert de valeur a injecter lors du patch (equivalent de clean_cache).
    clean_layer_inputs = [INPUT]
    clean_head_cache = []  # clean_head_cache[i][h] = tete h de la couche i, run propre
    x = INPUT
    for b in blocks:
        x, r1, heads = b(x, return_heads=True)
        clean_head_cache.append([h.clone() for h in heads])
        clean_layer_inputs.append(x)

    # Cache "corrompu" : memes couches, MEME entree corrompue -- sert d'etat
    # de reference pour reprendre le calcul en amont d'un site (methode b).
    corrupted_layer_inputs = [corrupted_input]
    x = corrupted_input
    for b in blocks:
        x, r1 = b(x)
        corrupted_layer_inputs.append(x)


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ── 5a. Methode "hook naif" : forward COMPLET des 5 couches a chaque patch ──
def full_forward_patch(layer_idx, head_idx):
    x = corrupted_input
    for i, b in enumerate(blocks):
        if i == layer_idx:
            x, _ = b(x, override_head=head_idx, override_value=clean_head_cache[i][head_idx])
        else:
            x, _ = b(x)
    return x


# ── 5b. Methode "partiel optimise" : reprend depuis l'entree EN CACHE de la
#         couche patchee (couches amont jamais rejouees) -- doit neanmoins
#         recalculer toute l'attention de cette couche (4 tetes), faute de
#         cache par tete adressable individuellement en PyTorch eager. ──
def partial_forward_patch(layer_idx, head_idx):
    x = corrupted_layer_inputs[layer_idx]
    for i in range(layer_idx, len(blocks)):
        b = blocks[i]
        if i == layer_idx:
            x, _ = b(x, override_head=head_idx, override_value=clean_head_cache[i][head_idx])
        else:
            x, _ = b(x)
    return x


def timed(fn, reps=20, warmup=3):
    with torch.no_grad():
        for _ in range(warmup):
            fn(); sync()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(times); n = len(s)
    return {"median": s[n // 2], "q25": s[max(0, n // 4)], "q75": s[min(n - 1, 3 * n // 4)]}


# ── 6. Reference : un forward complet propre -> corrompu (cout "naif sans patch") ──
def full_forward_no_patch():
    x = corrupted_input
    for b in blocks:
        x, _ = b(x)
    return x


full_forward_stats = timed(full_forward_no_patch)
print(f"\nForward complet (reference) : mediane={full_forward_stats['median']:.4f} ms "
      f"[q25={full_forward_stats['q25']:.4f} q75={full_forward_stats['q75']:.4f}]")

# ── 7. Balayage des 20 sites, 2 methodes ─────────────────────────────────────
rows = []
for layer_idx, prefix in enumerate(LAYER_PREFIXES):
    for head_idx in range(N_HEADS):
        site = f"{prefix}_mha_ao_h{head_idx + 1}"
        stat_full = timed(lambda li=layer_idx, hi=head_idx: full_forward_patch(li, hi))
        stat_partial = timed(lambda li=layer_idx, hi=head_idx: partial_forward_patch(li, hi))
        rows.append({
            "site": site, "layer_idx": layer_idx, "head_idx": head_idx + 1,
            "full_forward_hook_med_ms": stat_full["median"],
            "full_forward_hook_q25": stat_full["q25"], "full_forward_hook_q75": stat_full["q75"],
            "partial_forward_med_ms": stat_partial["median"],
            "partial_forward_q25": stat_partial["q25"], "partial_forward_q75": stat_partial["q75"],
        })
        print(f"  {site:32s} hook_naif={stat_full['median']:7.4f} ms   "
              f"partiel_optimise={stat_partial['median']:7.4f} ms")

results = {
    "parity_max_abs_diff": parity_max_abs_diff,
    "full_forward_ms": full_forward_stats,
    "sites": rows,
}
out_path = EXPORT_DIR / "pytorch_bench_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1)
print(f"\nResultats PyTorch ecrits -> {out_path}")
