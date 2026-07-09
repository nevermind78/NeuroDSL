"""
Miroir PyTorch strict de notebook/real_llm.ipynb -- meme architecture (RMSNorm,
attention multi-tete sans biais, SwiGLU, embeddings positionnels appris,
Linear final AVEC biais), memes hyperparametres, meme corpus/tokenizer/split,
batch_size=1 (le moteur NeuroDSL n'a pas de dimension batch, voir le plan),
clip_grad_value_ (pas clip_grad_norm_, car adamw_step! de NeuroDSL clippe par
valeur : src/kernels.jl:445, gc = clamp.(dW, -clip, clip)).

Sert de verification croisee de correction (val loss finale proche de celle
de NeuroDSL) et de comparaison vitesse/memoire -- secondaire, pas la
revendication principale (voir le plan : la preuve qualitative que NeuroDSL
apprend reellement du langage est le livrable central).
"""
import json
import math
import time
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1234)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

HERE = Path(__file__).resolve().parent
CORPUS_PATH = HERE / "data" / "tinyshakespeare" / "input.txt"

# ── 1. Corpus (meme fichier que real_llm.ipynb -- pas de second telechargement) ──
assert CORPUS_PATH.is_file(), f"Corpus absent : {CORPUS_PATH} -- executer real_llm.ipynb d'abord (il le telecharge)."
text = CORPUS_PATH.read_text(encoding="utf-8")
print(f"Corpus charge : {len(text)} caracteres")

# ── 2. Tokenizer caractere -- MEME convention que Julia : sort(unique(collect(text))) ──
# Python sorted() sur des caracteres compare aussi par point de code Unicode,
# donc le meme ordre que Julia tant que le fichier est identique (verifie ci-dessous).
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}  # 0-base (Python) -- IDs differents de Julia
itos = {i: c for c, i in stoi.items()}      # (1-base), mais MEME ordre de caracteres,
                                             # donc les deux tokenizers sont equivalents.
print(f"vocab_size = {vocab_size}")
print(f"10 premiers caracteres (tries) = {chars[:10]}")
print("(a comparer avec les 10 premiers caracteres imprimes par real_llm.ipynb -- doivent etre identiques)")

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

# ── 3. Split train/val 90/10 par position -- identique a real_llm.ipynb ──────
data = encode(text)
n_total = len(data)
n_train = int(0.9 * n_total)
train_ids = data[:n_train]
val_ids = data[n_train:]
print(f"train: {len(train_ids)} caracteres  |  val: {len(val_ids)} caracteres")

# ── 4. Architecture -- copie fidele de src/layers.jl ─────────────────────────
BLOCK_SIZE = 256
DIM = 256
N_HEADS = 4
HIDDEN_DIM = 512
N_LAYERS = 4
D_HEAD = DIM // N_HEADS


class RMSNorm(nn.Module):
    """Meme formule que src/layers.jl:19-26 (op :rmsnorm) -- PAS nn.LayerNorm."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms_inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms_inv * self.gamma


class MultiHeadAttention(nn.Module):
    """Copie de src/layers.jl:51-93 : Q/K/V/O sans biais, masque causal,
    scale = 1/sqrt(d_head) (src/dispatch.jl :scale_mask)."""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        seq = x.size(0)
        q = self.Wq(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        k = self.Wk(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        v = self.Wv(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(0, 1).reshape(seq, self.dim)
        return self.Wo(out)


class LlamaBlock(nn.Module):
    """Copie de src/layers.jl:96-125 : pre-norm, residuel, SwiGLU (gate*sigmoid(gate)*up,
    src/kernels.jl swiglu_fwd!), hidden_dim independant de dim."""
    def __init__(self, dim, n_heads, hidden_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down

    def forward(self, x):
        r1 = x + self.attn(self.norm1(x))
        xn2 = self.norm2(r1)
        gate = self.w1(xn2)
        up = self.w3(xn2)
        swiglu = gate * torch.sigmoid(gate) * up
        return r1 + self.w2(swiglu)


class CharLM(nn.Module):
    """Copie de build_char_lm_graph : Embedding(tok)+Embedding(pos) -> LlamaModel -> Linear(bias=True)."""
    def __init__(self, vocab_size, dim, n_heads, hidden_dim, n_layers, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.blocks = nn.ModuleList([LlamaBlock(dim, n_heads, hidden_dim) for _ in range(n_layers)])
        self.lm_head = nn.Linear(dim, vocab_size, bias=True)  # NeuroDSL Linear() defaut bias=True
        self._init_weights()

    def _init_weights(self):
        # nn.Linear par defaut (kaiming_uniform_ a=sqrt(5)) donne deja Uniform(-1/sqrt(fan_in),
        # 1/sqrt(fan_in)) -- identique a src/layers.jl:35-36, rien a changer pour les Linear.
        # nn.Embedding par defaut est normal(0,1) -- DIFFERENT de src/layers.jl:9
        # (Uniform(-1/sqrt(dim), 1/sqrt(dim))) -- corrige ici pour une comparaison equitable.
        k_tok = 1.0 / math.sqrt(self.tok_emb.embedding_dim)
        k_pos = 1.0 / math.sqrt(self.pos_emb.embedding_dim)
        nn.init.uniform_(self.tok_emb.weight, -k_tok, k_tok)
        nn.init.uniform_(self.pos_emb.weight, -k_pos, k_pos)

    def forward(self, token_ids, pos_ids):
        x = self.tok_emb(token_ids) + self.pos_emb(pos_ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(x)


model = CharLM(vocab_size, DIM, N_HEADS, HIDDEN_DIM, N_LAYERS, BLOCK_SIZE).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Modele : {n_params} parametres (~{n_params/1e6:.2f}M)")

# ── 5. Echantillonnage, perte de validation, generation -- memes conventions ─
import random
rng = random.Random(123)


def sample_window(ids, block_size):
    i = rng.randint(0, len(ids) - block_size - 1)
    tokens = ids[i:i + block_size]
    labels = ids[i + 1:i + block_size + 1]
    return tokens, labels


@torch.no_grad()
def val_loss_fn(model, val_ids, block_size, n_windows=64):
    model.eval()
    max_start = len(val_ids) - block_size - 1
    starts = [round(i) for i in torch.linspace(0, max_start, n_windows).tolist()]
    total = 0.0
    pos_ids = torch.arange(block_size, device=DEVICE)
    for i in starts:
        tokens = torch.tensor(val_ids[i:i + block_size], dtype=torch.long, device=DEVICE)
        labels = torch.tensor(val_ids[i + 1:i + block_size + 1], dtype=torch.long, device=DEVICE)
        logits = model(tokens, pos_ids)
        loss = F.cross_entropy(logits, labels)
        total += loss.item()
    model.train()
    return total / n_windows


@torch.no_grad()
def generate_text(model, seed_text="\n", n_chars=300, temperature=0.8, block_size=BLOCK_SIZE,
                   gen_rng=None, use_argmax=False):
    model.eval()
    gen_rng = gen_rng or random.Random(777)
    ctx = encode(seed_text)
    out_chars = []
    for _ in range(n_chars):
        window = ctx[-block_size:] if len(ctx) > block_size else ctx
        t = len(window)
        tokens = torch.tensor(window, dtype=torch.long, device=DEVICE)
        pos_ids = torch.arange(t, device=DEVICE)
        logits = model(tokens, pos_ids)
        row = logits[-1]
        if use_argmax:
            next_id = int(torch.argmax(row).item())
        else:
            p = torch.softmax(row / temperature, dim=-1).cpu()
            r = gen_rng.random()
            cum = 0.0
            next_id = len(p) - 1
            for idx, pi in enumerate(p.tolist()):
                cum += pi
                if r <= cum:
                    next_id = idx
                    break
        ctx.append(next_id)
        out_chars.append(itos[next_id])
    model.train()
    return "".join(out_chars)


# ── 6. Sanite : perte initiale ~= ln(vocab_size) ─────────────────────────────
tokens0, labels0 = sample_window(train_ids, BLOCK_SIZE)
tokens0_t = torch.tensor(tokens0, dtype=torch.long, device=DEVICE)
labels0_t = torch.tensor(labels0, dtype=torch.long, device=DEVICE)
pos_ids_full = torch.arange(BLOCK_SIZE, device=DEVICE)
with torch.no_grad():
    logits0 = model(tokens0_t, pos_ids_full)
    loss0 = F.cross_entropy(logits0, labels0_t).item()
print(f"Perte initiale mesuree : {loss0:.4f}   (attendu ln({vocab_size}) = {math.log(vocab_size):.4f})")
assert abs(loss0 - math.log(vocab_size)) < 0.5, "Perte initiale trop loin de ln(vocab_size)"

print("\n--- Echantillon AVANT tout entrainement (poids aleatoires) ---")
sample_before = generate_text(model, gen_rng=random.Random(777))
print(sample_before)

# ── 7. Entrainement -- memes hyperparametres que real_llm.ipynb ─────────────
N_STEPS = 20_000
LR = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

train_losses = []
val_history = []
sample_steps = {500, 2500, 10000}
VAL_EVERY = 500

if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

t_start = time.time()
for step in range(1, N_STEPS + 1):
    tokens, labels = sample_window(train_ids, BLOCK_SIZE)
    tokens_t = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
    labels_t = torch.tensor(labels, dtype=torch.long, device=DEVICE)

    optimizer.zero_grad()
    logits = model(tokens_t, pos_ids_full)
    loss = F.cross_entropy(logits, labels_t)
    loss.backward()
    # clip PAR VALEUR, pas par norme -- miroir exact de adamw_step! (src/kernels.jl:445,
    # gc = clamp.(dW, -clip, clip)), pas clip_grad_norm_.
    nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    optimizer.step()

    train_losses.append(loss.item())

    if step % VAL_EVERY == 0 or step == 1:
        vl = val_loss_fn(model, val_ids, BLOCK_SIZE)
        val_history.append((step, vl))
        print(f"step {step:6d} | train {train_losses[-1]:.4f} | val {vl:.4f} | "
              f"ppl {math.exp(vl):.2f} | bits/char {vl/math.log(2):.3f}")

    if step in sample_steps:
        s = generate_text(model, gen_rng=random.Random(777))
        print(f"\n--- Echantillon @ pas {step} ---\n{s}\n")

if DEVICE.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.time() - t_start
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE.type == "cuda" else 0.0

print(f"\nEntrainement termine : {N_STEPS} pas en {elapsed:.1f} s "
      f"({1000*elapsed/N_STEPS:.2f} ms/pas moyen)")
if DEVICE.type == "cuda":
    print(f"VRAM pic : {peak_vram_mb:.1f} MB")

# ── 8. Echantillon final + comparaison argmax ────────────────────────────────
print("\n--- Echantillon APRES entrainement (temperature=0.8) ---")
sample_after = generate_text(model, n_chars=400, gen_rng=random.Random(777))
print(sample_after)

print("\n--- Echantillon APRES entrainement (argmax, pour comparaison documentee) ---")
sample_after_argmax = generate_text(model, n_chars=200, gen_rng=random.Random(777), use_argmax=True)
print(sample_after_argmax)

final_val = val_history[-1][1]
print(f"\nVal loss finale : {final_val:.4f} nats/char "
      f"(perplexite {math.exp(final_val):.2f}, {final_val/math.log(2):.3f} bits/char)")

# ── 9. Sauvegarde pour comparaison ───────────────────────────────────────────
results = {
    "vocab_size": vocab_size,
    "chars": "".join(chars),
    "n_params": n_params,
    "n_steps": N_STEPS,
    "elapsed_s": elapsed,
    "ms_per_step": 1000 * elapsed / N_STEPS,
    "peak_vram_mb": peak_vram_mb,
    "initial_loss": loss0,
    "final_train_loss": train_losses[-1],
    "final_val_loss": final_val,
    "val_history": val_history,
    "sample_before": sample_before,
    "sample_after": sample_after,
}
out_path = HERE / "real_llm_pytorch_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print(f"\nResultats ecrits -> {out_path}")
