# WIND-T20 — constantes du théorème 1 (règles neurone par neurone : complétude contre exactitude transverse).
# CPU seulement (MKL ≤ 4 fils). Poids réels (safetensors, lecteur maison), états h depuis les méta WIND.
# Par point (modèle, couche, prompt) :
#   P0  parité : f(v) recalculé = sortie MLP stockée ;
#   H1  excès d'Euler E = J h − (f(v) − f_base) ;
#   H2  injectivité : Cholesky de la Gram 𝒢 (2F×2F à porte, F×F sans porte), λ_min/λ_max (itérations) ;
#   τ*  distorsion transverse minimale d'une règle neurone par neurone COMPLÈTE : τ*² = Eᵀ(Λ_h 𝒢⁻¹ Λ_hᵀ)⁻¹E ;
#   r_½ distorsion transverse de la règle du demi (modèles à porte) ; borne inférieure √λ_min‖E‖/σ_max(Λ_h).
# USAGE : python wind_T20_thm1.py MODEL LAYER PROMPT     (MODEL ∈ qwen, gemma, gpt2)
import os
for v in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "4"
import sys, json, struct, time
import numpy as np
import scipy.linalg as sla

NB = os.path.dirname(os.path.abspath(__file__))
MODEL, LAYER, PROMPT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
OUT = os.path.join(NB, "wind_T20_thm1.json")

def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def st_tensor(path, name):
    h, off = st_header(path)
    e = h[name]; a, b = e["data_offsets"]
    with open(path, "rb") as f:
        f.seek(off + a); raw = f.read(b - a)
    if e["dtype"] == "BF16":
        x = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif e["dtype"] == "F32":
        x = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(e["dtype"])
    return x.reshape(e["shape"]).astype(np.float64)

def gemma_path(name):
    idx = json.load(open(os.path.join(NB, "gemma-2-2b", "model.safetensors.index.json")))["weight_map"]
    return os.path.join(NB, "gemma-2-2b", idx[name])

sig = lambda x: 1.0 / (1.0 + np.exp(-x))
C = np.sqrt(2.0 / np.pi)
def gelu_t(x):  t = np.tanh(C * (x + 0.044715 * x**3)); return 0.5 * x * (1 + t)
def dgelu_t(x): t = np.tanh(C * (x + 0.044715 * x**3)); return 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * C * (1 + 3 * 0.044715 * x**2)
def silu(x):  return x * sig(x)
def dsilu(x): s = sig(x); return s * (1 + x * (1 - s))

t0 = time.time()
if MODEL == "qwen":
    meta = json.load(open(os.path.join(NB, "wind_data_T8", f"wind_meta_p{PROMPT}.json")))
    P = os.path.join(NB, "qwen2.5-1.5b-instruct", "model.safetensors"); pre = f"model.layers.{LAYER}."
    Wg = st_tensor(P, pre + "mlp.gate_proj.weight"); Wu = st_tensor(P, pre + "mlp.up_proj.weight")
    Wd = st_tensor(P, pre + "mlp.down_proj.weight"); gam = st_tensor(P, pre + "post_attention_layernorm.weight")
    h = np.array(meta["H"][LAYER], dtype=np.float64); D = h.size
    r = 1.0 / np.sqrt(np.mean(h**2) + 1e-6); Fd = gam * r            # norme figée (diagonale)
    phi, dphi, gated, post = silu, dsilu, True, np.ones(D)
    m_ref = np.array(meta["MLP"][LAYER]); rmeta = meta["rms_inv_norm2"][LAYER]
elif MODEL == "gemma":
    meta = json.load(open(os.path.join(NB, "wind_data_GEMMA", f"wind_meta_p{PROMPT}.json")))
    pre = f"model.layers.{LAYER}."
    Wg = st_tensor(gemma_path(pre + "mlp.gate_proj.weight"), pre + "mlp.gate_proj.weight")
    Wu = st_tensor(gemma_path(pre + "mlp.up_proj.weight"), pre + "mlp.up_proj.weight")
    Wd = st_tensor(gemma_path(pre + "mlp.down_proj.weight"), pre + "mlp.down_proj.weight")
    w2 = st_tensor(gemma_path(pre + "pre_feedforward_layernorm.weight"), pre + "pre_feedforward_layernorm.weight")
    wp = st_tensor(gemma_path(pre + "post_feedforward_layernorm.weight"), pre + "post_feedforward_layernorm.weight")
    h = np.array(meta["H"][LAYER], dtype=np.float64); D = h.size
    r = 1.0 / np.sqrt(np.mean(h**2) + 1e-6); Fd = (1.0 + w2) * r
    v0 = Fd * h; g0 = Wg @ v0; u0 = Wu @ v0; y = Wd @ (gelu_t(g0) * u0)
    post = (1.0 + wp) / np.sqrt(np.mean(y**2) + 1e-6)                 # post-norme figée, absorbée dans W_d
    Wd = post[:, None] * Wd
    phi, dphi, gated = gelu_t, dgelu_t, True
    m_ref = np.array(meta["MLP"][LAYER]); rmeta = meta["rms_inv_norm2"][LAYER]
elif MODEL == "gpt2":
    meta = json.load(open(os.path.join(NB, "wind_data_GPT2", f"wind_meta_p{PROMPT}.json")))
    P = os.path.join(NB, "gpt2", "model.safetensors"); pre = f"h.{LAYER}."
    Wfc = st_tensor(P, pre + "mlp.c_fc.weight"); bfc = st_tensor(P, pre + "mlp.c_fc.bias")
    Wpr = st_tensor(P, pre + "mlp.c_proj.weight"); bpr = st_tensor(P, pre + "mlp.c_proj.bias")
    gam = st_tensor(P, pre + "ln_2.weight"); bet = st_tensor(P, pre + "ln_2.bias")
    h = np.array(meta["H"][LAYER], dtype=np.float64); D = h.size
    hc = h - h.mean(); r = 1.0 / np.sqrt(np.mean(hc**2) + 1e-5)
    Win = Wfc.T; Wd = Wpr.T                                           # (3072×768), (768×3072)
    phi, dphi, gated = gelu_t, dgelu_t, False
    m_ref = np.array(meta["MLP"][LAYER]); rmeta = meta["rms_inv_norm2"][LAYER]
else:
    raise SystemExit("modèle inconnu")

rec = {"model": MODEL, "layer": LAYER, "prompt": PROMPT, "D": int(D)}
if gated:
    Fn = Wg.shape[0]
    v = Fd * h; g = Wg @ v; u = Wu @ v
    f = Wd @ (phi(g) * u)
    rec["P0"] = float(np.linalg.norm(f - m_ref) / np.linalg.norm(m_ref))
    Ag = Wg * Fd[None, :]; Au = Wu * Fd[None, :]                       # dérivées par rapport à h (norme figée)
    a1, a2 = u * dphi(g), phi(g)                                        # coefficients exacts β*, γ*
    J = Wd @ (a1[:, None] * Ag + a2[:, None] * Au)
    E = J @ h - f                                                       # f_base = f(0) = 0
    tgt = f
    nrm = h / np.linalg.norm(h)                                         # normale de l'hyperplan transverse
    radial_cols = [g, u]                                                # Λ_h δ = W_d (δβ⊙g + δγ⊙u)
    fams = [Ag, Au]
    sfr = sig(g) if MODEL == "qwen" else 0.5 * (1 + np.tanh(C * (g + 0.044715 * g**3)))   # σ figé : φ(g) = g·s(g)
    half = [0.5 * u * sfr - a1, -0.5 * a2]                              # règle du demi (σ figé) − exacte
else:
    Fn = Win.shape[0]
    Pc = lambda x: x - x.mean(axis=-1, keepdims=True)
    v = gam * r * hc + bet; z = Win @ v + bfc; f = Wd @ phi(z) + bpr
    rec["P0"] = float(np.linalg.norm(f - m_ref) / np.linalg.norm(m_ref))
    A = (Win * (gam * r)[None, :]); A = A - A.mean(axis=1, keepdims=True)   # W_in·F, F = Γr·(I − 11ᵀ/D)
    a1 = dphi(z)
    J = Wd @ (a1[:, None] * A)
    zb = Win @ bet + bfc; fbase = Wd @ phi(zb) + bpr                   # base : h = 0 ⇒ v = β
    tgt = f - fbase
    E = J @ h - tgt
    nrm = hc / np.linalg.norm(hc)
    radial_cols = [A @ h]
    fams = [A]
    half = None
rec["rms_check"] = float(abs(r - rmeta) / rmeta)
rec["E_rel"] = float(np.linalg.norm(E) / np.linalg.norm(tgt))
rec["euler_deg"] = float(np.dot(J @ h, tgt) / np.dot(tgt, tgt))
Pm = np.eye(D) - np.outer(nrm, nrm)
JP = J @ Pm; nJP = np.linalg.norm(JP)
rec["JP_F"] = float(nJP)
# Gram 𝒢 : ⟨d_i ⊗ a_i, d_l ⊗ b_l⟩ = (d_iᵀd_l)(a_iᵀ P b_l)
DD = Wd.T @ Wd
AP = [Fm @ Pm for Fm in fams]
nb = len(fams); G = np.empty((nb * Fn, nb * Fn))
for i in range(nb):
    for j in range(i, nb):
        blk = DD * (AP[i] @ fams[j].T)
        G[i*Fn:(i+1)*Fn, j*Fn:(j+1)*Fn] = blk
        if j != i: G[j*Fn:(j+1)*Fn, i*Fn:(i+1)*Fn] = blk.T
del DD, AP
# λ_max par puissance (avant Cholesky, qui écrase G)
x = np.random.default_rng(0).standard_normal(G.shape[0]); x /= np.linalg.norm(x)
for _ in range(40):
    y = G @ x; lmax = np.linalg.norm(y); x = y / lmax
rec["lam_max"] = float(lmax)
# Λ_hᵀ (2F × D) et règle du demi : produits avant factorisation
Lh = np.vstack([c[:, None] * Wd.T for c in radial_cols])            # (nb·F) × D
if half is not None:
    dh = np.concatenate(half); rec["r_half"] = float(np.sqrt(max(dh @ (G @ dh), 0.0)) / nJP)
    rec["half_complete_check"] = float(np.linalg.norm(Lh.T @ dh + E) / np.linalg.norm(E))
try:
    cf = sla.cho_factor(G, lower=True, overwrite_a=True, check_finite=False)
    rec["cholesky_ok"] = True
except np.linalg.LinAlgError:
    rec["cholesky_ok"] = False
    json_out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    json_out[f"{MODEL}_L{LAYER}_p{PROMPT}"] = rec; json.dump(json_out, open(OUT, "w"))
    print(rec); raise SystemExit
# λ_min par itération inverse
x = np.random.default_rng(1).standard_normal(Lh.shape[0]); x /= np.linalg.norm(x)
for _ in range(40):
    y = sla.cho_solve(cf, x, check_finite=False); nx = np.linalg.norm(y); x = y / nx
rec["lam_min"] = float(1.0 / nx)
rec["lam_ratio"] = rec["lam_min"] / rec["lam_max"]
# τ*² = Eᵀ (Λ_h 𝒢⁻¹ Λ_hᵀ)⁻¹ E
X = sla.cho_solve(cf, Lh, check_finite=False)                        # 𝒢⁻¹ Λ_hᵀ
Mm = Lh.T @ X; Mm = 0.5 * (Mm + Mm.T)
w = np.linalg.solve(Mm, E)
tau = float(np.sqrt(max(E @ w, 0.0)))
rec["tau_star"] = tau; rec["r_star"] = tau / nJP
smax_Lh = float(np.linalg.norm(Lh, 2))
rec["lower_bound_rel"] = float(np.sqrt(rec["lam_min"]) * np.linalg.norm(E) / smax_Lh / nJP)
# règle optimale : δ* = −𝒢⁻¹Λ_hᵀ M⁻¹ E ; vérifications (complétude, distorsion)
dstar = -X @ w
rec["opt_complete_check"] = float(np.linalg.norm(Lh.T @ dstar + E) / np.linalg.norm(E))
rec["opt_coef_rel"] = float(np.linalg.norm(dstar) / np.linalg.norm(np.concatenate([a1, a2]) if gated else a1))
rec["secs"] = time.time() - t0
json_out = json.load(open(OUT)) if os.path.exists(OUT) else {}
json_out[f"{MODEL}_L{LAYER}_p{PROMPT}"] = rec
json.dump(json_out, open(OUT, "w"), indent=1)
print(json.dumps(rec))
