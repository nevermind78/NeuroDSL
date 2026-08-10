# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic ISOLÉ et RAPIDE -- ne rejoue PAS Phase A (backbone déjà sauvegardé
# dans graft_experiment_v2_export/base par le run précédent). Reproduit UNIQUEMENT
# la branche 1 (diag_frozen) pendant 300 pas au lieu de 4000, avec des mesures
# supplémentaires à chaque pas de log : loss d'entraînement, ||R(x;theta)||
# (sortie brute du greffon avant le gate alpha), alpha, norme du gradient de
# alpha, recovery. Objectif : localiser le moment exact et la cause de
# l'effondrement observé dans real_llm_graft_experiment_v2.jl (recovery finale
# catastrophique sur les 4 branches).
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Statistics, Printf, LinearAlgebra

dev = NeuroDSL.Backend.CUDADevice()
export_dir = joinpath(@__DIR__, "graft_experiment_v2_export")
isdir(export_dir) || error("Backbone Phase A introuvable -- lancez d'abord real_llm_graft_experiment_v2.jl")

# ── Corpus (déjà en cache local, pas de téléchargement) ─────────────────────
const CORPUS_PATH = joinpath(@__DIR__, "data", "tinyshakespeare", "input.txt")
text = read(CORPUS_PATH, String)
function build_char_tokenizer(text::String)
    chars = sort(unique(collect(text)))
    stoi = Dict(c => i for (i, c) in enumerate(chars))
    return chars, stoi
end
encode(text::AbstractString, stoi::Dict{Char,Int}) = [stoi[c] for c in text]
decode(ids::AbstractVector{<:Integer}, chars::Vector{Char}) = String(chars[ids])
chars, stoi = build_char_tokenizer(text)
vocab_size = length(chars)
data = encode(text, stoi)
n_total = length(data)
n_train = floor(Int, 0.9 * n_total)
train_ids = data[1:n_train]
val_ids   = data[n_train+1:end]

block_size = 256
dim        = 256
n_heads    = 4
hidden_dim = 512
n_layers   = 4
d_head     = dim ÷ n_heads

function sample_window(rng, ids::Vector{Int}, block_size::Int)
    i = rand(rng, 1:(length(ids) - block_size))
    tokens = ids[i:i+block_size-1]
    labels = ids[i+1:i+block_size]
    return tokens, labels
end

# ── Reconstruire la MÊME fenêtre cible (LUCENTIO, gap_pos=104, j=112) SANS
# rejouer tout le screening -- juste besoin d'UNE fenêtre corrompue valide sur
# CE backbone, peu importe si le token de corruption diffère bit-à-bit de
# l'original (n'affecte pas le diagnostic recherché). ──────────────────────
function all_speaker_headers(val_ids::Vector{Int}, chars::Vector{Char}; min_name_len::Int=4)
    text_val = decode(val_ids, chars)
    headers = NamedTuple[]
    for m in eachmatch(r"\n([A-Z][A-Z ]{2,})\:", text_val)
        name = String(strip(m.captures[1]))
        length(name) >= min_name_len || continue
        push!(headers, (; pos=m.offset + 1, name))
    end
    return headers
end
function build_candidates(headers, block_size::Int; k::Int=3, margin::Int=5)
    candidates = NamedTuple[]
    n = length(headers)
    for i in 1:n-1
        for jx in i+1:n
            gap = headers[jx].pos - headers[i].pos
            gap > block_size - 20 && break
            headers[i].name != headers[jx].name && continue
            kk = min(k, length(headers[i].name) - 1)
            kk < 1 && continue
            window_start = headers[i].pos - margin
            window_start < 1 && continue
            p1 = headers[i].pos - window_start + 1
            p2 = headers[jx].pos - window_start + 1
            p2 + kk > block_size && continue
            has_intervening_different = any(headers[m2].pos > headers[i].pos && headers[m2].pos < headers[jx].pos &&
                                             headers[m2].name != headers[i].name for m2 in i+1:jx-1)
            variant = has_intervening_different ? :long_gap : :adjacent
            push!(candidates, (; window_start, p1, p2, k=kk, name=headers[i].name, gap, variant))
        end
    end
    return candidates
end
headers = all_speaker_headers(val_ids, chars)
candidates_raw = build_candidates(headers, block_size)
lucentio_candidates = filter(c -> c.name == "LUCENTIO" && c.gap == 104 && (c.p2 + c.k - 1) == 112, candidates_raw)
target_c = first(lucentio_candidates)

tokens_clean = val_ids[target_c.window_start:target_c.window_start+block_size-1]
j_target = target_c.p2 + target_c.k - 1
tokens_corrupt = copy(tokens_clean)
orig_id = tokens_corrupt[target_c.p1 + target_c.k]
new_id = orig_id == vocab_size ? 1 : orig_id + 1   # déterministe, peu importe lequel
tokens_corrupt[target_c.p1 + target_c.k] = new_id
@printf("Fenêtre LUCENTIO reconstruite : j_target=%d, token corrompu %d -> %d\n", j_target, orig_id, new_id)

# ── Charger le backbone Phase A sauvegardé ──────────────────────────────────
ns_b = :diag_quick
g_b = NeuroDSL.NeuroGraph(namespace=ns_b, device=dev)
NeuroDSL.load_graph!(g_b, ns_b, joinpath(export_dir, "base"))
logits_sym = :lm_head_out   # nom standard produit par Linear(...,:lm_head)

# Vérifier le nom exact du symbole logits dans le graphe chargé
cands_logits = filter(s -> occursin("lm_head", String(s)), collect(keys(g_b.nodes[ns_b])))
println("Symboles logits candidats : ", cands_logits)
logits_sym = Symbol(:lm_head, :_out) in cands_logits ? Symbol(:lm_head, :_out) : cands_logits[1]
println("logits_sym choisi = ", logits_sym)

# ── Référence clean/corrupted sur CE backbone (avant tout greffon) ──────────
NeuroDSL.set!(g_b, :token_ids, tokens_clean; atom_type=NeuroDSL.Datom, namespace=ns_b)
NeuroDSL.set!(g_b, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns_b)
NeuroDSL.invalidate_all!(g_b; namespace=ns_b)
clean_output_ref = copy(Array(NeuroDSL.demand!(g_b, logits_sym; namespace=ns_b)))

NeuroDSL.set!(g_b, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns_b)
NeuroDSL.invalidate_all!(g_b; namespace=ns_b)
corrupted_output_ref = copy(Array(NeuroDSL.demand!(g_b, logits_sym; namespace=ns_b)))

denom0 = norm(corrupted_output_ref[j_target,:] .- clean_output_ref[j_target,:])
@printf("||corrupted_ref - clean_ref|| à j_target = %.6f  (denominateur de recovery_metric)\n", denom0)

row_metric_ref(out) = NeuroDSL.recovery_metric(Array(out)[j_target:j_target, :], clean_output_ref[j_target:j_target, :], corrupted_output_ref[j_target:j_target, :])

# ── Greffe (identique à branch1 : DOMINANT_SITE, gelé) ──────────────────────
DOMINANT_SITE = :layer_1_mha_ao_h2
GRAFT_DIM, GRAFT_HEADS, GRAFT_HIDDEN = d_head, 2, 128
Random.seed!(7001)
NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.seed!(7001)
prefix = Symbol(:shadow_diag_frozen_, DOMINANT_SITE)
new_out, handle = NeuroDSL.graft_shadow_block!(g_b, ns_b, DOMINANT_SITE, GRAFT_DIM, GRAFT_HEADS, GRAFT_HIDDEN;
                                                alpha0=0f0, zero_out_proj=false, prefix=prefix)

function freeze_backbone!(g, ns, keep_prefix)
    n = 0
    for (sym, nd) in g.nodes[ns]
        nd.is_param || continue
        if keep_prefix !== nothing && startswith(String(sym), String(keep_prefix))
            continue
        end
        nd.is_param = false
        n += 1
    end
    return n
end
n_frozen = freeze_backbone!(g_b, ns_b, handle.prefix)
ps = NeuroDSL.params(g_b; namespace=ns_b)
@printf("params gelés=%d, entraînables=%d (dont alpha)\n", n_frozen, length(ps))

# ── Correctif 1 : theta du greffon (tout sauf alpha) reçoit un LR nettement
# plus petit + une weight decay non nulle, pour borner sa dérive sous AdamW.
# alpha garde son LR/wd actuels (le gate lui-même n'a pas montré de problème).
LR_B      = 1f-3
LR_THETA_DIV = parse(Float32, get(ENV, "LR_THETA_DIV", "20"))
WD_THETA     = parse(Float32, get(ENV, "WD_THETA", "0.01"))
LR_THETA  = LR_B / LR_THETA_DIV
alpha_idx = findall(p -> p.name == handle.alpha_sym, ps)
theta_idx = findall(p -> p.name != handle.alpha_sym && startswith(String(p.name), String(handle.prefix)), ps)
@assert length(alpha_idx) + length(theta_idx) == length(ps) "partition alpha/theta incomplète -- vérifier le préfixe"
println("  -> ", length(alpha_idx), " param(s) alpha (LR=", LR_B, ", wd=0), ",
        length(theta_idx), " param(s) theta (LR=", LR_THETA, ", wd=", WD_THETA, ")")

m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
rng_cont = MersenneTwister(9000)

N_STEPS = 300
LOG_EVERY = 5

@printf("\n%6s | %10s | %10s | %10s | %10s | %10s\n", "step", "train_loss", "recovery", "alpha", "||R(x)||", "|grad_alpha|")
for t in 1:N_STEPS
    tokens, labels = sample_window(rng_cont, train_ids, block_size)
    NeuroDSL.set!(g_b, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns_b)
    NeuroDSL.set!(g_b, :pos_ids, collect(1:block_size); atom_type=NeuroDSL.Datom, namespace=ns_b)
    NeuroDSL.set!(g_b, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns_b)
    NeuroDSL.invalidate_all!(g_b; namespace=ns_b)
    loss_val = NeuroDSL.demand!(g_b, :loss; namespace=ns_b)
    train_loss = Float64(sum(Array(loss_val)))
    NeuroDSL.backward_graph_sparse!(g_b, :loss; namespace=ns_b, prune_frozen=true)
    grad_alpha_norm = Float64(norm(Array(NeuroDSL.node(g_b, handle.alpha_sym; namespace=ns_b).gradient)))
    if !isempty(alpha_idx)
        NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps[alpha_idx]], [p.gradient for p in ps[alpha_idx]],
                                     m1s[alpha_idx], m2s[alpha_idx], LR_B, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
    end
    if !isempty(theta_idx)
        NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps[theta_idx]], [p.gradient for p in ps[theta_idx]],
                                     m1s[theta_idx], m2s[theta_idx], LR_THETA, 0.9f0, 0.999f0, 1f-8, t, 1f0, WD_THETA)
    end
    NeuroDSL.invalidate_all!(g_b; namespace=ns_b)

    if t % LOG_EVERY == 0 || t == 1
        # Recovery sur la fenêtre corrompue diagnostique
        NeuroDSL.set!(g_b, :token_ids, tokens_corrupt; atom_type=NeuroDSL.Datom, namespace=ns_b)
        NeuroDSL.invalidate_all!(g_b; namespace=ns_b)
        out = NeuroDSL.demand!(g_b, logits_sym; namespace=ns_b)
        r = row_metric_ref(out)
        alpha_val = Float64(Array(NeuroDSL.node(g_b, handle.alpha_sym; namespace=ns_b).value)[1])
        R_val = Array(NeuroDSL.demand!(g_b, handle.R_sym; namespace=ns_b))
        R_norm = norm(R_val) / sqrt(length(R_val))  # RMS par élément, comparable entre tenseurs
        any(isnan, R_val) && println("   !!! NaN détecté dans R(x;theta) au pas ", t)
        isnan(train_loss) && println("   !!! NaN détecté dans train_loss au pas ", t)
        @printf("%6d | %10.4f | %10.4f | %10.4f | %10.4f | %10.6f\n", t, train_loss, r, alpha_val, R_norm, grad_alpha_norm)
        NeuroDSL.invalidate_all!(g_b; namespace=ns_b)
    end
end
println("\nTerminé.")
