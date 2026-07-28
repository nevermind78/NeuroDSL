# ══════════════════════════════════════════════════════════════════════════════
# Analyse de Fourier / perte restreinte-exclue sur le modèle grokké (p=17).
# Port direct de la méthodologie de Nanda et al. (artilce/Grokking_Demo.ipynb,
# cellules 64-137) :
#   1. Identification des fréquences-clés PAR LES DONNÉES (norme de
#      l'embedding token dans la base de Fourier -- cellule 69), PAS copiées
#      de p=113.
#   2. Reconstruction "restreinte" des activations MLP à partir de CES
#      fréquences (cellule 128/131) -> perte restreinte (test de SUFFISANCE :
#      le circuit trigonométrique suffit-il seul à générer les bons logits ?).
#   3. Reconstruction "exclue" (cellule 136/137) -> perte exclue (test de
#      NÉCESSITÉ : sans ces fréquences, le modèle retombe-t-il à la loss de
#      base ?).
#
# Ce script est un banc de développement/débogage (validé ici sur le
# checkpoint p17_seed0 -- qui n'a PAS grokké -- uniquement pour vérifier que
# l'API NeuroDSL utilisée ne plante pas, PAS pour en tirer une conclusion
# scientifique). Les mêmes fonctions seront recopiées dans les cellules du
# notebook final, appliquées au checkpoint qui aura réellement grokké.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, LinearAlgebra, Statistics

const P2         = parse(Int, get(ENV, "GROK_P", "17"))
const DIM2       = 128
const N_HEADS2   = 4
const D_MLP2     = 512
const SEQ_LEN2   = 3
const EQ_TOK2    = P2 + 1

function build_grok_graph2(dev, nsx::Symbol)
    g = NeuroDSL.NeuroGraph(namespace=nsx, device=dev)
    NeuroDSL.set!(g, :token_ids, ones(Int, SEQ_LEN2); atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN2); atom_type=NeuroDSL.Datom, namespace=nsx)
    tok_emb = NeuroDSL.Embedding(P2 + 1, DIM2)(g, :token_ids, :tok; namespace=nsx)
    pos_emb = NeuroDSL.Embedding(SEQ_LEN2, DIM2)(g, :pos_ids, :pos; namespace=nsx)
    x0 = :embed_sum
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(x0, [tok_emb, pos_emb], :add; namespace=nsx))
    ao = NeuroDSL.MultiHeadAttention(DIM2, N_HEADS2; batched=true)(g, x0, :blk_mha; namespace=nsx)
    r1 = :blk_res1
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(r1, [x0, ao], :add; namespace=nsx))
    k1 = 1f0 / sqrt(Float32(DIM2))
    W1 = (NeuroDSL.Backend.rand32(dev, D_MLP2, DIM2) .- 0.5f0) .* (2k1)
    NeuroDSL.set!(g, :blk_mlp_w1, W1; is_param=true, namespace=nsx)
    k2 = 1f0 / sqrt(Float32(D_MLP2))
    W2 = (NeuroDSL.Backend.rand32(dev, DIM2, D_MLP2) .- 0.5f0) .* (2k2)
    NeuroDSL.set!(g, :blk_mlp_w2, W2; is_param=true, namespace=nsx)
    pre = :blk_mlp_pre
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(pre, [r1, :blk_mlp_w1], :matmul;
                       attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=nsx))
    post = :blk_mlp_post
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(post, [pre], :relu; namespace=nsx))
    mo = :blk_mlp_out
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(mo, [post, :blk_mlp_w2], :matmul;
                       attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=nsx))
    out = :blk_out
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(out, [r1, mo], :add; namespace=nsx))
    logits = NeuroDSL.Linear(DIM2, P2; bias=false)(g, out, :unembed; namespace=nsx)
    sel = zeros(Float32, 1, SEQ_LEN2); sel[1, SEQ_LEN2] = 1f0
    NeuroDSL.set!(g, :sel_last, sel; atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:final_logits, [:sel_last, logits], :matmul; namespace=nsx))
    NeuroDSL.set!(g, :final_label, [1]; atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:final_logits, :final_label], :cross_entropy; namespace=nsx))
    return g, r1, mo, post
end

set_input2!(g, nsx, a, b) = begin
    NeuroDSL.set!(g, :token_ids, [a, b, EQ_TOK2]; atom_type=NeuroDSL.Datom, namespace=nsx)
    NeuroDSL.set!(g, :pos_ids, collect(1:SEQ_LEN2); atom_type=NeuroDSL.Datom, namespace=nsx)
end

# ── Collecte des activations pour les p^2 entrées (ordre a extérieur/b intérieur, 0-indexé) ──
function collect_all(g, nsx, r1_sym, post_sym)
    p = P2
    n = p * p
    acts   = zeros(Float32, n, D_MLP2)
    resid  = zeros(Float32, n, DIM2)
    logits = zeros(Float32, n, p)
    labels = zeros(Int, n)
    i = 1
    for a0 in 0:p-1, b0 in 0:p-1
        set_input2!(g, nsx, a0 + 1, b0 + 1)
        NeuroDSL.invalidate_all!(g; namespace=nsx)
        post_val = Array(NeuroDSL.demand!(g, post_sym; namespace=nsx))
        r1_val   = Array(NeuroDSL.demand!(g, r1_sym; namespace=nsx))
        lg       = Array(NeuroDSL.demand!(g, :final_logits; namespace=nsx))
        acts[i, :]  = post_val[end, :]
        resid[i, :] = r1_val[end, :]
        logits[i, :] = lg[1, :]
        labels[i] = ((a0 + b0) % p) + 1
        i += 1
    end
    return acts, resid, logits, labels
end

function loss_fn2(logits::AbstractMatrix{Float32}, labels::AbstractVector{Int})
    n = size(logits, 1)
    tot = 0.0
    for i in 1:n
        row = logits[i, :]
        m = maximum(row)
        e = exp.(row .- m)
        p = e ./ sum(e)
        tot += -log(max(p[labels[i]], 1f-30))
    end
    return tot / n
end

# ── Base de Fourier sur Z/pZ (cellule 64 de Nanda) ───────────────────────────
function fourier_basis_matrix(p::Int)
    names = ["Constant"]
    rows = Vector{Vector{Float32}}()
    push!(rows, ones(Float32, p))
    for freq in 1:(p ÷ 2)
        push!(rows, Float32[sin(2f0 * Float32(pi) * freq * k / p) for k in 0:p-1]); push!(names, "Sin $freq")
        push!(rows, Float32[cos(2f0 * Float32(pi) * freq * k / p) for k in 0:p-1]); push!(names, "Cos $freq")
    end
    F = permutedims(hcat(rows...))                          # (n_components, p)
    F = F ./ sqrt.(sum(F .^ 2, dims=2))
    return F, names
end

# ── Fréquences-clés identifiées à partir de l'embedding (cellule 69) ─────────
function identify_key_freqs(g, nsx; top_k=4)
    p = P2
    F, _names = fourier_basis_matrix(p)
    WE = Array(NeuroDSL.node(g, :tok_E; namespace=nsx).value)[1:p, :]   # (p, dim) -- exclut le token "="
    fourier_embed = F * WE                                              # (n_components, dim)
    comp_norms = sqrt.(sum(fourier_embed .^ 2, dims=2))[:, 1]
    freq_scores = Dict{Int,Float32}()
    for freq in 1:(p ÷ 2)
        si = 2freq; ci = 2freq + 1     # indices 1-based dans F (1=Constant, 2=Sin1, 3=Cos1, ...)
        freq_scores[freq] = max(comp_norms[si], comp_norms[ci])
    end
    ranked = sort(collect(freq_scores), by = x -> -x[2])
    key_freqs = [f for (f, s) in ranked[1:min(top_k, length(ranked))]]
    return key_freqs, freq_scores
end

# ── Vecteurs cos/sin(freq*(a+b)) sur les p^2 entrées (ordre a extérieur/b intérieur) ──
function angle_vec(p, freq; f=cos)
    n = p * p
    v = zeros(Float32, n)
    i = 1
    for a0 in 0:p-1, b0 in 0:p-1
        v[i] = Float32(f(freq * 2 * pi / p * (a0 + b0)))
        i += 1
    end
    return v ./ norm(v)
end

# ── Reconstruction restreinte (cellule 128/131) : GARDE seulement les
#    fréquences-clés + le terme constant (moyenne) -- test de SUFFISANCE. ────
function restricted_neuron_acts(acts, key_freqs, p)
    approx = repeat(mean(acts, dims=1), size(acts, 1), 1)
    for freq in key_freqs
        cvec = angle_vec(p, freq; f=cos)
        svec = angle_vec(p, freq; f=sin)
        approx .+= cvec * (cvec' * acts)
        approx .+= svec * (svec' * acts)
    end
    return approx
end

# ── Reconstruction exclue (cellule 136/137) : RETIRE les fréquences-clés
#    (PAS de terme constant retiré -- test de NÉCESSITÉ). ────────────────────
function excluded_neuron_acts(acts, key_freqs, p)
    approx = zeros(Float32, size(acts))
    for freq in key_freqs
        cvec = angle_vec(p, freq; f=cos)
        svec = angle_vec(p, freq; f=sin)
        approx .+= cvec * (cvec' * acts)
        approx .+= svec * (svec' * acts)
    end
    return acts .- approx
end

function get_restricted_loss(g, nsx, acts, logits, labels, key_freqs, W2, WU)
    p = P2
    ra = restricted_neuron_acts(acts, key_freqs, p)
    restricted_logits = ra * W2' * WU'
    # correction de biais -- aligne les moyennes (cellule 131), la reconstruction
    # restreinte ne porte QUE la variation MLP, pas le biais implicite complet.
    restricted_logits .+= mean(logits, dims=1) .- mean(restricted_logits, dims=1)
    return loss_fn2(restricted_logits, labels)
end

function get_excluded_loss(g, nsx, acts, resid, labels, key_freqs, W2, WU)
    p = P2
    ea = excluded_neuron_acts(acts, key_freqs, p)
    excluded_resid = ea * W2' .+ resid
    excluded_logits = excluded_resid * WU'
    return loss_fn2(excluded_logits, labels)
end

# ── Banc de test mécanique (sur le checkpoint négatif, juste pour valider l'API) ──
if abspath(PROGRAM_FILE) == @__FILE__
    dev = NeuroDSL.Backend.CUDADevice()
    nsx = :grok_reload
    g, r1_sym, mo_sym, post_sym = build_grok_graph2(dev, nsx)
    ckpt = joinpath(@__DIR__, "grok_ckpt", "p17_seed0")
    NeuroDSL.load_graph!(g, nsx, ckpt; overwrite=true)
    println("Graphe rechargé depuis $ckpt")

    acts, resid, logits, labels = collect_all(g, nsx, r1_sym, post_sym)
    println("acts=", size(acts), " resid=", size(resid), " logits=", size(logits))
    println("Perte totale recalculée (contrôle) : ", loss_fn2(logits, labels))

    key_freqs, scores = identify_key_freqs(g, nsx; top_k=4)
    println("Fréquences-clés (data-driven) : ", key_freqs)
    println("Scores par fréquence : ", sort(collect(scores), by=x->-x[2]))

    W2 = Array(NeuroDSL.node(g, :blk_mlp_w2; namespace=nsx).value)
    WU = Array(NeuroDSL.node(g, :unembed_W; namespace=nsx).value)
    rl = get_restricted_loss(g, nsx, acts, logits, labels, key_freqs, W2, WU)
    el = get_excluded_loss(g, nsx, acts, resid, labels, key_freqs, W2, WU)
    println("Perte restreinte (sufficiency, checkpoint NON grokké -- attendu : pas informatif) : ", rl)
    println("Perte exclue     (necessity,   checkpoint NON grokké -- attendu : pas informatif) : ", el)
end
