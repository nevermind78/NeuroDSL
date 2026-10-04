# ══════════════════════════════════════════════════════════════════════════════
# WIND-3 -- outils communs (vent HORIZONTAL : influence entre positions)
#
# Tout est défini ICI, au niveau du script (aucune modification de src/) :
#   * nœuds de capture  w3cap_x{k} = layer_k_out + z_k   (z_k paramètre nul, n×D)
#                       w3cap_r{l} = layer_l_res1 + zr_l (idem, pour le cotangent de
#                       la sortie d'attention) -- x + 0f0 est bit-exact ; le gradient
#                       accumulé dans z_k EST ∂R/∂x_k pour TOUTES les positions ;
#   * nœuds stop-gradient w3sg_{l}_{h} entre pr_h (probabilités d'attention) et
#     :batched_pv -- forward = copie exacte ; backward = dy (T) ou rien (A, AN) ;
#   * :rmsnorm_frozen (dénominateurs gelés à leur valeur propre ; variante AN) ;
#   * lecture R = <final_norm_out[n,:], w_read>, w_read = W_U[top1] − W_U[top2]
#     (= différence de logits top-1 − top-2, comme WIND-1) ;
#   * sonde P = <x_{n,28}, u> (u quelconque) pour les esquisses de Jacobiennes.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, LinearAlgebra, Statistics

const W3_MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const W3_CKPT      = joinpath(W3_MODEL_DIR, "qwen2_neurodsl")
const NS = :qwen2
const L  = 28
const D  = 1536
const NH = 12

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())
gpu_used_mib() = try
    parse(Int, strip(split(read(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`, String), '\n')[1]))
catch
    -1
end

res_sym(k) = k == 0 ? :tok_out : Symbol("layer_", k, "_out")
r1_sym(l)  = Symbol("layer_", l, "_res1")
mha_sym(l) = Symbol("layer_", l, "_mha_output_out")
pr_sym(l, h) = Symbol("layer_", l, "_mha_pr_h", h)
ao3_sym(l) = Symbol("layer_", l, "_mha_ao3")
capx(k)  = Symbol("w3cap_x", k);  zx(k) = Symbol("w3z_x", k)
capr(l)  = Symbol("w3cap_r", l);  zr(l) = Symbol("w3z_r", l)
sg_sym(l, h) = Symbol("w3sg_", l, "_", h)
const NORM_SYMS = vcat([Symbol("layer_", l, "_norm", j, "_out") for l in 1:L for j in 1:2], [:final_norm_out])

const SG_ON      = Ref(false)
const FROZEN_RMS = Dict{Symbol,Any}()
const READ       = Dict{Symbol,Any}()     # :n, :w (CuVector Float32), :u (CuVector Float32)

function register_w3_ops!()
    haskey(NeuroDSL.CUSTOM_OPS, :w3_sg) && return
    # --- stop-gradient sur les probabilités d'attention -----------------------
    NeuroDSL.register_op!(:w3_sg, (dev, out, inp, attrs, osym, onode, ctx) -> (out .= inp[1]))
    NeuroDSL.CUSTOM_SHAPE_RULES[:w3_sg] = (i, a) -> size(i[1])
    NeuroDSL.CTX_REBUILD[:w3_sg] = (dev, rule, nd, iv) -> Dict{Symbol,Any}()
    NeuroDSL.GRAD_RULES[:w3_sg] = (dev, dy, ctx, inp) -> (SG_ON[] ? (nothing,) : (copy(dy),))
    # --- lecture R (différence de logits via la norme finale) -----------------
    NeuroDSL.register_op!(:w3_readout, (dev, out, inp, attrs, osym, onode, ctx) -> begin
        n = READ[:n]
        out .= vec(sum(view(inp[1], n:n, :) .* reshape(READ[:w], 1, :); dims=2))
    end)
    NeuroDSL.CUSTOM_SHAPE_RULES[:w3_readout] = (i, a) -> (1,)
    NeuroDSL.CTX_REBUILD[:w3_readout] = (dev, rule, nd, iv) -> Dict{Symbol,Any}()
    NeuroDSL.GRAD_RULES[:w3_readout] = (dev, dy, ctx, inp) -> begin
        n = READ[:n]; dx = similar(inp[1]); fill!(dx, 0f0)
        view(dx, n:n, :) .= reshape(READ[:w], 1, :) .* reshape(dy, 1, 1)
        (dx,)
    end
    # --- sonde P = <x_{n,28}, u> ----------------------------------------------
    NeuroDSL.register_op!(:w3_probe, (dev, out, inp, attrs, osym, onode, ctx) -> begin
        n = READ[:n]
        out .= vec(sum(view(inp[1], n:n, :) .* reshape(READ[:u], 1, :); dims=2))
    end)
    NeuroDSL.CUSTOM_SHAPE_RULES[:w3_probe] = (i, a) -> (1,)
    NeuroDSL.CTX_REBUILD[:w3_probe] = (dev, rule, nd, iv) -> Dict{Symbol,Any}()
    NeuroDSL.GRAD_RULES[:w3_probe] = (dev, dy, ctx, inp) -> begin
        n = READ[:n]; dx = similar(inp[1]); fill!(dx, 0f0)
        view(dx, n:n, :) .= reshape(READ[:u], 1, :) .* reshape(dy, 1, 1)
        (dx,)
    end
    # --- RMSNorm à dénominateur gelé (valeur propre) ---------------------------
    NeuroDSL.register_op!(:rmsnorm_frozen, (dev, out, inp, attrs, osym, onode, ctx) -> begin
        out .= inp[1] .* FROZEN_RMS[osym] .* reshape(inp[2], 1, :)
    end)
    NeuroDSL.CUSTOM_SHAPE_RULES[:rmsnorm_frozen] = (i, a) -> size(i[1])
    NeuroDSL.CTX_REBUILD[:rmsnorm_frozen] = (dev, rule, nd, iv) -> Dict{Symbol,Any}(:rms => FROZEN_RMS[rule.output])
    NeuroDSL.GRAD_RULES[:rmsnorm_frozen] = (dev, dy, ctx, inp) -> begin
        (dy .* ctx[:rms] .* reshape(inp[2], 1, :), nothing)
    end
    return
end

function _redirect_consumers!(g, src::Symbol, dst::Symbol)
    cons = copy(get(NeuroDSL._consumers_index!(g, NS), src, Symbol[]))
    for cs in cons
        r = g.rules[NS][cs]
        newin = [s == src ? dst : s for s in r.inputs]
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(cs, newin, r.op; attrs=r.attrs, namespace=NS,
                                                atom_type=r.atom_type))
    end
    return cons
end

"""Charge Qwen et installe l'instrumentation. Retourne (g, dev, orig_norm_rules, W_U)."""
function load_instrumented()
    register_w3_ops!()
    dev = NeuroDSL.Backend.CUDADevice()
    g = NeuroDSL.NeuroGraph(namespace=NS, device=dev)
    NeuroDSL.load_graph!(g, NS, W3_CKPT)
    reclaim()
    for (s, nd) in g.nodes[NS]            # aucun poids entraînable : pas de gradient de poids
        nd.is_param && (nd.is_param = false)
    end
    # placeholders (taille réelle posée par set_prompt!)
    for k in 0:L
        NeuroDSL.set!(g, zx(k), zeros(Float32, 1, D); is_param=true, namespace=NS)
        cons = _redirect_consumers!(g, res_sym(k), capx(k))
        isempty(cons) && error("aucun consommateur pour $(res_sym(k))")
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(capx(k), [res_sym(k), zx(k)], :add; namespace=NS))
    end
    for l in 1:L
        NeuroDSL.set!(g, zr(l), zeros(Float32, 1, D); is_param=true, namespace=NS)
        cons = _redirect_consumers!(g, r1_sym(l), capr(l))
        length(cons) == 2 || error("res1 couche $l : consommateurs inattendus $cons")
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(capr(l), [r1_sym(l), zr(l)], :add; namespace=NS))
        for h in 1:NH
            c = get(NeuroDSL._consumers_index!(g, NS), pr_sym(l, h), Symbol[])
            (length(c) == 1 && c[1] == ao3_sym(l)) || error("pr_h $l,$h : consommateurs inattendus $c")
            _redirect_consumers!(g, pr_sym(l, h), sg_sym(l, h))
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(sg_sym(l, h), [pr_sym(l, h)], :w3_sg; namespace=NS))
        end
    end
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:w3_R, [:final_norm_out], :w3_readout; namespace=NS))
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:w3_P, [capx(L)], :w3_probe; namespace=NS))
    orig = Dict(s => g.rules[NS][s] for s in NORM_SYMS)     # APRÈS recâblage
    all(r.op == :rmsnorm for r in values(orig)) || error("règle de norme inattendue")
    W_U = NeuroDSL.node(g, :lm_head_W; namespace=NS).value
    return g, dev, orig, W_U
end

function swap_norms!(g, orig, frozen::Bool)
    for s in NORM_SYMS
        r = orig[s]
        cur = g.rules[NS][s]
        want = frozen ? :rmsnorm_frozen : :rmsnorm
        cur.op == want && continue
        NeuroDSL.addrule!(g, frozen ? NeuroDSL.GraphRule(s, r.inputs, :rmsnorm_frozen; attrs=r.attrs,
                                                          namespace=NS, atom_type=r.atom_type) : r)
    end
end

"""Pose le prompt (ids 1-indexés), les params de capture n×D, et invalide tout."""
function set_prompt!(g, ids::Vector{Int})
    n = length(ids)
    NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=NS)
    NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=NS)
    for k in 0:L; NeuroDSL.set!(g, zx(k), zeros(Float32, n, D); is_param=true, namespace=NS); end
    for l in 1:L; NeuroDSL.set!(g, zr(l), zeros(Float32, n, D); is_param=true, namespace=NS); end
    NeuroDSL.invalidate_all!(g; namespace=NS)
    READ[:n] = n
    return n
end

"""Copie des valeurs de tous les nœuds À RÈGLE (activations) -- les poids ne sont plus
marqués is_param (pour qu'aucun gradient de poids ne soit calculé), donc
capture_activations les copierait : on n'en veut pas (6 Go)."""
capture_rule_nodes(g) = Dict{Symbol,Any}(s => copy(nd.value) for (s, nd) in g.nodes[NS]
                                          if nd.value !== nothing && haskey(g.rules[NS], s))

set_readout!(dev, w::Vector{Float64}) = (READ[:w] = NeuroDSL.Backend.to_device(dev, Float32.(w)); nothing)
set_probe!(dev, u::Vector{Float64})   = (READ[:u] = NeuroDSL.Backend.to_device(dev, Float32.(u)); nothing)

"""R en Float64, lu sur la ligne n de final_norm_out (précision de lecture maximale)."""
function readout64(g, w64::Vector{Float64})
    n = READ[:n]
    h = NeuroDSL.demand!(g, :final_norm_out; namespace=NS)[n:n, :]
    return dot(Float64.(vec(Array(h))), w64)
end

"""Un backward complet pour la variante V ∈ (:T,:A,:AN) depuis `seed` (:w3_R ou :w3_P).
Retourne (Gx, Gr) : Gx[k+1] = ∂seed/∂x_k (n×D, k=0..L), Gr[l] = ∂seed/∂res1_l (n×D)."""
function variant_grads!(g, orig, V::Symbol; seed::Symbol=:w3_R)
    SG_ON[] = V in (:A, :AN)
    swap_norms!(g, orig, V == :AN)
    NeuroDSL.demand!(g, seed; namespace=NS)
    NeuroDSL.backward_graph!(g, seed; namespace=NS, prune_frozen=true)
    Gx = [Float64.(Array(NeuroDSL.node(g, zx(k); namespace=NS).gradient)) for k in 0:L]
    Gr = [Float64.(Array(NeuroDSL.node(g, zr(l); namespace=NS).gradient)) for l in 1:L]
    SG_ON[] = false
    return Gx, Gr
end

"""JVP centrée de R le long de v à la ligne i de x_s, pour la variante V
(A/AN : les 12×(L−s) pr_h en aval épinglés à `cache` ; AN : normes gelées déjà en place)."""
function fd_jvp(g, cache, V, s, i, v::Vector{Float64}, eps::Float64, w64)
    n = READ[:n]; src = res_sym(s)
    dev = g.device
    r = zeros(2)
    for (j, sg) in enumerate((1.0, -1.0))
        P = zeros(Float32, n, D); P[i, :] .= Float32.(sg * eps .* v)
        NeuroDSL.patch_node!(g, src, Dict(src => cache[src] .+ NeuroDSL.Backend.to_device(dev, P)); namespace=NS)
        if V in (:A, :AN)
            for l in s+1:L, h in 1:NH
                NeuroDSL.patch_node!(g, pr_sym(l, h), cache; namespace=NS)
            end
        end
        r[j] = readout64(g, w64)
    end
    NeuroDSL.patch_node!(g, src, cache; namespace=NS)
    # pas effectif (arrondi fp32 de la perturbation) : normalisé comme WIND-1
    return (r[1] - r[2]) / (2eps)
end

"""Knockout de l'arête (n ← i) à la couche l, toutes têtes. mode :remove (p[n,i]=0, sans
renormalisation) ou :mask (−∞ pré-softmax ⇔ p[n,j]/(1−p[n,i]), protocole Geva et al. 2023).
layers : couches bloquées (défaut : [l]). Retourne (R, δ_l) où δ_l = Δ sortie MHA[n] à la couche l."""
function knockout(g, cleanP::Dict, cache, layers, i, mode::Symbol, w64)
    n = READ[:n]
    for l in layers, h in 1:NH
        P = copy(cleanP[(l, h)])
        if mode == :remove
            P[n, i] = 0f0
        elseif mode == :mask
            p = Float64.(P[n, :]); pi_ = p[i]; p[i] = 0.0; p ./= (1 - pi_); P[n, :] .= Float32.(p)
        elseif mode != :none
            error("mode $mode")
        end
        NeuroDSL.patch_node!(g, pr_sym(l, h), Dict(pr_sym(l, h) => P); namespace=NS)
    end
    R = readout64(g, w64)
    l0 = first(layers)
    δ = Float64.(vec(Array(NeuroDSL.demand!(g, mha_sym(l0); namespace=NS)[n:n, :]))) .-
        Float64.(vec(Array(cache[mha_sym(l0)][n:n, :])))
    for l in layers, h in 1:NH
        NeuroDSL.patch_node!(g, pr_sym(l, h), cache; namespace=NS)
    end
    return R, δ
end

erank2(σ) = (p = σ .^ 2 ./ sum(σ .^ 2); p = p[p .> 0]; exp(-sum(p .* log.(p))))
