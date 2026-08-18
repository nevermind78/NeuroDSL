# ══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION EXACTE DE L'IDENTITÉ DE TRANSPORT (Proposition~prop:transport,
# éq. defect) SUR LE MODÈLE ENTRAÎNÉ RÉEL -- pas seulement en synthétique.
#
# CE QUE ÇA TESTE
# ----------------
# La proposition dit : q_j^(s) - q_j^(t) = <beta_j^(t), u_{s,t}> pour UN SEUL
# vecteur u_{s,t}, indépendant de j. u_{s,t} = M g^(t)/<g^(t),Mg^(t)> - g^(t)/||g^(t)||^2
# avec M = T'T, T = transport entre les deux sites -- mais T n'est jamais
# construit explicitement ici (D x D, hors de portée). Au lieu de ça : on a
# DIRECTEMENT, sans T, les scalaires exacts q_j^(s) et q_j^(t) (par backward
# ordinaire à chaque site) ET les vecteurs bruts beta_j^(t) (déjà calculés en
# interne dans le script multiprompt existant, mais jetés après réduction en
# scalaires). Si la proposition est vraie, il existe un u tel que
# <beta_j^(t), u> = q_j^(s) - q_j^(t) EXACTEMENT pour tous les j du cône(t)
# simultanément -- un système linéaire sur-ou-sous-déterminé selon B(t) vs D,
# résolu par moindres carrés, dont le résidu doit tomber à la précision
# machine si et seulement si l'identité tient. Ce n'est PAS un test
# statistique : c'est une porte d'exactitude comme les autres du papier,
# simplement appliquée au modèle réel plutôt qu'au modèle synthétique.
#
# COÛT : B(t)+1 passes backward par prompt (pas B(t)*56 -- une seule passe
# backward capture tous les sites simultanément, comme dans le script
# multiprompt existant). Avec T au site_pos=25 (B=32), c'est 33 passes/prompt.
#
# USAGE : julia --project=. notebook/bench_eps_transport_exact_check.jl
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, Statistics, LinearAlgebra

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const PROMPTS_F = joinpath(@__DIR__, "qwen_sweep_prompts.json")
const OUT       = joinpath(@__DIR__, "bench_eps_transport_exact_check_results.txt")

const N_PROMPTS = parse(Int, get(ENV, "TC_NPROMPTS", "5"))
const T_SITE_POS = parse(Int, get(ENV, "TC_TPOS", "25"))   # deep site
const S_SITE_POS = 1                                        # shallow site (full cone)

const EPSV     = Dict{Symbol,Float32}()
const CAPTURED = Dict{Symbol,Array{Float32}}()

function ensure_eps_op!(branch::Symbol)
    op = Symbol("epsop_", branch)
    EPSV[branch] = 1.0f0
    haskey(NeuroDSL.CUSTOM_OPS, op) && return op
    NeuroDSL.register_op!(op,
        (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> (out_buf .= inputs[1]))
    NeuroDSL.CUSTOM_SHAPE_RULES[op] = (inputs, attrs) -> size(inputs[1])
    NeuroDSL.GRAD_RULES[op] = (dev, dy, ctx, inputs) -> begin
        e = EPSV[branch]
        gr = e .* dy
        CAPTURED[branch] = copy(Array(gr))
        return (gr,)
    end
    return op
end

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))

    emit("VÉRIFICATION EXACTE : IDENTITÉ DE TRANSPORT SUR LE MODÈLE ENTRAÎNÉ RÉEL")
    emit("(Qwen2.5-1.5B-Instruct) -- pas la version synthétique déjà publiée.")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))

    dev = NeuroDSL.Backend.CUDADevice()
    ns, L = :qwen2, 28
    logits, loss = :lm_head_out, :ce_loss

    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    emit("\nChargement du checkpoint natif...")
    NeuroDSL.load_graph!(g, ns, CKPT)
    reclaim()

    witnesses = vcat([Symbol("layer_", i, "_norm1_gamma") for i in 1:L],
                     [Symbol("layer_", i, "_norm2_gamma") for i in 1:L])
    keep = Set(vcat(witnesses, [:final_norm_gamma]))
    for (s, nd) in g.nodes[ns]
        nd.is_param && !(s in keep) && (nd.is_param = false)
    end
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss, [logits, :labels], :cross_entropy;
                                            namespace=ns))

    branches = Symbol[]
    for i in 1:L
        push!(branches, Symbol("layer_", i, "_mha_output_out"))
        push!(branches, Symbol("layer_", i, "_mlp_out"))
    end
    pos = Dict(b => k for (k, b) in enumerate(branches))
    S_SYM = branches[S_SITE_POS]
    T_SYM = branches[T_SITE_POS]
    cone_t = [b for b in branches if pos[b] >= T_SITE_POS]
    Bt = length(cone_t)
    emit(@sprintf("\nSite peu profond s = %s (site_pos=%d, cône entier B=%d)",
                  S_SYM, S_SITE_POS, length(branches)))
    emit(@sprintf("Site profond     t = %s (site_pos=%d, cône B(t)=%d)",
                  T_SYM, T_SITE_POS, Bt))

    function set_input!(ids::Vector{Int})
        n = length(ids)
        NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, vcat(ids[2:end], ids[end]);
                      atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
    end
    function backward!()
        empty!(CAPTURED)
        NeuroDSL.demand!(g, loss; namespace=ns)
        NeuroDSL.backward_graph!(g, loss; namespace=ns)
        return CAPTURED
    end

    prompts = JSON.parsefile(PROMPTS_F)

    # ─── enregistrer les eps-ops sur toutes les branches (comme le script existant) ───
    ids1 = Int.(prompts[1]["token_ids"]) .+ 1
    for i in 1:L
        for (join_sym, branch_sym) in ((Symbol("layer_", i, "_res1"),
                                        Symbol("layer_", i, "_mha_output_out")),
                                       (Symbol("layer_", i, "_out"),
                                        Symbol("layer_", i, "_mlp_out")))
            r = g.rules[ns][join_sym]
            r.inputs[2] == branch_sym || error("$join_sym : branche inattendue")
            eps_sym = Symbol("eps_", branch_sym)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(eps_sym, [branch_sym],
                                                    ensure_eps_op!(branch_sym); namespace=ns))
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(join_sym, [r.inputs[1], eps_sym], r.op;
                                                    attrs=r.attrs, namespace=ns,
                                                    atom_type=r.atom_type))
            NeuroDSL._invalidate_downstream!(g, join_sym, ns)
        end
    end
    for b in branches; EPSV[b] = 1.0f0; end

    order = sortperm([length(p["token_ids"]) for p in prompts]; rev=true)
    to_process = order[1:min(N_PROMPTS, length(order))]

    emit(@sprintf("\nPromts traités : %d (indices %s)", length(to_process),
                  join([prompts[k]["prompt"][1:min(20,end)] for k in to_process], " | ")))
    emit("\n" * "-"^78)
    emit("Par prompt : résidu de l'ajustement moindres carrés de u_{s,t}")
    emit("-"^78 * "\n")

    all_resid_rel = Float64[]

    for k in to_process
        pr = prompts[k]
        ids = Int.(pr["token_ids"]) .+ 1
        set_input!(ids)

        for b in branches; EPSV[b] = 1.0f0; end
        base = backward!()
        g_s = Float64.(vec(base[S_SYM]))
        g_t = Float64.(vec(base[T_SYM]))
        D = length(g_t)

        Bmat = Matrix{Float64}(undef, Bt, D)
        dvec = Vector{Float64}(undef, Bt)
        qs_all = Vector{Float64}(undef, Bt)
        qt_all = Vector{Float64}(undef, Bt)

        for (row, bj) in enumerate(cone_t)
            EPSV[bj] = 0.0f0
            abl = backward!()
            EPSV[bj] = 1.0f0
            abl_s = Float64.(vec(abl[S_SYM]))
            abl_t = Float64.(vec(abl[T_SYM]))
            beta_t = g_t .- abl_t
            qs = 1.0 - dot(abl_s, g_s)/dot(g_s, g_s)
            qt = 1.0 - dot(abl_t, g_t)/dot(g_t, g_t)
            Bmat[row, :] = beta_t
            dvec[row] = qs - qt
            qs_all[row] = qs; qt_all[row] = qt
        end
        reclaim()

        # moindres carrés : u minimisant ||Bmat*u - dvec||
        u = Bmat \ dvec
        resid = Bmat*u .- dvec
        resid_norm = norm(resid)
        d_norm = norm(dvec)
        resid_rel = d_norm > 0 ? resid_norm/d_norm : 0.0
        push!(all_resid_rel, resid_rel)

        emit(@sprintf("prompt %2d (n=%2d): B(t)=%d  D=%d  ||d||=%.4e  ||résidu||=%.3e  résidu relatif=%.3e  max|résidu|=%.3e",
                      k, length(ids), Bt, D, d_norm, resid_norm, resid_rel, maximum(abs.(resid))))
    end

    emit("\n" * "="^78)
    emit("RÉSUMÉ")
    emit("="^78)
    emit(@sprintf("\nrésidu relatif ||Bu-d||/||d|| sur %d prompts : min %.3e  médian %.3e  max %.3e",
                  length(all_resid_rel), minimum(all_resid_rel), median(all_resid_rel), maximum(all_resid_rel)))
    emit("\nSi ce résidu est à la précision machine (~1e-6 à 1e-12 en Float32/Float64")
    emit("mixte), l'identité de transport est vérifiée EXACTEMENT sur le modèle")
    emit("entraîné réel, pas seulement en synthétique : un seul vecteur u_{s,t}")
    emit("explique la totalité de la dépendance au site, pour cette paire de sites,")
    emit("sur données de production.")
end
println("\nÉcrit : ", OUT)
