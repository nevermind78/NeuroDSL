# ══════════════════════════════════════════════════════════════════════════════
# WIND -- SONDE DE CALIBRATION (AVANT pré-enregistrement)
#
# Ne mesure AUCUNE quantité visée par les hypothèses (pas de spectre, pas de
# produit de normes, pas de rang effectif). Sert uniquement à fixer :
#   (a) les temps (chargement, forward complet, un pas de JVP mono-couche) ;
#   (b) les normes du flux résiduel au DERNIER token, couche par couche
#       (pour choisir eps relatif) + normes de la ligne BOS (activations massives) ;
#   (c) la convergence en eps de la JVP par différence centrée, sur quelques
#       directions aléatoires, à quelques couches ;
#   (d) porte indépendante : Jacobienne de la branche MLP ANALYTIQUE (CPU,
#       Float64, à partir des poids) vs différence finie (moteur GPU Float32) ;
#   (e) porte de gel d'attention : épingler les pr_h aux valeurs propres doit
#       reproduire la couche propre bit-à-bit ;
#   (f) porte de restauration : après patchs, logits propres bit-identiques.
#
# USAGE : julia --project=. notebook/wind_calib_probe.jl
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, LinearAlgebra, Statistics

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const OUT       = joinpath(@__DIR__, "wind_calib_probe_results.txt")
const PROMPT_IDX = 5   # "The capital of the country where the Eiffel Tower stands is"

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND -- SONDE DE CALIBRATION (aucune quantité d'hypothèse mesurée ici)")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))

    dev = NeuroDSL.Backend.CUDADevice()
    ns, L, D, NH = :qwen2, 28, 1536, 12
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    t0 = time()
    NeuroDSL.load_graph!(g, ns, CKPT)
    reclaim()
    emit(@sprintf("Chargement : %.1f s", time() - t0))

    prompts = JSON.parsefile(joinpath(@__DIR__, "qwen_sweep_prompts.json"))
    ids = Int.(prompts[PROMPT_IDX]["token_ids"]) .+ 1
    n = length(ids)
    emit(@sprintf("Prompt %d : %s (%d tokens)", PROMPT_IDX, repr(prompts[PROMPT_IDX]["prompt"]), n))
    NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)

    res_sym(k) = k == 0 ? :tok_out : Symbol("layer_", k, "_out")
    mha_sym(l) = Symbol("layer_", l, "_mha_output_out")
    mlp_sym(l) = Symbol("layer_", l, "_mlp_out")
    pr_sym(l, h) = Symbol("layer_", l, "_mha_pr_h", h)

    NeuroDSL.demand!(g, :lm_head_out; namespace=ns)   # échauffement JIT
    NeuroDSL.invalidate_all!(g; namespace=ns)
    t0 = time(); NeuroDSL.demand!(g, :lm_head_out; namespace=ns); NeuroDSL.CUDA.synchronize()
    emit(@sprintf("Forward complet (post-JIT) : %.1f ms", 1000*(time() - t0)))
    clean = NeuroDSL.capture_activations(g, ns)
    clean_logits = copy(Array(clean[:lm_head_out]))
    haskey(clean, :tok_out) || error("nœud :tok_out absent")

    emit("\nNormes du flux résiduel (sortie de couche k) :")
    emit(@sprintf("%4s %12s %12s %12s %12s", "k", "||x_k,n||", "max|x_k,n|", "||x_k,BOS||", "max|x_k,BOS|"))
    xnorm = zeros(L+1)
    for k in 0:L
        X = Float64.(Array(clean[res_sym(k)]))
        xnorm[k+1] = norm(X[n, :])
        emit(@sprintf("%4d %12.3f %12.3f %12.3f %12.3f", k, norm(X[n,:]), maximum(abs.(X[n,:])),
                      norm(X[1,:]), maximum(abs.(X[1,:]))))
    end

    # ─── JVP de la somme des branches de la couche k+1, dernier token ─────────
    function branch_jvp(k, v::Vector{Float64}, eps; frozen_attn::Bool=false)
        src = res_sym(k); l = k + 1
        x0 = clean[src]
        P = zeros(Float32, n, D); P[n, :] .= Float32.(eps .* v)
        Pd = NeuroDSL.Backend.to_device(dev, P)
        outs = Vector{Vector{Float64}}(undef, 2); ins = similar(outs)
        for (i, s) in enumerate((1, -1))
            xp = x0 .+ Float32(s) .* Pd
            NeuroDSL.patch_node!(g, src, Dict(src => xp); namespace=ns)
            if frozen_attn
                for h in 1:NH
                    NeuroDSL.patch_node!(g, pr_sym(l, h), clean; namespace=ns)
                end
            end
            a = Float64.(Array(NeuroDSL.demand!(g, mha_sym(l); namespace=ns))[n, :])
            m = Float64.(Array(NeuroDSL.demand!(g, mlp_sym(l); namespace=ns))[n, :])
            outs[i] = a .+ m
            ins[i] = Float64.(Array(xp)[n, :])
        end
        Bv = (outs[1] .- outs[2]) ./ (2eps)
        veff = (ins[1] .- ins[2]) ./ (2eps)
        return Bv, veff
    end
    restore!(k) = NeuroDSL.patch_node!(g, res_sym(k), clean; namespace=ns)

    # (a) temps d'un pas JVP
    rng = MersenneTwister(7)
    for k in (0, 13, 26)
        v = randn(rng, D); v ./= norm(v)
        branch_jvp(k, v, 1e-2 * xnorm[k+1])
        t0 = time()
        for _ in 1:5; branch_jvp(k, v, 1e-2 * xnorm[k+1]); end
        NeuroDSL.CUDA.synchronize()
        emit(@sprintf("Temps d'une JVP centrée mono-couche (k=%d -> %d) : %.2f ms", k, k+1, 1000*(time()-t0)/5))
        t0 = time()
        for _ in 1:5; branch_jvp(k, v, 1e-2 * xnorm[k+1]; frozen_attn=true); end
        NeuroDSL.CUDA.synchronize()
        emit(@sprintf("   idem, attention gelée (12 pr_h épinglés)     : %.2f ms", 1000*(time()-t0)/5))
        restore!(k)
    end

    # (c) convergence en eps (eps absolu = rel * ||x_k,n||)
    emit("\nConvergence en eps de la JVP centrée (3 directions aléatoires par couche)")
    emit("  écart relatif ||Bv(eps) - Bv(eps_ref)|| / ||Bv(eps_ref)||, eps_ref = 1e-3 relatif")
    rels = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    for k in (0, 6, 13, 20, 25, 26, 27)
        for trial in 1:3
            v = randn(rng, D); v ./= norm(v)
            ref, _ = branch_jvp(k, v, 1e-3 * xnorm[k+1])
            line = @sprintf("  k=%2d d%d ||Bv||=%10.4f :", k, trial, norm(ref))
            for r in rels
                Bv, _ = branch_jvp(k, v, r * xnorm[k+1])
                line *= @sprintf(" %.0e:%.1e", r, norm(Bv .- ref) / norm(ref))
            end
            emit(line)
        end
        restore!(k)
    end

    # (d) porte analytique : Jacobienne de la branche MLP de la couche l par rapport à res1
    emit("\nPorte analytique MLP (couche l, dernier token) : dM/dr via formule fermée vs DF")
    for l in (2, 14, 27)
        r1s = Symbol("layer_", l, "_res1")
        r = Float64.(Array(clean[r1s])[n, :])
        W1 = Float64.(Array(NeuroDSL.node(g, Symbol("layer_", l, "_mlp_w1"); namespace=ns).value))
        W2 = Float64.(Array(NeuroDSL.node(g, Symbol("layer_", l, "_mlp_w2"); namespace=ns).value))
        W3 = Float64.(Array(NeuroDSL.node(g, Symbol("layer_", l, "_mlp_w3"); namespace=ns).value))
        γ  = Float64.(Array(NeuroDSL.node(g, Symbol("layer_", l, "_norm2_gamma"); namespace=ns).value))
        ρ = 1 / sqrt(mean(r .^ 2) + 1e-6)
        z = ρ .* γ .* r
        gz = W1 * z; uz = W3 * z
        σ = 1 ./ (1 .+ exp.(-gz)); silu = gz .* σ; dsilu = σ .* (1 .+ gz .* (1 .- σ))
        # porte forward : la sortie MLP analytique doit égaler celle du moteur
        m_an = W2 * (silu .* uz)
        m_en = Float64.(Array(clean[mlp_sym(l)])[n, :])
        emit(@sprintf("  l=%2d forward MLP analytique vs moteur : rel=%.2e", l, norm(m_an .- m_en)/norm(m_en)))
        for trial in 1:3
            v = randn(rng, D); v ./= norm(v)
            dz = ρ .* γ .* (v .- (ρ^2 / D) .* r .* dot(r, v))
            jv_an = W2 * ((dsilu .* uz) .* (W1 * dz) .+ silu .* (W3 * dz))
            # DF sur le moteur : perturber res1 (ligne n), lire mlp_out
            eps = 1e-2 * norm(r)
            P = zeros(Float32, n, D); P[n, :] .= Float32.(eps .* v)
            Pd = NeuroDSL.Backend.to_device(dev, P)
            o = Vector{Vector{Float64}}(undef, 2)
            for (i, s) in enumerate((1, -1))
                NeuroDSL.patch_node!(g, r1s, Dict(r1s => clean[r1s] .+ Float32(s) .* Pd); namespace=ns)
                o[i] = Float64.(Array(NeuroDSL.demand!(g, mlp_sym(l); namespace=ns))[n, :])
            end
            jv_fd = (o[1] .- o[2]) ./ (2eps)
            emit(@sprintf("  l=%2d d%d ||J v|| an=%.5e  DF=%.5e  rel.err=%.2e", l, trial,
                          norm(jv_an), norm(jv_fd), norm(jv_an .- jv_fd)/norm(jv_an)))
        end
        NeuroDSL.patch_node!(g, r1s, clean; namespace=ns)
    end

    # (e) porte de gel : épingler pr_h propres sans perturbation = couche propre exacte
    emit("\nPorte de gel d'attention (sans perturbation) :")
    for k in (0, 13, 26)
        l = k + 1
        restore!(k)
        for h in 1:NH; NeuroDSL.patch_node!(g, pr_sym(l, h), clean; namespace=ns); end
        a = Array(NeuroDSL.demand!(g, mha_sym(l); namespace=ns))
        m = Array(NeuroDSL.demand!(g, mlp_sym(l); namespace=ns))
        emit(@sprintf("  couche %2d : mha bit-identique=%s  mlp bit-identique=%s", l,
                      a == Array(clean[mha_sym(l)]), m == Array(clean[mlp_sym(l)])))
        restore!(k)
    end

    # (f) restauration : logits propres bit-identiques
    for k in 0:L-1; restore!(k); end
    NeuroDSL.invalidate_all!(g; namespace=ns)
    lg = Array(NeuroDSL.demand!(g, :lm_head_out; namespace=ns))
    emit(@sprintf("\nPorte de restauration : logits propres bit-identiques = %s", lg == clean_logits))
    reclaim()
end
println("\nÉcrit : ", OUT)
