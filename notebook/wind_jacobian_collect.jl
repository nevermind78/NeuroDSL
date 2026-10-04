# ══════════════════════════════════════════════════════════════════════════════
# WIND-1 -- COLLECTE DES JACOBIENNES PAR COUCHE (dernier token), 3 variantes
# Pré-enregistrement : notebook/wind_preregistration.md (écrit AVANT ce script).
#
# J_k = ∂x_{k+1,n}/∂x_{k,n} (bloc diagonal du dernier token), 1536×1536, colonne par
# colonne par différence finie centrée sur la base canonique, en ne recalculant que la
# couche k+1 (cône réactif de NeuroDSL). Variantes :
#   T  : vraie Jacobienne
#   A  : les 12 pr_h (probabilités d'attention) de la couche épinglés à leur valeur propre
#   AN : A + dénominateurs RMSNorm gelés (op custom :rmsnorm_frozen, échelle propre)
# Portes G1-G4 ici (G5 = wind_adjoint_gate.jl). Matrices -> WIND_OUTDIR (Float32 brut).
#
# USAGE : WIND_OUTDIR=<dossier> julia --project=. notebook/wind_jacobian_collect.jl
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, LinearAlgebra, Statistics

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const OUTDIR    = get(ENV, "WIND_OUTDIR", joinpath(@__DIR__, "wind_data"))
const RES       = get(ENV, "WIND_RES", joinpath(@__DIR__, "wind_jacobian_collect_results.txt"))
const PROMPT_IDXS = parse.(Int, split(get(ENV, "WIND_PROMPTS", "5,6,12,17,22"), ","))
const VARIANTS  = Symbol.(split(get(ENV, "WIND_VARIANTS", "T,A,AN"), ","))
const EPS_REL   = 5e-3
const EPS_REL0  = 1e-2      # k = 0 (embedding, norme ~0.8)
const NCOL      = parse(Int, get(ENV, "WIND_NCOL", "1536"))   # < 1536 : test de fumée uniquement

mkpath(OUTDIR)
reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())

# ─── op RMSNorm à dénominateur gelé ───────────────────────────────────────────
const FROZEN_RMS = Dict{Symbol,Any}()
NeuroDSL.register_op!(:rmsnorm_frozen,
    (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> begin
        x, gamma = inputs[1], inputs[2]
        out_buf .= x .* FROZEN_RMS[out_sym] .* reshape(gamma, 1, :)
        out_buf
    end)
NeuroDSL.CUSTOM_SHAPE_RULES[:rmsnorm_frozen] = (inputs, attrs) -> size(inputs[1])

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-1 -- COLLECTE DES JACOBIENNES PAR COUCHE (dernier token)")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))
    emit("Prompts : $(PROMPT_IDXS)   variantes : $(VARIANTS)   sortie : $(OUTDIR)   colonnes : $(NCOL)")
    NCOL < 1536 && emit("*** TEST DE FUMÉE : Jacobiennes incomplètes, résultats NON utilisables ***")

    dev = NeuroDSL.Backend.CUDADevice()
    ns, L, D, NH = :qwen2, 28, 1536, 12
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    NeuroDSL.load_graph!(g, ns, CKPT)
    reclaim()

    res_sym(k) = k == 0 ? :tok_out : Symbol("layer_", k, "_out")
    mha_sym(l) = Symbol("layer_", l, "_mha_output_out")
    mlp_sym(l) = Symbol("layer_", l, "_mlp_out")
    pr_sym(l, h) = Symbol("layer_", l, "_mha_pr_h", h)
    norm_syms = vcat([Symbol("layer_", l, "_norm", j, "_out") for l in 1:L for j in 1:2], [:final_norm_out])
    orig_norm_rules = Dict(s => g.rules[ns][s] for s in norm_syms)
    all(r.op == :rmsnorm for r in values(orig_norm_rules)) || error("règle de norme inattendue")

    function swap_norms!(frozen::Bool)
        for s in norm_syms
            r = orig_norm_rules[s]
            newr = frozen ? NeuroDSL.GraphRule(s, r.inputs, :rmsnorm_frozen; attrs=r.attrs,
                                               namespace=ns, atom_type=r.atom_type) : r
            NeuroDSL.addrule!(g, newr)
        end
    end

    W_U = NeuroDSL.node(g, :lm_head_W; namespace=ns).value
    prompts = JSON.parsefile(joinpath(@__DIR__, "qwen_sweep_prompts.json"))

    for pidx in PROMPT_IDXS
        ids = Int.(prompts[pidx]["token_ids"]) .+ 1
        n = length(ids)
        emit("\n" * "="^90)
        emit(@sprintf("PROMPT %d : %s (%d tokens)", pidx, repr(prompts[pidx]["prompt"]), n))
        emit("="^90)
        swap_norms!(false)
        NeuroDSL.set!(g, :token_ids, ids; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.demand!(g, :lm_head_out; namespace=ns)
        clean = NeuroDSL.capture_activations(g, ns)
        for s in norm_syms
            FROZEN_RMS[s] = copy(NeuroDSL.node(g, s; namespace=ns).aux_data[:rms_inv])
        end
        logits_n = Float64.(Array(clean[:lm_head_out])[n, :])
        order = sortperm(logits_n; rev=true)
        y1, y2 = order[1], order[2]
        w_read = Float64.(vec(Array(W_U[y1:y1, :]))) .- Float64.(vec(Array(W_U[y2:y2, :])))
        X = [Float64.(Array(clean[res_sym(k)])[n, :]) for k in 0:L]
        xnorm = norm.(X)
        emit(@sprintf("top-1 id=%d (logit %.3f)  top-2 id=%d (logit %.3f)", y1-1, logits_n[y1], y2-1, logits_n[y2]))

        meta = Dict("prompt_idx"=>pidx, "prompt"=>prompts[pidx]["prompt"], "n"=>n,
                    "top1_id"=>y1-1, "top2_id"=>y2-1, "logit_top1"=>logits_n[y1], "logit_top2"=>logits_n[y2],
                    "xnorm"=>xnorm, "X"=>X, "w_read"=>w_read,
                    "gamma_final"=>Float64.(Array(NeuroDSL.node(g, :final_norm_gamma; namespace=ns).value)),
                    "rms_inv_final"=>Float64(Array(FROZEN_RMS[:final_norm_out])[n]),
                    "eps_rel"=>EPS_REL, "eps_rel0"=>EPS_REL0, "gates"=>Dict{String,Any}())

        epsk(k) = (k == 0 ? EPS_REL0 : EPS_REL) * xnorm[k+1]

        # ─── une JVP directionnelle (vecteur quelconque) mono-couche -> (J v) ───────
        function pin!(l, V)
            V in (:A, :AN) || return
            for h in 1:NH; NeuroDSL.patch_node!(g, pr_sym(l, h), clean; namespace=ns); end
        end
        function layer_jvp_dir(k, v::Vector{Float64}, eps, V)
            src = res_sym(k); l = k + 1
            P = zeros(Float32, n, D); P[n, :] .= Float32.(eps .* v)
            Pd = NeuroDSL.Backend.to_device(dev, P)
            r = Vector{Vector{Float64}}(undef, 2)
            for (i, s) in enumerate((1f0, -1f0))
                NeuroDSL.patch_node!(g, src, Dict(src => clean[src] .+ s .* Pd); namespace=ns)
                pin!(l, V)
                a = Float64.(vec(Array(NeuroDSL.demand!(g, mha_sym(l); namespace=ns)[n:n, :])))
                m = Float64.(vec(Array(NeuroDSL.demand!(g, mlp_sym(l); namespace=ns)[n:n, :])))
                r[i] = a .+ m
            end
            return v .+ (r[1] .- r[2]) ./ (2eps)
        end
        # ─── Jacobienne complète mono-couche, colonnes canoniques, accumulée sur GPU ─
        function layer_jacobian(k, V)
            src = res_sym(k); l = k + 1
            eps = epsk(k)
            x0 = clean[src]
            x0n = Float32.(vec(Array(x0[n:n, :])))
            Jg = NeuroDSL.CUDA.zeros(Float32, D, D)
            veff = ones(Float64, D)      # colonnes non calculées (fumée seulement) : B=0, J=I
            xp = copy(x0); xm = copy(x0)
            for j in 1:NCOL
                copyto!(xp, x0); copyto!(xm, x0)
                view(xp, n:n, j:j) .+= Float32(eps)
                view(xm, n:n, j:j) .-= Float32(eps)
                veff[j] = (Float64(x0n[j] + Float32(eps)) - Float64(x0n[j] - Float32(eps))) / (2eps)
                NeuroDSL.patch_node!(g, src, Dict(src => xp); namespace=ns); pin!(l, V)
                ap = NeuroDSL.demand!(g, mha_sym(l); namespace=ns)[n:n, :]
                mp = NeuroDSL.demand!(g, mlp_sym(l); namespace=ns)[n:n, :]
                NeuroDSL.patch_node!(g, src, Dict(src => xm); namespace=ns); pin!(l, V)
                am = NeuroDSL.demand!(g, mha_sym(l); namespace=ns)[n:n, :]
                mm = NeuroDSL.demand!(g, mlp_sym(l); namespace=ns)[n:n, :]
                view(Jg, :, j) .= vec((ap .- am) .+ (mp .- mm)) ./ Float32(2eps)
            end
            NeuroDSL.patch_node!(g, src, clean; namespace=ns)
            B = Float64.(Array(Jg))
            B ./= reshape(veff, 1, D)        # corrige la taille de pas effective (arrondi fp32)
            return Matrix{Float64}(I, D, D) .+ B
        end
        # ─── JVP multi-couches directe (perturber x_s, lire x_t) ───────────────────
        function multi_jvp_dir(s, t, v, eps, V)
            src = res_sym(s)
            P = zeros(Float32, n, D); P[n, :] .= Float32.(eps .* v)
            Pd = NeuroDSL.Backend.to_device(dev, P)
            r = Vector{Vector{Float64}}(undef, 2)
            for (i, sg) in enumerate((1f0, -1f0))
                NeuroDSL.patch_node!(g, src, Dict(src => clean[src] .+ sg .* Pd); namespace=ns)
                if V in (:A, :AN)
                    for l in s+1:t, h in 1:NH
                        NeuroDSL.patch_node!(g, pr_sym(l, h), clean; namespace=ns)
                    end
                end
                r[i] = Float64.(vec(Array(NeuroDSL.demand!(g, res_sym(t); namespace=ns)[n:n, :])))
            end
            NeuroDSL.patch_node!(g, src, clean; namespace=ns)
            return (r[1] .- r[2]) ./ (2eps)
        end

        rng = MersenneTwister(1000 + pidx)
        # ─── G2 : Jacobienne MLP analytique vs DF (variante T, norme vraie) ─────────
        g2 = Float64[]
        for l in (2, 14, 27)
            r1s = Symbol("layer_", l, "_res1")
            r = Float64.(vec(Array(clean[r1s][n:n, :])))
            W1 = Float64.(Array(NeuroDSL.node(g, Symbol("layer_", l, "_mlp_w1"); namespace=ns).value))
            W2 = Float64.(Array(NeuroDSL.node(g, Symbol("layer_", l, "_mlp_w2"); namespace=ns).value))
            W3 = Float64.(Array(NeuroDSL.node(g, Symbol("layer_", l, "_mlp_w3"); namespace=ns).value))
            γ  = Float64.(Array(NeuroDSL.node(g, Symbol("layer_", l, "_norm2_gamma"); namespace=ns).value))
            ρ = 1 / sqrt(mean(r .^ 2) + 1e-6); z = ρ .* γ .* r
            gz = W1 * z; uz = W3 * z; σ = 1 ./ (1 .+ exp.(-gz)); silu = gz .* σ
            dsilu = σ .* (1 .+ gz .* (1 .- σ))
            for _ in 1:2
                v = randn(rng, D); v ./= norm(v)
                dz = ρ .* γ .* (v .- (ρ^2 / D) .* r .* dot(r, v))
                jv_an = W2 * ((dsilu .* uz) .* (W1 * dz) .+ silu .* (W3 * dz))
                eps = EPS_REL * norm(r)
                P = zeros(Float32, n, D); P[n, :] .= Float32.(eps .* v)
                Pd = NeuroDSL.Backend.to_device(dev, P)
                o = Vector{Vector{Float64}}(undef, 2)
                for (i, sg) in enumerate((1f0, -1f0))
                    NeuroDSL.patch_node!(g, r1s, Dict(r1s => clean[r1s] .+ sg .* Pd); namespace=ns)
                    o[i] = Float64.(vec(Array(NeuroDSL.demand!(g, mlp_sym(l); namespace=ns)[n:n, :])))
                end
                push!(g2, norm(jv_an .- (o[1] .- o[2]) ./ (2eps)) / norm(jv_an))
            end
            NeuroDSL.patch_node!(g, r1s, clean; namespace=ns)
        end
        emit(@sprintf("G2 MLP analytique vs DF : max err rel = %.2e  (seuil 1e-3) -> %s",
                      maximum(g2), maximum(g2) < 1e-3 ? "OK" : "ÉCHEC"))
        meta["gates"]["G2_max"] = maximum(g2)

        for V in VARIANTS
            swap_norms!(V == :AN)
            if V == :AN
                # G4b : forward propre avec normes gelées vs propre
                NeuroDSL.patch_node!(g, :tok_out, clean; namespace=ns)
                lgf = Float64.(Array(NeuroDSL.demand!(g, :lm_head_out; namespace=ns))[n, :])
                g4b = norm(lgf .- logits_n) / norm(logits_n)
                emit(@sprintf("G4b forward normes gelées vs propre : err rel logits = %.2e (seuil 1e-5) -> %s",
                              g4b, g4b < 1e-5 ? "OK" : "ÉCHEC"))
                meta["gates"]["G4b"] = g4b
            end
            if V in (:A, :AN)
                # G4a : épinglage sans perturbation, couche 14 et 27 bit-identiques
                ok = true
                for l in (14, 27)
                    NeuroDSL.patch_node!(g, res_sym(l-1), clean; namespace=ns)
                    pin!(l, V)
                    a = Array(NeuroDSL.demand!(g, mha_sym(l); namespace=ns))
                    if V == :A
                        ok &= (a == Array(clean[mha_sym(l)]))
                    else
                        ok &= (norm(Float64.(a) .- Float64.(Array(clean[mha_sym(l)]))) /
                               norm(Float64.(Array(clean[mha_sym(l)]))) < 1e-5)
                    end
                end
                emit("G4a épinglage attention sans perturbation (couches 14, 27) : " * (ok ? "OK" : "ÉCHEC"))
                meta["gates"]["G4a_$(V)"] = ok
            end

            t0 = time()
            Js = Array{Float32}(undef, D, D, L)
            g1 = Float64[]; g1b = Float64[]
            for k in 0:L-1
                J = layer_jacobian(k, V)
                Js[:, :, k+1] .= Float32.(J)
                # G1 : 4 directions aléatoires -- eps vs 2eps, et colonnes vs JVP directionnelle
                if V == :T || k % 7 == 0
                    for _ in 1:4
                        v = randn(rng, D); v ./= norm(v)
                        j1 = layer_jvp_dir(k, v, epsk(k), V)
                        j2 = layer_jvp_dir(k, v, 2epsk(k), V)
                        push!(g1, norm(j1 .- j2) / norm(j1))
                        push!(g1b, norm(J * v .- j1) / norm(j1))
                    end
                    NeuroDSL.patch_node!(g, res_sym(k), clean; namespace=ns)
                end
                (k % 7 == 6) && (reclaim(); emit(@sprintf("  [%s] couches 0..%d faites (%.0f s)", V, k, time()-t0)))
            end
            emit(@sprintf("[%s] collecte : %.0f s", V, time() - t0))
            emit(@sprintf("G1 [%s] eps vs 2eps : médiane %.2e  max %.2e  (seuils 1e-3 / 1e-2) -> %s", V,
                          median(g1), maximum(g1), (median(g1) < 1e-3 && maximum(g1) < 1e-2) ? "OK" : "ÉCHEC"))
            emit(@sprintf("G1b [%s] J(colonnes)·v vs JVP directionnelle : médiane %.2e  max %.2e", V,
                          median(g1b), maximum(g1b)))
            meta["gates"]["G1_$(V)_med"] = median(g1); meta["gates"]["G1_$(V)_max"] = maximum(g1)
            meta["gates"]["G1b_$(V)_med"] = median(g1b); meta["gates"]["G1b_$(V)_max"] = maximum(g1b)

            # ─── G3 : composition exacte, JVP multi-couches directe vs produit ──────
            pairs = V == :T ? [(1,8), (1,28), (7,14), (7,28), (14,21), (14,28), (21,28)] : [(7,21)]
            g3 = Float64[]
            for (s, t) in pairs
                v = randn(rng, D); v ./= norm(v)
                pv = copy(v)
                for k in s:t-1; pv = Float64.(Js[:, :, k+1]) * pv; end
                best = Inf; line = @sprintf("  G3 [%s] s=%2d t=%2d ||Jv||=%10.4f", V, s, t, norm(pv))
                for er in (5e-3, 1e-2)
                    d = multi_jvp_dir(s, t, v, er * xnorm[s+1], V)
                    e = norm(d .- pv) / norm(pv); best = min(best, e)
                    line *= @sprintf("  eps=%.0e: err=%.2e", er, e)
                end
                push!(g3, best); emit(line)
            end
            emit(@sprintf("G3 [%s] médiane err = %.2e (seuil 1e-2) -> %s", V, median(g3),
                          median(g3) < 1e-2 ? "OK" : "ÉCHEC"))
            meta["gates"]["G3_$(V)"] = g3

            fn = joinpath(OUTDIR, "wind_J_p$(pidx)_$(V).bin")
            open(fn, "w") do f; write(f, Js); end
            Js = nothing; reclaim()
        end
        swap_norms!(false)
        open(joinpath(OUTDIR, "wind_meta_p$(pidx).json"), "w") do f; JSON.print(f, meta); end
        emit("Écrit : wind_J_p$(pidx)_{$(join(VARIANTS, ","))}.bin + wind_meta_p$(pidx).json")
    end
end
println("\nÉcrit : ", RES)
