# ══════════════════════════════════════════════════════════════════════════════
# WIND-3 -- EXPLORATOIRE (post hoc) : normes des contributions d'arête ‖δ_{ℓ,i}‖ =
# ‖Σ_h p_h[n,i] v_{kv(h)}[i] W_O^{h}ᵀ‖ (analyse « par normes » de Kobayashi et al. 2020),
# pour tester directement le mécanisme « puits = valeur de petite norme » sur Qwen.
# Contrôle intégré : Σ_i δ_{ℓ,i} = sortie d'attention du dernier token (reconstruction).
# USAGE : julia --project=. notebook/wind3_edge_norms.jl
# ══════════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "wind3_common.jl"))
const RES = joinpath(@__DIR__, "wind3_edge_norms_results.txt")
const PROMPT_IDXS = [5, 6, 12, 17, 22]
const DH = 128

open(RES, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("WIND-3 -- EXPLORATOIRE : normes des contributions d'arête (dernier token ← i)")
    g, dev, orig, W_U = load_instrumented()
    prompts = JSON.parsefile(joinpath(@__DIR__, "qwen_sweep_prompts.json"))
    TOK = JSON.parsefile(joinpath(@__DIR__, "wind_data", "wind3_tokens.json"))
    out = Dict{String,Any}()
    for pidx in PROMPT_IDXS
        ids = Int.(prompts[pidx]["token_ids"]) .+ 1
        n = set_prompt!(g, ids)
        NeuroDSL.demand!(g, :lm_head_out; namespace=NS)
        c = capture_rule_nodes(g)
        E = zeros(L, n); vn = zeros(L, n); worst = 0.0
        for l in 1:L
            WO = Float64.(Array(NeuroDSL.node(g, Symbol("layer_", l, "_mha_output_W"); namespace=NS).value))
            V = [Float64.(Array(c[Symbol("layer_", l, "_mha_v_h", j)])) for j in 1:2]
            tot = zeros(D)
            for i in 1:n
                δ = zeros(D)
                for h in 1:NH
                    p = Float64(Array(c[pr_sym(l, h)])[n, i]); kv = (h - 1) ÷ 6 + 1
                    δ .+= p .* (WO[:, (h-1)*DH+1:h*DH] * V[kv][i, :])
                end
                E[l, i] = norm(δ); tot .+= δ
                vn[l, i] = norm(reduce(vcat, [WO[:, (h-1)*DH+1:h*DH] * V[(h-1)÷6+1][i, :] for h in 1:NH]))
            end
            ref = Float64.(vec(Array(c[mha_sym(l)][n:n, :])))
            worst = max(worst, norm(tot .- ref) / norm(ref))
        end
        toks = TOK[string(pidx)]["tokens"]
        emit(@sprintf("\nPROMPT %d : contrôle de reconstruction Σ_i δ = sortie MHA[n] : pire err rel = %.2e", pidx, worst))
        emit("  position : moyenne_ℓ ‖δ_{ℓ,i}‖ (contribution pondérée)  |  moyenne_ℓ ‖(v_i W_O^h)_h‖ (non pondérée)")
        for i in 1:n
            emit(@sprintf("   %2d %-10s %8.3f  |  %8.3f", i, repr(toks[i]), mean(E[:, i]), mean(vn[:, i])))
        end
        sh = sum(E[:, 1]) / sum(E[:, 1:n-1])
        emit(@sprintf("  part du puits dans Σ_{ℓ,i<n} ‖δ‖ : %.3f ; norme non pondérée du puits / médiane des autres : %.3f",
                      sh, mean(vn[:, 1]) / median(vec(mean(vn[:, 2:n-1]; dims=1)))))
        out[string(pidx)] = Dict("E" => [E[l, :] for l in 1:L], "vn" => [vn[l, :] for l in 1:L], "sink_share" => sh)
        c = nothing; reclaim()
    end
    open(joinpath(@__DIR__, "wind_data", "wind3_edge_norms.json"), "w") do f; JSON.print(f, out); end
end
println("\nÉcrit : ", RES)
