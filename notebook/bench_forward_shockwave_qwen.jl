# ══════════════════════════════════════════════════════════════════════════════
# UNE TRACE UNIQUE : PROPAGATION D'UNE ABLATION FORWARD ET SON ANTI-ONDE --
# machinerie de patching déjà publiée (src/patching.jl), PAS branch_order_theorem
#
# CE QUE C'EST, ET CE QUE C'EST DÉLIBÉRÉMENT PAS
# -----------------------------------------------
# Demande : « au lieu de laisser le choc Δ(x) se propager en aval où il
# déclenche l'effet Hydra, calculer et injecter une anti-onde immédiate pour
# l'annihiler à la source ». Verdict pris AVANT ce script (voir le rapport) :
# l'effet Hydra (McGrath et al., citation non vérifiée avec la précision
# exigée par ce projet -- voir le rapport, pas de bibliographie inventée) est
# un phénomène FORWARD/fonctionnel : ablater une valeur CALCULÉE change ce que
# d'AUTRES composants CALCULENT en aval, et le résultat final récupère souvent
# plus que la corruption intermédiaire ne le suggère.
#
# branch_order_theorem.tex décompose exactement un lecteur ARRIÈRE (gradient)
# sous des portes IDENTITÉ EN FORWARD par construction (Définition 1) -- q_j,
# rho_j, c_j ne peuvent PAS voir une perturbation forward, par construction,
# pas par lacune de mesure. Ce script n'y touche donc pas et n'édite pas
# branch_order_theorem.tex. Il réutilise UNIQUEMENT la machinerie de patching
# causal déjà publiée et testée (capture_activations, patch_node!,
# recovery_metric de src/patching.jl) -- rien de neuf n'est inventé pour
# l'opération centrale.
#
# CONCEPTION, PRÉ-ENREGISTRÉE
# ----------------------------
# UN prompt, UN site ablaté, UNE trace couche par couche -- pas une moyenne.
#   - Site ablaté   : layer_14_mha_output_out (branche attention, milieu de
#     pile, L=28) -- mis à ZÉRO dans le graphe FORWARD (composant supprimé,
#     ablation causale standard), pas de porte eps, rien à voir avec q_j.
#   - Choc mesuré   : Delta(i) := ||x_i^ablaté - x_i^clean|| / ||x_i^clean||
#     à la sortie de CHAQUE couche i = 14..28, plus la perte finale.
#   - Signature Hydra recherchée : Delta(i) qui PLAFONNE ou DÉCROÎT avant la
#     sortie (la corruption relative au flux résiduel se résorbe), plutôt que
#     de croître sans borne -- observation directe, pas une réplication de la
#     méthode de décomposition en effets directs de l'article original.
#   - Anti-onde     : au site layer_18_out (4 couches en aval de l'ablation),
#     PATCH_NODE! restaure la valeur CLAIRE (capture_activations du run
#     propre). Prédiction : Delta(i) = 0 EXACTEMENT (à la précision flottante)
#     pour tout i >= 18, par construction du graphe déterministe -- ceci est
#     vérifié, pas supposé.
#   - Pont q_j (UN point, pas une corrélation) : q_j exact au site ablaté,
#     même prompt, machinerie de bench_eps_exact_ablation_qwen.jl réutilisée
#     telle quelle. Rapporté comme un point isolé, explicitement non
#     généralisable, pour un coût quasi nul étant donné que la trace existe
#     déjà.
#
# USAGE : julia --project=. notebook/bench_forward_shockwave_qwen.jl
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, Random, Printf, JSON, LinearAlgebra, Statistics

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const CKPT      = joinpath(MODEL_DIR, "qwen2_neurodsl")
const OUT       = joinpath(@__DIR__, "bench_forward_shockwave_qwen_results.txt")

reclaim() = (GC.gc(); NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.reclaim())

open(OUT, "w") do io
    emit(s) = (println(io, s); println(s); flush(io))
    emit("TRACE UNIQUE : ABLATION FORWARD ET ANTI-ONDE -- Qwen2.5-1.5B-Instruct")
    emit("Machinerie de src/patching.jl (capture_activations, patch_node!,")
    emit("recovery_metric), déjà publiée. AUCUN lien avec branch_order_theorem.tex :")
    emit("les portes eps y sont identité en forward par construction, donc ne")
    emit("peuvent structurellement pas voir ce que cette trace mesure.")
    emit("Date : " * strip(read(`date -u "+%Y-%m-%dT%H:%M:%SZ"`, String)))

    dev = NeuroDSL.Backend.CUDADevice()
    ns, L = :qwen2, 28
    logits, loss = :lm_head_out, :ce_loss

    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    emit("\nChargement du checkpoint natif...")
    NeuroDSL.load_graph!(g, ns, CKPT)
    reclaim()
    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss, [logits, :labels], :cross_entropy;
                                            namespace=ns))

    prompt = "The capital of the country where the Eiffel Tower stands is"
    prompts = JSON.parsefile(joinpath(@__DIR__, "qwen_sweep_prompts.json"))
    entry = prompts[findfirst(p -> p["prompt"] == prompt, prompts)]
    ids = Int.(entry["token_ids"]) .+ 1
    emit(@sprintf("\nPrompt : %s  (%d tokens)", repr(prompt), length(ids)))

    function set_input!(idv::Vector{Int})
        n = length(idv)
        NeuroDSL.set!(g, :token_ids, idv; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :pos_ids, collect(1:n); atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, vcat(idv[2:end], idv[end]);
                      atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
    end

    # ─── run propre : capture toutes les activations ────────────────────────
    set_input!(ids)
    NeuroDSL.demand!(g, logits; namespace=ns)
    clean_cache = NeuroDSL.capture_activations(g, ns)
    clean_loss  = Float64(sum(Array(NeuroDSL.demand!(g, loss; namespace=ns))))
    emit(@sprintf("\nRun propre : perte = %.6f", clean_loss))

    layer_out(i) = Symbol("layer_", i, "_out")
    ABL_SITE = :layer_14_mha_output_out
    ANTI_SITE = layer_out(18)

    # ─── porte de correction : le run propre re-demandé doit être bit-identique ──
    NeuroDSL.invalidate_all!(g; namespace=ns)
    v2 = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))
    v1 = copy(Array(clean_cache[logits]))
    gate_ok = (v1 == v2)
    emit(@sprintf("Porte : run propre redemandé bit-identique au premier : %s",
                  gate_ok ? "oui" : "ÉCHEC"))
    gate_ok || (emit("✗ ARRÊT -- déterminisme rompu."); return)

    # ─── ablation forward : le composant est mis à ZÉRO, pas gaté en eps ────
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, ABL_SITE; namespace=ns)
    abl_shape = size(NeuroDSL.node(g, ABL_SITE; namespace=ns).value)
    zero_cache = Dict{Symbol,Any}(ABL_SITE => NeuroDSL.Backend.zeros32(dev, abl_shape...))
    NeuroDSL.patch_node!(g, ABL_SITE, zero_cache; namespace=ns)
    NeuroDSL.demand!(g, logits; namespace=ns)
    ablated_cache = NeuroDSL.capture_activations(g, ns)
    ablated_loss  = Float64(sum(Array(NeuroDSL.demand!(g, loss; namespace=ns))))

    emit("\n" * "-"^78)
    emit(@sprintf("CHOC : site ablaté = %s (couche 14, branche attention -> zéro)", ABL_SITE))
    emit("-"^78)
    emit(@sprintf("\n%7s %12s %12s %14s", "couche", "||Delta||", "||clean||", "Delta relatif"))
    dnorm(i) = begin
        c = Array(clean_cache[layer_out(i)]); a = Array(ablated_cache[layer_out(i)])
        (norm(a .- c), norm(c))
    end
    for i in 14:L
        dn, cn = dnorm(i)
        emit(@sprintf("%7d %12.4f %12.4f %14.6f", i, dn, cn, dn/cn))
    end
    emit(@sprintf("\nPerte : propre %.6f -> ablatée %.6f  (Δperte = %+.6f)",
                  clean_loss, ablated_loss, ablated_loss - clean_loss))

    # ─── anti-onde : restauration exacte au site ANTI_SITE ──────────────────
    emit("\n" * "-"^78)
    emit(@sprintf("ANTI-ONDE : patch_node! restaure %s à sa valeur PROPRE", ANTI_SITE))
    emit("-"^78)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    NeuroDSL.demand!(g, ABL_SITE; namespace=ns)
    NeuroDSL.patch_node!(g, ABL_SITE, zero_cache; namespace=ns)
    NeuroDSL.demand!(g, ANTI_SITE; namespace=ns)
    NeuroDSL.patch_node!(g, ANTI_SITE, clean_cache; namespace=ns)
    NeuroDSL.demand!(g, logits; namespace=ns)
    healed_cache = NeuroDSL.capture_activations(g, ns)
    healed_loss  = Float64(sum(Array(NeuroDSL.demand!(g, loss; namespace=ns))))
    rm = NeuroDSL.recovery_metric(Array(NeuroDSL.demand!(g, logits; namespace=ns)),
                                   Array(clean_cache[logits]), Array(ablated_cache[logits]))

    emit(@sprintf("\n%7s %12s %12s %14s", "couche", "||Delta||", "||clean||", "Delta relatif"))
    for i in 14:L
        c = Array(clean_cache[layer_out(i)]); h = Array(healed_cache[layer_out(i)])
        dn = norm(h .- c); cn = norm(c)
        emit(@sprintf("%7d %12.4e %12.4f %14.4e", i, dn, cn, dn/cn))
    end
    emit(@sprintf("\nPerte : propre %.6f  ablatée %.6f  anti-ondée %.6f  (Δperte = %+.2e)",
                  clean_loss, ablated_loss, healed_loss, healed_loss - clean_loss))
    emit(@sprintf("recovery_metric (logits) = %.6f  (1.0 = récupération totale)", rm))

    emit("\n" * "-"^78)
    emit("LECTURE DE LA SIGNATURE HYDRA (une seule trace, pas une réplication)")
    emit("-"^78)
    ds = [dnorm(i)[1]/dnorm(i)[2] for i in 14:L]
    grows = all(ds[k+1] >= ds[k] - 1e-6 for k in 1:length(ds)-1)
    emit(@sprintf("\n  Delta relatif : couche 14 = %.4f -> couche 28 = %.4f", ds[1], ds[end]))
    emit(grows ?
        "  Croissance monotone (ou quasi) -- pas de signature de compensation" *
        " visible sur cette trace : la corruption s'accumule plutôt que de se résorber." :
        "  NON monotone -- le Delta relatif plafonne ou décroît à au moins un point" *
        " avant la sortie : signature compatible avec une auto-compensation.")

    # ─── pont, UN point, explicitement non généralisable ────────────────────
    emit("\n" * "="^78)
    emit("PONT VERS q_j -- UN SEUL POINT, AUCUNE CORRÉLATION POSSIBLE À n=1")
    emit("="^78)
    emit("\nCalcul de q_j exact au site ablaté, même prompt, machinerie de")
    emit("bench_eps_exact_ablation_qwen.jl réutilisée telle quelle (portes eps,")
    emit("backward uniquement -- indépendant de tout ce qui précède).")

    branches = Symbol[]
    for i in 1:L
        push!(branches, Symbol("layer_", i, "_mha_output_out"))
        push!(branches, Symbol("layer_", i, "_mlp_out"))
    end
    pos = Dict(b => k for (k, b) in enumerate(branches))
    B_of(b) = length(branches) - pos[b] + 1

    NeuroDSL.invalidate_all!(g; namespace=ns)
    set_input!(ids)
    EPSV = Dict{Symbol,Float32}(); CAPTURED = Dict{Symbol,Array{Float32}}()
    function ensure_eps_op!(branch::Symbol)
        op = Symbol("epsop_", branch)
        EPSV[branch] = 1.0f0
        haskey(NeuroDSL.CUSTOM_OPS, op) && return op
        NeuroDSL.register_op!(op,
            (dev, out_buf, inputs, attrs, out_sym, out_node, ctx) -> (out_buf .= inputs[1]))
        NeuroDSL.CUSTOM_SHAPE_RULES[op] = (inputs, attrs) -> size(inputs[1])
        NeuroDSL.GRAD_RULES[op] = (dev, dy, ctx, inputs) -> begin
            gr = EPSV[branch] .* dy
            CAPTURED[branch] = copy(Array(gr))
            return (gr,)
        end
        return op
    end
    for i in 1:L
        for (join_sym, bsym) in ((Symbol("layer_", i, "_res1"), Symbol("layer_", i, "_mha_output_out")),
                                  (Symbol("layer_", i, "_out"),  Symbol("layer_", i, "_mlp_out")))
            r = g.rules[ns][join_sym]
            r.inputs[2] == bsym || error("$join_sym : branche inattendue")
            es = Symbol("eps_", bsym)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(es, [bsym], ensure_eps_op!(bsym); namespace=ns))
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(join_sym, [r.inputs[1], es], r.op;
                                                    attrs=r.attrs, namespace=ns, atom_type=r.atom_type))
            NeuroDSL._invalidate_downstream!(g, join_sym, ns)
        end
    end
    backward!() = (empty!(CAPTURED); NeuroDSL.demand!(g, loss; namespace=ns);
                   NeuroDSL.backward_graph!(g, loss; namespace=ns); copy(CAPTURED))
    for b in branches; EPSV[b] = 1.0f0; end
    base = backward!()
    EPSV[ABL_SITE] = 0.0f0
    abl_bwd = backward!()
    EPSV[ABL_SITE] = 1.0f0
    # q_j DOIT être lu à un site EN AMONT de ABL_SITE (le site standard de tout
    # le reste de l'article : la couche 1), PAS au site ablaté lui-même : lire
    # CAPTURED[ABL_SITE] avant/après avoir fermé la porte de ABL_SITE donne
    # trivialement 1 par la Corollaire du branchement obligatoire (ce piège a
    # été trouvé après une première exécution -- voir la correction dans le
    # fichier de résultats). READ_SITE doit être en amont de ABL_SITE dans
    # l'ordre des branches pour que ABL_SITE soit dans son cône.
    READ_SITE = :layer_1_mha_output_out
    g0 = abl_bwd[READ_SITE]; gg = base[READ_SITE]
    q_at_site = 1.0 - Float64(dot(vec(g0), vec(gg)))/Float64(dot(vec(gg), vec(gg)))

    emit(@sprintf("\n  q_j pour la branche %s, lue au site %s : %+.5f",
                  ABL_SITE, READ_SITE, q_at_site))
    emit(@sprintf("  Delta relatif à la sortie (couche 28), sans anti-onde : %.4f", ds[end]))
    emit(@sprintf("  recovery_metric avec anti-onde                        : %.4f", rm))
    emit("\n  Un point. Ni signe, ni magnitude relative ne peuvent être généralisés.")
    emit("  Rapporté parce que le coût marginal était nul, pas comme un résultat.")

    reclaim()
end
println("\nÉcrit : ", OUT)
