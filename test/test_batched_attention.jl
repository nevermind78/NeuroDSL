using Test, Random, LinearAlgebra, Statistics

# ══════════════════════════════════════════════════════════════════════════
# Attention multi-têtes batchée (gemm_strided_batched, vues zero-copy) --
# conçu avec Fable le 2026-07-10 (src/layers.jl, src/dispatch.jl,
# src/backward.jl, src/kernels.jl). Discipline : correction AVANT vitesse,
# comparaison ÉLÉMENT PAR ÉLÉMENT (pas isapprox par norme sur toute la
# matrice -- leçon apprise cette session sur un faux négatif).
#
# NOTE méthodologique : `Backend.rand32(::CUDADevice,...)` délègue à
# `CUDA.rand` (RNG côté GPU, jamais réinitialisé par `Random.seed!` -- vérifié
# dans src/backend.jl:16). Deux graphes construits séparément avec le même
# `Random.seed!` n'ont donc PAS forcément les mêmes poids sur CUDA -- toute
# comparaison "batched vs non-batched" copie explicitement les poids d'un
# graphe vers l'autre via `_copy_params!` plutôt que de compter sur la graine.
# ══════════════════════════════════════════════════════════════════════════

function _build_attn_graph(dev, ns::Symbol; batched::Bool, seq::Int=16, dim::Int=32, n_heads::Int=4)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    X = randn(Float32, seq, dim)
    NeuroDSL.set!(g, :input, X; namespace=ns)
    out = NeuroDSL.MultiHeadAttention(dim, n_heads; batched=batched)(g, :input, :mha; namespace=ns)
    return g, out
end

function _build_llama_graph(dev, ns::Symbol; batched_attn::Bool, seq::Int=16, dim::Int=32, n_heads::Int=4,
                             hidden_dim::Int=64, n_layers::Int=2)
    g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
    X = randn(Float32, seq, dim)
    NeuroDSL.set!(g, :input, X; namespace=ns)
    out = NeuroDSL.LlamaModel(n_layers, dim, n_heads, hidden_dim; batched_attn=batched_attn)(g, :input; namespace=ns)
    return g, out
end

# Copie tous les paramètres de src_g (par NOM de symbole, pas par ordre
# d'itération de Dict) vers dst_g -- rend deux graphes structurellement
# identiques (mêmes noms de paramètres, batched change seulement les nœuds
# intermédiaires) directement comparables, sans dépendre du RNG.
function _copy_params!(dst_g, dst_ns, src_g, src_ns)
    for (sym, nd) in src_g.nodes[src_ns]
        if nd.is_param
            NeuroDSL.set!(dst_g, sym, Array(nd.value); is_param=true, namespace=dst_ns)
        end
    end
    NeuroDSL.invalidate_all!(dst_g; namespace=dst_ns)
end

@testset "Attention multi-têtes batchée" begin
    devices = NeuroDSL.Backend.CUDA_AVAILABLE ?
        [NeuroDSL.Backend.CPUDevice(), NeuroDSL.Backend.CUDADevice()] :
        [NeuroDSL.Backend.CPUDevice()]

    for dev in devices
        devname = dev isa NeuroDSL.Backend.CPUDevice ? "CPU" : "CUDA"

        @testset "$devname -- équivalence forward, élément par élément" begin
            g_std, out_std   = _build_attn_graph(dev, Symbol(:std_fwd_, devname); batched=false)
            g_bat, out_bat   = _build_attn_graph(dev, Symbol(:bat_fwd_, devname); batched=true)
            _copy_params!(g_bat, g_bat.active_ns, g_std, g_std.active_ns)
            NeuroDSL.set!(g_bat, :input, Array(NeuroDSL.node(g_std, :input).value); namespace=g_bat.active_ns)
            NeuroDSL.invalidate_all!(g_bat)
            v_std = Array(NeuroDSL.demand!(g_std, out_std))
            v_bat = Array(NeuroDSL.demand!(g_bat, out_bat))
            @test size(v_std) == size(v_bat)
            @test maximum(abs.(v_std .- v_bat)) < 1f-4
        end

        @testset "$devname -- différences finies (batched=true)" begin
            g, out = _build_attn_graph(dev, Symbol(:bat_fd_, devname); batched=true)
            for psym in (:mha_q_W, :mha_k_W, :mha_v_W, :mha_output_W)
                ok, max_err = grad_check(g, psym, out; eps=Float32(1e-3), tol=Float32(2e-2))
                @test ok
            end
        end

        @testset "$devname -- gradients batched == non-batched, élément par élément" begin
            g_std, out_std = _build_attn_graph(dev, Symbol(:std_grad_, devname); batched=false)
            g_bat, out_bat = _build_attn_graph(dev, Symbol(:bat_grad_, devname); batched=true)
            _copy_params!(g_bat, g_bat.active_ns, g_std, g_std.active_ns)
            NeuroDSL.set!(g_bat, :input, Array(NeuroDSL.node(g_std, :input).value); namespace=g_bat.active_ns)
            NeuroDSL.invalidate_all!(g_bat)
            NeuroDSL.demand!(g_std, out_std); NeuroDSL.backward_graph!(g_std, out_std)
            NeuroDSL.demand!(g_bat, out_bat); NeuroDSL.backward_graph!(g_bat, out_bat)
            for psym in (:mha_q_W, :mha_k_W, :mha_v_W, :mha_output_W)
                gs = Array(NeuroDSL.node(g_std, psym).gradient)
                gb = Array(NeuroDSL.node(g_bat, psym).gradient)
                @test maximum(abs.(gs .- gb)) < 1f-4
            end
        end

        @testset "$devname -- backward_graph_sparse! ne coupe pas le flux de gradient" begin
            g, out = _build_attn_graph(dev, Symbol(:bat_sparse_, devname); batched=true)
            loss = Symbol(:bat_sparse_loss_, devname)
            NeuroDSL.addrule!(g, NeuroDSL.GraphRule(loss, [out], :sum_matrix; namespace=g.active_ns))
            NeuroDSL.demand!(g, loss)
            NeuroDSL.backward_graph!(g, loss; sparse=true)
            for psym in (:mha_q_W, :mha_k_W, :mha_v_W, :mha_output_W)
                gr = NeuroDSL.node(g, psym).gradient
                @test gr !== nothing
                @test all(isfinite, Array(gr))
            end
        end

        @testset "$devname -- patch_node! sur une vue (sc_h) n'écrit jamais dans le tenseur batché" begin
            g, out = _build_attn_graph(dev, Symbol(:bat_patch_, devname); batched=true)
            NeuroDSL.demand!(g, out)
            sc3_before = copy(Array(NeuroDSL.node(g, :mha_sc3).value))
            cache = NeuroDSL.capture_activations(g)
            patched_val = Array(cache[:mha_sc_h2]) .+ 5f0
            NeuroDSL.patch_node!(g, :mha_sc_h2, Dict(:mha_sc_h2 => patched_val))
            NeuroDSL.demand!(g, out)
            sc3_after = Array(NeuroDSL.node(g, :mha_sc3).value)
            @test sc3_before == sc3_after   # le patch n'a jamais touché le tenseur groupé
            @test Array(NeuroDSL.node(g, :mha_sc_h2).value) == patched_val
        end

        @testset "$devname -- capture_activations copie indépendamment les vues" begin
            g, out = _build_attn_graph(dev, Symbol(:bat_capture_, devname); batched=true)
            NeuroDSL.demand!(g, out)
            cache = NeuroDSL.capture_activations(g)
            sc_h1_cached = copy(Array(cache[:mha_sc_h1]))
            # Nouveau pas : la valeur de sc_h1 (et sc3) change
            NeuroDSL.set!(g, :input, randn(Float32, size(NeuroDSL.node(g,:input).value)...))
            NeuroDSL.invalidate_all!(g)
            NeuroDSL.demand!(g, out)
            @test Array(cache[:mha_sc_h1]) == sc_h1_cached   # le cache n'a pas bougé
        end

        @testset "$devname -- sweep_patch_sites! donne une recovery finie sur le graphe batché" begin
            ns_bat = Symbol(:bat_sweep_, devname)
            g_bat, out_bat = _build_attn_graph(dev, ns_bat; batched=true)

            Xc = randn(Float32, size(Array(NeuroDSL.node(g_bat,:input).value))...)
            NeuroDSL.set!(g_bat, :input, Xc); NeuroDSL.invalidate_all!(g_bat)
            clean_bat = copy(Array(NeuroDSL.demand!(g_bat, out_bat)))
            clean_cache_bat = NeuroDSL.capture_activations(g_bat)
            Xc2 = copy(Xc); Xc2[1,:] .= randn(Float32, size(Xc,2))
            NeuroDSL.set!(g_bat, :input, Xc2); NeuroDSL.invalidate_all!(g_bat)
            corrupt_bat = copy(Array(NeuroDSL.demand!(g_bat, out_bat)))
            corrupt_cache_bat = NeuroDSL.capture_activations(g_bat)

            r_bat = NeuroDSL.sweep_patch_sites!(g_bat, out_bat, [:mha_sc_h2], clean_cache_bat, corrupt_cache_bat, clean_bat, corrupt_bat)
            @test isfinite(r_bat[1].recovery)
        end

        @testset "$devname -- l'attention batchée n'arme jamais _POOLED_EXECUTION_SEEN" begin
            before = NeuroDSL._POOLED_EXECUTION_SEEN[]
            g, out = _build_attn_graph(dev, Symbol(:bat_pooled_, devname); batched=true)
            NeuroDSL.demand!(g, out)
            NeuroDSL.backward_graph!(g, out)
            @test NeuroDSL._POOLED_EXECUTION_SEEN[] == before
        end

        @testset "$devname -- LlamaModel(batched_attn=true) : forward+backward cohérents, non-régression" begin
            g_std, out_std = _build_llama_graph(dev, Symbol(:std_llama_, devname); batched_attn=false)
            g_bat, out_bat = _build_llama_graph(dev, Symbol(:bat_llama_, devname); batched_attn=true)
            _copy_params!(g_bat, g_bat.active_ns, g_std, g_std.active_ns)
            NeuroDSL.set!(g_bat, :input, Array(NeuroDSL.node(g_std, :input).value); namespace=g_bat.active_ns)
            NeuroDSL.invalidate_all!(g_bat)
            v_std = Array(NeuroDSL.demand!(g_std, out_std))
            v_bat = Array(NeuroDSL.demand!(g_bat, out_bat))
            @test maximum(abs.(v_std .- v_bat)) < 1f-3   # tolérance un peu plus large : 2 couches de propagation d'erreur BLAS
            NeuroDSL.backward_graph!(g_std, out_std)
            NeuroDSL.backward_graph!(g_bat, out_bat)
            for (sym, nd) in g_std.nodes[g_std.active_ns]
                nd.is_param || continue
                gs = Array(nd.gradient)
                gb = Array(NeuroDSL.node(g_bat, sym; namespace=g_bat.active_ns).gradient)
                @test maximum(abs.(gs .- gb)) < 1f-2
            end
        end
    end
end
