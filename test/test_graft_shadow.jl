# graft_shadow_block! (src/graph_surgery.jl) : greffe résiduelle gatée
# F(x) = x + alpha * R(x;theta), alpha scalaire APPRENABLE.
#
# Deux façons distinctes d'obtenir F(x)=x à l'initialisation, comparées ici :
#   - alpha0=0, zero_out_proj=false ("rezero")     : theta aléatoire, gate à zéro.
#   - alpha0=1, zero_out_proj=true  ("net2net")    : gate à un, sorties de R à zéro.
#   - alpha0=0, zero_out_proj=true  ("degenerate") : point de selle vrai (les deux à la fois).
#
# Correction avant tout script d'expérimentation, même discipline que test_surgery.jl :
# identité bit-exacte, falsifiabilité, puis propriétés ALGÉBRIQUES exactes des gradients
# (pas seulement "petit"), vérifiées par inspection directe + recalcul indépendant.

@testset "Greffe gatée (graft_shadow_block!)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    vocab_size, dim, n_heads, hidden_dim, n_layers, prefix_len = 20, 64, 4, 128, 4, 8

    function _build(ns, seed)
        Random.seed!(seed)
        g, logits = NeuroDSL.build_induction_graph(dev, ns; vocab_size=vocab_size, dim=dim, n_heads=n_heads,
                                                    hidden_dim=hidden_dim, n_layers=n_layers, prefix_len=prefix_len)
        tokens, labels = NeuroDSL.sample_induction_sequence(MersenneTwister(seed + 1), vocab_size, prefix_len)
        NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        return g, logits
    end

    @testset "Identité exacte : alpha0=0 (theta aléatoire, rezero)" begin
        ns = :shadow_id_rezero
        g, logits = _build(ns, 101)
        ref_output = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))
        NeuroDSL.graft_shadow_block!(g, ns, :layer_2_out, dim, n_heads, hidden_dim;
                                      alpha0=0f0, zero_out_proj=false)
        post_output = Array(NeuroDSL.demand!(g, logits; namespace=ns))
        @test ref_output == post_output   # bit-exact malgré theta ENTIÈREMENT aléatoire
    end

    @testset "Identité exacte : zero_out_proj=true (net2net, alpha=1)" begin
        ns = :shadow_id_net2net
        g, logits = _build(ns, 102)
        ref_output = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))
        NeuroDSL.graft_shadow_block!(g, ns, :layer_2_out, dim, n_heads, hidden_dim;
                                      alpha0=1f0, zero_out_proj=true)
        post_output = Array(NeuroDSL.demand!(g, logits; namespace=ns))
        @test ref_output == post_output   # bit-exact : R(x) ≡ 0 par construction, alpha0 sans effet
    end

    @testset "Falsifiabilité : perturbation non nulle casse l'égalité bit-exacte" begin
        ns = :shadow_falsif
        g, logits = _build(ns, 103)
        ref_output = copy(Array(NeuroDSL.demand!(g, logits; namespace=ns)))
        _, handle = NeuroDSL.graft_shadow_block!(g, ns, :layer_2_out, dim, n_heads, hidden_dim;
                                                  alpha0=0f0, zero_out_proj=false)
        NeuroDSL.set_params!(g, ns, Dict(handle.alpha_sym => Float32[0.37f0]))
        NeuroDSL.invalidate_all!(g; namespace=ns)
        pert_output = Array(NeuroDSL.demand!(g, logits; namespace=ns))
        @test pert_output != ref_output   # confirme que le test bit-exact ci-dessus n'est pas vide
    end

    @testset "Point de selle vrai (Prop. 3) : alpha0=0 ET zero_out_proj=true" begin
        ns = :shadow_degenerate
        g, logits = _build(ns, 104)
        _, handle = NeuroDSL.graft_shadow_block!(g, ns, :layer_2_out, dim, n_heads, hidden_dim;
                                                  alpha0=0f0, zero_out_proj=true)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.demand!(g, :loss; namespace=ns)
        NeuroDSL.backward_graph!(g, :loss; namespace=ns)

        alpha_grad = NeuroDSL.node(g, handle.alpha_sym; namespace=ns).gradient
        @test alpha_grad !== nothing
        @test all(iszero, alpha_grad)   # ∂L/∂alpha EXACTEMENT nul (R(x)≡0, pas juste petit)

        for sym in (Symbol(handle.prefix, :_mlp_w1), Symbol(handle.prefix, :_mha, :_q_W))
            gr = NeuroDSL.node(g, sym; namespace=ns).gradient
            @test gr === nothing || all(iszero, gr)   # ∂L/∂theta EXACTEMENT nul aussi
        end
    end

    @testset "Gradient shadowing (Prop. 4) : alpha0=0, theta aléatoire -- signal immédiat sur alpha" begin
        ns = :shadow_rezero_grad
        g, logits = _build(ns, 105)
        _, handle = NeuroDSL.graft_shadow_block!(g, ns, :layer_2_out, dim, n_heads, hidden_dim;
                                                  alpha0=0f0, zero_out_proj=false)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.demand!(g, :loss; namespace=ns)
        NeuroDSL.backward_graph!(g, :loss; namespace=ns)

        alpha_grad = NeuroDSL.node(g, handle.alpha_sym; namespace=ns).gradient
        @test alpha_grad !== nothing
        @test alpha_grad[1] != 0f0   # signal réel dès le 1er backward (R(x)≠0 car theta aléatoire)

        # Vérification indépendante par différences finies -- ne relit AUCUN nœud intermédiaire
        # non-paramètre (ceux-là sont explicitement effacés par le "nettoyage final" de
        # backward_graph! une fois consommés, comportement existant documenté dans
        # src/backward.jl:432-436 -- pas quelque chose que ce test doit contourner).
        eps = 1f-3
        base_alpha = copy(Array(NeuroDSL.node(g, handle.alpha_sym; namespace=ns).value))
        NeuroDSL.set_params!(g, ns, Dict(handle.alpha_sym => base_alpha .+ eps))
        NeuroDSL.invalidate_all!(g; namespace=ns)
        loss_plus = sum(Array(NeuroDSL.demand!(g, :loss; namespace=ns)))
        NeuroDSL.set_params!(g, ns, Dict(handle.alpha_sym => base_alpha .- eps))
        NeuroDSL.invalidate_all!(g; namespace=ns)
        loss_minus = sum(Array(NeuroDSL.demand!(g, :loss; namespace=ns)))
        fd_grad = (loss_plus - loss_minus) / (2eps)
        @test isapprox(alpha_grad[1], fd_grad; rtol=5f-2, atol=1f-2)

        # theta reste EXACTEMENT gelé tant que alpha=0 (chaîne multipliée par alpha=0 exactement).
        theta_grad = NeuroDSL.node(g, Symbol(handle.prefix, :_mlp_w1); namespace=ns).gradient
        @test theta_grad === nothing || all(iszero, theta_grad)
    end

    @testset "Net2Net (alpha0=1, zero_out_proj=true) : les couches directement zérotées apprennent immédiatement" begin
        ns = :shadow_net2net_grad
        g, logits = _build(ns, 106)
        _, handle = NeuroDSL.graft_shadow_block!(g, ns, :layer_2_out, dim, n_heads, hidden_dim;
                                                  alpha0=1f0, zero_out_proj=true)
        NeuroDSL.invalidate_all!(g; namespace=ns)
        NeuroDSL.demand!(g, :loss; namespace=ns)
        NeuroDSL.backward_graph!(g, :loss; namespace=ns)

        # Propriété Net2Net/Fixup : seule la matrice DIRECTEMENT mise à zéro (celle qui multiplie
        # une activation en amont déjà non nulle) reçoit un gradient immédiat -- PAS les couches
        # encore plus en amont (mlp_w1), qui restent gelées tant que mlp_w2 lui-même n'a pas bougé.
        mlp_w2_grad = NeuroDSL.node(g, Symbol(handle.prefix, :_mlp_w2); namespace=ns).gradient
        output_W_grad = NeuroDSL.node(g, Symbol(handle.prefix, :_mha, :_output_W); namespace=ns).gradient
        @test mlp_w2_grad !== nothing && !all(iszero, mlp_w2_grad)
        @test output_W_grad !== nothing && !all(iszero, output_W_grad)

        mlp_w1_grad = NeuroDSL.node(g, Symbol(handle.prefix, :_mlp_w1); namespace=ns).gradient
        @test mlp_w1_grad === nothing || all(iszero, mlp_w1_grad)   # pas encore -- mlp_w2 doit bouger d'abord
    end
end
