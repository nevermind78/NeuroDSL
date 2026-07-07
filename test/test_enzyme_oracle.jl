# test_enzyme_oracle.jl — Oracle de gradient EXACT (précision machine) via Enzyme.jl.
#
# Complémentaire, pas concurrent, de grad_check (test_backward.jl:11-50, différences finies,
# erreur de troncature O(eps^2)) : compare directement la valeur produite par chaque entrée
# GRAD_RULES écrite à la main (src/backward.jl) contre une différentiation INDÉPENDANTE du
# même calcul via Enzyme (transformation de code au niveau LLVM, pas d'approximation numérique).
# Un désaccord entre les deux serait un signal à documenter, jamais à ignorer.
#
# Deux mécanismes, choisis selon comment l'op existe réellement dans le code (jamais de formule
# dupliquée qui pourrait diverger silencieusement de src/kernels.jl) :
#   - noyau mutant (:rmsnorm, :swiglu, :softmax -- ont un `*_fwd!` dédié dans src/kernels.jl) :
#     on différentie directement le VRAI noyau de production via le mode reverse "mutant"
#     d'Enzyme (l'ombre du buffer de sortie est semée avec dy, les ombres des entrées sont lues
#     après coup) -- aucun risque de formule dupliquée, aucun coût de "double forward".
#   - wrapper scalaire (:matmul, :mse_loss -- pas de noyau dédié) : loss_fn(args..., dy) =
#     dot(dy, f(args...)), technique déjà validée dans notebook/Enzyme.ipynb.
#
# Piège trouvé empiriquement en construisant ce fichier (à ne pas réintroduire) : si `dy` est
# capturé en closure sur une variable GLOBALE non typée plutôt que passé en paramètre explicite
# `Enzyme.Const(dy)`, Enzyme échoue avec "Duplicated Returns not yet handled" (l'inférence de
# type perd la concrétude du type de retour du wrapper scalaire). Tous les wrappers ci-dessous
# passent donc `dy` en paramètre explicite, jamais en closure sur un global.

using Enzyme
using LinearAlgebra

function enzyme_grad_rmsnorm(x::AbstractMatrix{Float32}, gamma::AbstractVector{Float32},
                              dy::AbstractMatrix{Float32}; eps::Float32=1f-6)
    dev = NeuroDSL.Backend.CPUDevice()
    nr, nc = size(x)
    out = zeros(Float32, nr, nc); rms_inv = zeros(Float32, nr)
    dx = zeros(Float32, nr, nc); dgamma = zeros(Float32, nc); drms_inv = zeros(Float32, nr)
    # Enzyme.autodiff ne transmet pas de mot-clé à la fonction différentiée (`eps=eps` serait
    # interprété comme un mot-clé de `autodiff` lui-même, qui n'en accepte pas) -- on fige eps
    # via une closure locale sur un paramètre concret (Float32), pas sur un global non typé.
    rmsnorm_fixed_eps!(dev, out, rms_inv, x, gamma) =
        NeuroDSL.rmsnorm_fwd!(dev, out, rms_inv, x, gamma; eps=eps)
    Enzyme.autodiff(Enzyme.Reverse, rmsnorm_fixed_eps!, Enzyme.Const(dev),
                    Enzyme.Duplicated(out, copy(dy)),
                    Enzyme.Duplicated(rms_inv, drms_inv),
                    Enzyme.Duplicated(x, dx),
                    Enzyme.Duplicated(gamma, dgamma))
    return dx, dgamma
end

function enzyme_grad_swiglu(gate::AbstractMatrix{Float32}, up::AbstractMatrix{Float32},
                             dy::AbstractMatrix{Float32})
    dev = NeuroDSL.Backend.CPUDevice()
    out = zeros(Float32, size(gate))
    dgate = zeros(Float32, size(gate)); dup = zeros(Float32, size(up))
    Enzyme.autodiff(Enzyme.Reverse, NeuroDSL.swiglu_fwd!, Enzyme.Const(dev),
                    Enzyme.Duplicated(out, copy(dy)),
                    Enzyme.Duplicated(gate, dgate), Enzyme.Duplicated(up, dup))
    return dgate, dup
end

function enzyme_grad_softmax(x::AbstractMatrix{Float32}, dy::AbstractMatrix{Float32})
    dev = NeuroDSL.Backend.CPUDevice()
    out = zeros(Float32, size(x))
    dx = zeros(Float32, size(x))
    Enzyme.autodiff(Enzyme.Reverse, NeuroDSL.softmax_fwd!, Enzyme.Const(dev),
                    Enzyme.Duplicated(out, copy(dy)), Enzyme.Duplicated(x, dx))
    return dx
end

function enzyme_grad_matmul(A::AbstractMatrix{Float32}, B::AbstractMatrix{Float32},
                             dy::AbstractMatrix{Float32}; trans_b::Bool)
    f(A, B) = trans_b ? A * B' : A * B
    loss_fn(A, B, dy) = dot(vec(dy), vec(f(A, B)))
    dA = zeros(Float32, size(A)); dB = zeros(Float32, size(B))
    Enzyme.autodiff(Enzyme.Reverse, loss_fn, Enzyme.Duplicated(A, dA), Enzyme.Duplicated(B, dB),
                    Enzyme.Const(dy))
    return dA, dB
end

function enzyme_grad_mse_loss(out::AbstractMatrix{Float32}, target::AbstractMatrix{Float32},
                               dy::AbstractVector{Float32})
    loss_fn(out, target, dy) = dot(dy, NeuroDSL.mse_loss_fwd(out, target))
    dout = zeros(Float32, size(out)); dtarget = zeros(Float32, size(target))
    Enzyme.autodiff(Enzyme.Reverse, loss_fn, Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(target, dtarget), Enzyme.Const(dy))
    return dout, dtarget
end

# ─────────────────────────────────────────────────────────────────────────────
# @testset — un par entrée GRAD_RULES, sur les MÊMES fixtures que test_backward.jl
# (lignes 65-186), pour que différences-finies et oracle exact soient directement
# comparables sur les mêmes entrées.
#
# Piège trouvé empiriquement (même trouvaille que plus tôt cette session, sur
# viz.jl/capture_snapshot) : `backward_graph!` efface `.gradient` de tout nœud
# NON-PARAMÈTRE une fois qu'il a été "consommé" (distribué à ses propres entrées) --
# src/backward.jl:432-436, `if !nd_out.is_param; nd_out.gradient = nothing; end`.
# Lire le gradient d'un nœud intermédiaire (:H, :out, :p, :C, ou même :L lui-même,
# qui n'est pas un paramètre) APRÈS l'appel donne toujours `nothing`. Plutôt que de
# lire ce gradient dans le graphe, chaque perte finale ci-dessous est `sum_matrix`
# (dont le gradient amont est trivialement `ones(...)`, pas une formule qui
# pourrait diverger silencieusement de src/kernels.jl -- c'est la définition même
# de `sum`) : `dy` est donc construit directement, jamais lu sur un nœud effacé.
# ─────────────────────────────────────────────────────────────────────────────
@testset "Enzyme — oracle de gradient exact (tol=1e-4)" begin
    dev = NeuroDSL.Backend.CPUDevice()
    etol = Float32(1e-4)

    @testset ":rmsnorm" begin
        g = NeuroDSL.NeuroGraph(namespace=:e_rmsnorm, device=dev)
        # :X marqué is_param=true (contrairement au fixture original de test_backward.jl, qui ne
        # teste que :gamma) -- src/backward.jl:440-449 efface .gradient de TOUTE feuille non-
        # paramètre en fin de backward_graph!, y compris :X ; il faut le lire ici, pas là-bas.
        NeuroDSL.set!(g, :X,     randn(Float32, 2, 4); is_param=true)
        NeuroDSL.set!(g, :gamma, ones(Float32, 4); is_param=true)
        NeuroDSL.@addrules g :e_rmsnorm begin
            H = rmsnorm(X, gamma)
            L = sum_matrix(H)
        end
        NeuroDSL.demand!(g, :L; namespace=:e_rmsnorm)
        NeuroDSL.backward_graph!(g, :L; namespace=:e_rmsnorm)

        X_val     = Array(NeuroDSL.node(g, :X; namespace=:e_rmsnorm).value)
        gamma_val = Array(NeuroDSL.node(g, :gamma; namespace=:e_rmsnorm).value)
        H_val     = Array(NeuroDSL.node(g, :H; namespace=:e_rmsnorm).value)
        dH        = ones(Float32, size(H_val))   # d(sum_matrix)/dH ≡ ones, pas une formule qui dérive
        dX_rules     = Array(NeuroDSL.node(g, :X; namespace=:e_rmsnorm).gradient)
        dgamma_rules = Array(NeuroDSL.node(g, :gamma; namespace=:e_rmsnorm).gradient)

        dX_enz, dgamma_enz = enzyme_grad_rmsnorm(X_val, gamma_val, dH)
        @test isapprox(dX_rules, dX_enz; atol=etol)
        @test isapprox(dgamma_rules, dgamma_enz; atol=etol)
    end

    @testset ":swiglu" begin
        g = NeuroDSL.NeuroGraph(namespace=:e_swiglu, device=dev)
        NeuroDSL.set!(g, :gate, randn(Float32, 2, 4); is_param=true)
        NeuroDSL.set!(g, :up,   randn(Float32, 2, 4); is_param=true)
        NeuroDSL.@addrules g :e_swiglu begin
            out = swiglu(gate, up)
            L   = sum_matrix(out)
        end
        NeuroDSL.demand!(g, :L; namespace=:e_swiglu)
        NeuroDSL.backward_graph!(g, :L; namespace=:e_swiglu)

        gate_val = Array(NeuroDSL.node(g, :gate; namespace=:e_swiglu).value)
        up_val   = Array(NeuroDSL.node(g, :up; namespace=:e_swiglu).value)
        out_val  = Array(NeuroDSL.node(g, :out; namespace=:e_swiglu).value)
        dout     = ones(Float32, size(out_val))
        dgate_rules = Array(NeuroDSL.node(g, :gate; namespace=:e_swiglu).gradient)
        dup_rules   = Array(NeuroDSL.node(g, :up; namespace=:e_swiglu).gradient)

        dgate_enz, dup_enz = enzyme_grad_swiglu(gate_val, up_val, dout)
        @test isapprox(dgate_rules, dgate_enz; atol=etol)
        @test isapprox(dup_rules, dup_enz; atol=etol)
    end

    @testset ":softmax" begin
        # Piège trouvé par la vérification de falsifiabilité (voir plan, étape "Vérification"
        # point 1) : L = sum_matrix(softmax(x)) est DÉGÉNÉRÉ -- chaque ligne de softmax(x) somme
        # à 1 par construction, donc dL/dx ≡ 0 identiquement quel que soit x, quelle que soit la
        # règle backward (correcte ou cassée). Un dp uniforme (ones) est exactement dans le noyau
        # du Jacobien de softmax_bwd! (p.*(dy .- sum(p.*dy))). Correction : pondérer par une
        # matrice constante w (feuille non-paramètre, jamais lue en gradient) via :mul avant le
        # sum_matrix -- d(sum_matrix(mul(p,w)))/dp = w (identité de la multiplication élément par
        # élément, pas une formule qui pourrait diverger de src/kernels.jl), et w non-uniforme
        # casse la dégénérescence.
        g = NeuroDSL.NeuroGraph(namespace=:e_softmax, device=dev)
        NeuroDSL.set!(g, :x, randn(Float32, 3, 4); is_param=true)
        NeuroDSL.set!(g, :w, randn(Float32, 3, 4); is_param=false)
        NeuroDSL.@addrules g :e_softmax begin
            p  = softmax(x)
            pw = mul(p, w)
            L  = sum_matrix(pw)
        end
        NeuroDSL.demand!(g, :L; namespace=:e_softmax)
        NeuroDSL.backward_graph!(g, :L; namespace=:e_softmax)

        x_val = Array(NeuroDSL.node(g, :x; namespace=:e_softmax).value)
        w_val = Array(NeuroDSL.node(g, :w; namespace=:e_softmax).value)
        dp    = w_val   # d(sum_matrix(mul(p,w)))/dp ≡ w, identité de :mul, pas une formule dupliquée
        dx_rules = Array(NeuroDSL.node(g, :x; namespace=:e_softmax).gradient)

        dx_enz = enzyme_grad_softmax(x_val, dp)
        @test isapprox(dx_rules, dx_enz; atol=etol)
    end

    @testset ":matmul trans_b=false" begin
        g = NeuroDSL.NeuroGraph(namespace=:e_mm, device=dev)
        NeuroDSL.set!(g, :A, randn(Float32, 2, 3); is_param=true)
        NeuroDSL.set!(g, :B, randn(Float32, 3, 4); is_param=true)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:C, [:A, :B], :matmul;
            attrs=Dict{Symbol,Any}(:trans_b=>false), namespace=:e_mm))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [:C], :sum_matrix; namespace=:e_mm))
        NeuroDSL.demand!(g, :L; namespace=:e_mm)
        NeuroDSL.backward_graph!(g, :L; namespace=:e_mm)

        A_val = Array(NeuroDSL.node(g, :A; namespace=:e_mm).value)
        B_val = Array(NeuroDSL.node(g, :B; namespace=:e_mm).value)
        C_val = Array(NeuroDSL.node(g, :C; namespace=:e_mm).value)
        dC    = ones(Float32, size(C_val))
        dA_rules = Array(NeuroDSL.node(g, :A; namespace=:e_mm).gradient)
        dB_rules = Array(NeuroDSL.node(g, :B; namespace=:e_mm).gradient)

        dA_enz, dB_enz = enzyme_grad_matmul(A_val, B_val, dC; trans_b=false)
        @test isapprox(dA_rules, dA_enz; atol=etol)
        @test isapprox(dB_rules, dB_enz; atol=etol)
    end

    @testset ":matmul trans_b=true" begin
        g = NeuroDSL.NeuroGraph(namespace=:e_mmt, device=dev)
        NeuroDSL.set!(g, :A, randn(Float32, 2, 3); is_param=true)
        NeuroDSL.set!(g, :B, randn(Float32, 4, 3); is_param=true)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:C, [:A, :B], :matmul;
            attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=:e_mmt))
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [:C], :sum_matrix; namespace=:e_mmt))
        NeuroDSL.demand!(g, :L; namespace=:e_mmt)
        NeuroDSL.backward_graph!(g, :L; namespace=:e_mmt)

        A_val = Array(NeuroDSL.node(g, :A; namespace=:e_mmt).value)
        B_val = Array(NeuroDSL.node(g, :B; namespace=:e_mmt).value)
        C_val = Array(NeuroDSL.node(g, :C; namespace=:e_mmt).value)
        dC    = ones(Float32, size(C_val))
        dA_rules = Array(NeuroDSL.node(g, :A; namespace=:e_mmt).gradient)
        dB_rules = Array(NeuroDSL.node(g, :B; namespace=:e_mmt).gradient)

        dA_enz, dB_enz = enzyme_grad_matmul(A_val, B_val, dC; trans_b=true)
        @test isapprox(dA_rules, dA_enz; atol=etol)
        @test isapprox(dB_rules, dB_enz; atol=etol)
    end

    @testset ":mse_loss" begin
        g = NeuroDSL.NeuroGraph(namespace=:e_mse, device=dev)
        NeuroDSL.set!(g, :pred,   randn(Float32, 2, 4); is_param=true)
        NeuroDSL.set!(g, :target, randn(Float32, 2, 4); is_param=true)
        NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:L, [:pred, :target], :mse_loss; namespace=:e_mse))
        NeuroDSL.demand!(g, :L; namespace=:e_mse)
        NeuroDSL.backward_graph!(g, :L; namespace=:e_mse)

        pred_val   = Array(NeuroDSL.node(g, :pred; namespace=:e_mse).value)
        target_val = Array(NeuroDSL.node(g, :target; namespace=:e_mse).value)
        dL         = Float32[1.0]   # seed racine de backward_graph! (src/backward.jl:344), pas lu sur le graphe
        dpred_rules   = Array(NeuroDSL.node(g, :pred; namespace=:e_mse).gradient)
        dtarget_rules = Array(NeuroDSL.node(g, :target; namespace=:e_mse).gradient)

        dpred_enz, dtarget_enz = enzyme_grad_mse_loss(pred_val, target_val, dL)
        @test isapprox(dpred_rules, dpred_enz; atol=etol)
        @test isapprox(dtarget_rules, dtarget_enz; atol=etol)
    end
end
