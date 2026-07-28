# Test diagnostique (PAS le notebook final) : confirme empiriquement le
# défaut du comparateur souple (sigmoid((x_i-x_j)/tau) avec tau FIXE en
# valeur absolue) avant de corriger tri.ipynb pour de bon.
using LinearAlgebra, NeuroDSL, Statistics

function neural_sort_raw(x::Vector{Float32}; tau=0.05f0, gamma=0.01f0, scale_invariant=false)
    n = length(x)
    diffs = x .- x'
    if scale_invariant
        s = std(x) + 1f-6
        diffs = diffs ./ s
    end
    S = 1f0 ./ (1f0 .+ exp.(.-diffs ./ tau))
    ranks = sum(S, dims=2) .- 0.5f0
    positions = Float32.(0:n-1)'
    dist_to_pos = -(ranks .- positions) .^ 2
    P = exp.(dist_to_pos ./ gamma)
    P ./= sum(P, dims=2)
    return vec(P' * x), P
end

function check_perm(orig::Vector{Float32}, sorted_::Vector{Float32}; atol=1f-3)
    ok_sorted = issorted(sorted_)
    ok_perm = sort(orig) ≈ sort(sorted_)  # même multi-ensemble de valeurs (permutation)
    return ok_sorted, ok_perm
end

println("=== Cas 1 : valeurs bien séparées (le cas déjà testé dans le notebook) ===")
x1 = Float32[8.4, 2.1, 9.9, 3.14, 5.5]
s1, _ = neural_sort_raw(x1)
ok_s, ok_p = check_perm(x1, s1)
println("résultat=", round.(s1,digits=3), "  issorted=", ok_s, "  est_permutation=", ok_p)

println("\n=== Cas 2 (ADVERSARIAL) : écarts petits par rapport à tau=0.05 (comparateur NON corrigé) ===")
x2 = Float32[1000.001, 1000.050, 1000.020, 1000.005, 1000.035]  # écarts ~0.005-0.05, comparable à tau
s2, P2 = neural_sort_raw(x2; tau=0.05f0)
ok_s2, ok_p2 = check_perm(x2, s2)
println("résultat=", round.(s2,digits=4), "  issorted=", ok_s2, "  est_permutation=", ok_p2)
println("P2 (lignes) :"); display(round.(P2, digits=3))

println("\n=== Cas 2 avec comparateur invariant à l'échelle (CORRIGÉ) ===")
s2b, P2b = neural_sort_raw(x2; tau=0.05f0, scale_invariant=true)
ok_s2b, ok_p2b = check_perm(x2, s2b)
println("résultat=", round.(s2b,digits=4), "  issorted=", ok_s2b, "  est_permutation=", ok_p2b)

println("\n=== Cas 3 (ADVERSARIAL) : très petite échelle absolue (tau=0.05 énorme en comparaison) ===")
x3 = Float32[0.00001, 0.00002, 0.00003, 0.000005, 0.000015]
s3, _ = neural_sort_raw(x3; tau=0.05f0)
ok_s3, ok_p3 = check_perm(x3, s3)
println("résultat=", s3, "  issorted=", ok_s3, "  est_permutation=", ok_p3)

println("\n=== Cas 3 avec comparateur invariant à l'échelle (CORRIGÉ) ===")
s3b, _ = neural_sort_raw(x3; tau=0.05f0, scale_invariant=true)
ok_s3b, ok_p3b = check_perm(x3, s3b)
println("résultat=", s3b, "  issorted=", ok_s3b, "  est_permutation=", ok_p3b)

println("\n=== Cas 4 (le cas bien séparé, avec correctif -- ne doit pas régresser) ===")
s4, _ = neural_sort_raw(x1; scale_invariant=true)
ok_s4, ok_p4 = check_perm(x1, s4)
println("résultat=", round.(s4,digits=3), "  issorted=", ok_s4, "  est_permutation=", ok_p4)
