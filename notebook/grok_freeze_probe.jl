# Sonde ciblée : pourquoi test_loss se gèle exactement alors que les poids
# continuent de bouger (correctif Float64 + eps=1e-8 original, p=3, DIM=128).
ENV["GROK_DEVICE"] = "cpu"
ENV["GROK_P"] = "3"
ENV["GROK_EPS"] = "1e-8"
ENV["GROK_WD"] = "1.0"
ENV["GROK_CLIP"] = "1e6"
ENV["GROK_STEPS"] = "1"   # on pilote la boucle nous-mêmes, pas besoin du budget interne
ENV["GROK_SEED"] = "999"

using NeuroDSL, Random, LinearAlgebra
include("grokking_demo_neurodsl.jl")  # définit build_grok_graph, set_input!, P, etc. et exécute 1 pas (rapide)

# Reconstruction manuelle propre pour contrôler chaque pas nous-mêmes
dev = NeuroDSL.Backend.CPUDevice()
nsx2 = :probe
Random.seed!(999)
g2, logits2, mo2, post2 = build_grok_graph(dev, nsx2)
pairs2, labels2 = build_dataset()
train_idx2, test_idx2 = train_test_split(pairs2, labels2; seed=598)

ps2 = NeuroDSL.params(g2; namespace=nsx2)
m1s2 = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps2]
m2s2 = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps2]
accs2 = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps2]
n_train2 = length(train_idx2)

function per_example_test_losses(g, nsx, pairs, labels, test_idx)
    losses = Float64[]
    for idx in test_idx
        a, b = pairs[idx]; v = labels[idx]
        set_input!(g, nsx, a, b)
        NeuroDSL.set!(g, :final_label, [v]; atom_type=NeuroDSL.Datom, namespace=nsx)
        NeuroDSL.invalidate_all!(g; namespace=nsx)
        push!(losses, Float64(sum(Array(NeuroDSL.demand!(g, :loss; namespace=nsx)))))
    end
    return losses
end

for t in 1:2500
    for acc in accs2; fill!(acc, 0f0); end
    for idx in train_idx2
        a, b = pairs2[idx]; v = labels2[idx]
        set_input!(g2, nsx2, a, b)
        NeuroDSL.set!(g2, :final_label, [v]; atom_type=NeuroDSL.Datom, namespace=nsx2)
        NeuroDSL.invalidate_all!(g2, ; namespace=nsx2)
        NeuroDSL.demand!(g2, :loss; namespace=nsx2)
        NeuroDSL.backward_graph!(g2, :loss; namespace=nsx2)
        for (i, p) in enumerate(ps2)
            p.gradient === nothing && continue
            accs2[i] .+= p.gradient
        end
    end
    for (i, p) in enumerate(ps2)
        p.gradient === nothing && continue
        p.gradient .= accs2[i] ./ Float32(n_train2)
        NeuroDSL.adamw_step!(dev, p.value, p.gradient, m1s2[i], m2s2[i], LR, 0.9f0, BETA2, EPS, t, CLIP, WD)
    end
    NeuroDSL.invalidate_all!(g2; namespace=nsx2)

    if t % 100 == 0 || t == 2500
        pel = per_example_test_losses(g2, nsx2, pairs2, labels2, test_idx2)
        n_floored = count(x -> x > 23.0, pel)
        i_unembed = findfirst(==(:unembed_W), [p.name for p in ps2])
        nu = norm(Array(ps2[i_unembed].value))
        gnorm = sum(Float64(norm(Array(p.gradient))) for p in ps2 if p.gradient !== nothing)
        println("t=$t  n_floored=$n_floored  |unembed|=", round(nu,digits=8), "  sum|grad|=", round(gnorm,sigdigits=4), "  per-example = ", round.(pel, digits=6), "  sum/6=", round(sum(pel)/6, digits=6))
        flush(stdout)
    end
end
