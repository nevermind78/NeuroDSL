# Utilitaires partagés par les 3 scripts F3 (contrôle / script A / script B) --
# voir notebook/graph_surgery.ipynb, section F3.
using NeuroDSL, Random

const VS, D, NH, HD, NL, PL = 20, 64, 4, 128, 3, 8

# Seed par pas (pas de RNG partagé à faire persister) -- rend chaque pas reproductible
# indépendamment de l'historique, condition nécessaire pour reprendre exactement après un
# vrai redémarrage de processus sans sérialiser l'état du RNG (serialization.jl ne le fait
# délibérément pas -- limitation documentée).
step_data(t::Int) = NeuroDSL.sample_induction_sequence(MersenneTwister(100_000 + t), VS, PL)

function train_step!(g, ns, logits, m1, m2, t, params_list)
    tokens, labels = step_data(t)
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :labels, labels; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    loss = NeuroDSL.demand!(g, :loss; namespace=ns)
    NeuroDSL.backward_graph!(g, :loss; namespace=ns)
    for p in params_list
        p.gradient === nothing && continue
        NeuroDSL.adamw_step!(NeuroDSL.Backend.CPUDevice(), p.value, p.gradient, m1[p.name], m2[p.name],
                              3f-3, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
    end
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Float64(sum(Array(loss)))
end
