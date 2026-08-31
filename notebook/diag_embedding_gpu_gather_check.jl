# Vérifie que le gather GPU-natif (`E[idx, :]` / `output_buffer .=
# view(E, idx, :)`) fonctionne SANS "scalar indexing disallowed" sur CUDA,
# ET mesure le gain réel vs le chemin CPU round-trip actuel
# (`src/dispatch.jl:600-615`, op `:embedding`) -- avant de toucher au code
# de production. Table de la taille réelle de Qwen2.5-1.5B (vocab=151936,
# dim=1536, Float32 ≈ 933 Mo).
using CUDA, Statistics

const VOCAB, DIM = 151936, 1536
E = CUDA.rand(Float32, VOCAB, DIM)
CUDA.synchronize()
println("Table embedding : $(VOCAB)x$(DIM) Float32 ≈ $(round(VOCAB*DIM*4/1024^2,digits=1)) Mo sur GPU")

idx = [12345]   # décodage : 1 seul token
out_gpu = CUDA.zeros(Float32, 1, DIM)

# ── Chemin ACTUEL (dispatch.jl:600-615) : aller-retour CPU complet ─────────
function embedding_cpu_roundtrip!(out, E, idx)
    E_cpu = Array(E)
    out_cpu = Array(out)
    for (i, row) in enumerate(idx)
        out_cpu[i, :] .= E_cpu[row, :]
    end
    out .= CuArray(out_cpu)
    return out
end

# ── Chemin PROPOSÉ : gather GPU-natif, aucun aller-retour de la table ──────
function embedding_gpu_native!(out, E, idx)
    out .= view(E, idx, :)
    return out
end

# Vérification de CORRECTION d'abord (avant tout chrono) :
r1 = Array(embedding_cpu_roundtrip!(copy(out_gpu), E, idx))
r2 = Array(embedding_gpu_native!(copy(out_gpu), E, idx))
println("Résultats identiques (CPU roundtrip vs GPU natif) : ", isapprox(r1, r2))

println("\nWarm-up...")
embedding_cpu_roundtrip!(out_gpu, E, idx); CUDA.synchronize()
embedding_gpu_native!(out_gpu, E, idx); CUDA.synchronize()

function bench_stacked(f::Function, n_iters::Int, n_trials::Int)
    CUDA.synchronize()
    trial_times = Float64[]
    for _ in 1:n_trials
        t = CUDA.@elapsed begin
            for _ in 1:n_iters
                f()
            end
        end
        push!(trial_times, t / n_iters)
    end
    return trial_times
end

const N_ITERS, N_TRIALS = 20, 5   # peu d'itérations : chaque appel CPU roundtrip transfère ~933Mo, coûteux à répéter beaucoup

t_cpu = bench_stacked(() -> embedding_cpu_roundtrip!(out_gpu, E, idx), N_ITERS, N_TRIALS)
t_gpu = bench_stacked(() -> embedding_gpu_native!(out_gpu, E, idx), N_ITERS, N_TRIALS)

fmt(x) = "$(round(1000*mean(x),digits=3))ms ± $(round(1000*std(x),digits=3))ms"
println("\n", "="^70)
println("Op :embedding, 1 token, table $(VOCAB)x$(DIM) (taille réelle Qwen2.5-1.5B)")
println("="^70)
println("Chemin ACTUEL (aller-retour CPU complet de la table)   : $(fmt(t_cpu))")
println("Chemin PROPOSÉ (gather GPU natif, view+broadcast)       : $(fmt(t_gpu))")
println("Speedup : $(round(mean(t_cpu)/mean(t_gpu),digits=1))x")
