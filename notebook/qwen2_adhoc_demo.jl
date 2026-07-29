# ══════════════════════════════════════════════════════════════════════════════
# qwen2_adhoc_demo.jl — étape 4 (mesure de temps) + démonstration directe.
#
# (1) Mesure le temps d'un SEUL passage avant sur le graphe Qwen2.5-1.5B-Instruct
#     chargé, à plusieurs longueurs de contexte -- résout la question du cache
#     KV avec un chiffre réel plutôt qu'une estimation.
# (2) Génère une courte suite (gloutonne, recalcul complet à chaque token --
#     même motif que `generate_text` de real_llm.ipynb, déjà validé contre la
#     référence HuggingFace à l'étape 3) sur 3 prompts NEUFS, tokenisés
#     extérieurement (même discipline que la porte de parité), et décode le
#     résultat en texte lisible via le vocabulaire HF.
#
# Progression imprimée et flushée à CHAQUE étape (prompt, pas de génération) --
# pas de bloc final unique.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6

println("Construction du graphe Qwen2.5-1.5B-Instruct..."); flush(stdout)
dev = NeuroDSL.Backend.CUDADevice()
ns = :qwen2
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
NeuroDSL.set!(g, :pos_ids, collect(1:8); atom_type=NeuroDSL.Datom, namespace=ns)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
out = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                           batched_attn=true, n_kv_heads=N_KV_HEADS,
                           qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out, :final_norm; namespace=ns)
logits_sym = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)
println("Chargement des poids (qwen2_neurodsl.json/.bin)..."); flush(stdout)
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
println("Prêt.\n"); flush(stdout)

function run_forward!(g, ns, tokens::Vector{Int})
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    t0 = time()
    o = Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))
    dt = time() - t0
    return o, dt
end

# ── (1) Temps d'un seul passage avant, plusieurs longueurs de contexte ──────
println("═"^74); println("(1) Temps d'un passage avant complet, par longueur de contexte"); println("═"^74)
flush(stdout)
for seqlen in (8, 16, 32, 64, 128, 256, 512)
    toks = rand(1:VOCAB_SIZE, seqlen)
    _, dt_warm = run_forward!(g, ns, toks)      # 1er appel : compilation CUDA incluse, ignoré pour la mesure
    times = Float64[]
    for _ in 1:5
        _, dt = run_forward!(g, ns, toks)
        push!(times, dt)
    end
    med = sort(times)[3]
    println("  seqlen=$(lpad(seqlen,4))  1er appel (compilation incluse)=$(round(dt_warm,digits=3))s   médiane sur 5 rappels=$(round(med*1000,digits=1))ms")
    flush(stdout)
end

# ── (2) Génération sur 3 prompts neufs ──────────────────────────────────────
println("\n", "═"^74); println("(2) Génération (gloutonne, recalcul complet par token) sur 3 prompts neufs"); println("═"^74)
flush(stdout)

adhoc = JSON.parsefile(joinpath(MODEL_DIR, "adhoc_prompts.json"))

# Décodage minimal : relit vocab.json (HF) pour reconstruire id -> texte.
# Ce n'est PAS un décodeur BPE général -- juste la table id->token de ce
# tokenizer précis, suffisante pour afficher une continuation lisible ici.
vocab = JSON.parsefile(joinpath(MODEL_DIR, "vocab.json"))
id2tok = Dict{Int,String}(v => k for (k, v) in vocab)
function decode_ids(ids::Vector{Int})
    pieces = String[]
    for id in ids
        t = get(id2tok, id, "<?$(id)?>")
        # convention GPT-2/BPE : Ġ = espace précédent, Ċ = retour à la ligne
        t = replace(t, "Ġ" => " "); t = replace(t, "Ċ" => "\n")
        push!(pieces, t)
    end
    return join(pieces)
end

for entry in adhoc
    prompt = entry["prompt"]
    tokens0 = Int.(entry["token_ids"])
    println("\n--- Prompt : ", repr(prompt), " ---"); flush(stdout)
    cur = copy(tokens0) .+ 1   # 1-indexé pour NeuroDSL
    gen_ids0 = Int[]           # 0-indexé (HF) pour le décodage
    for step in 1:12
        o, dt = run_forward!(g, ns, cur)
        nxt0 = argmax(o[end, :]) - 1
        push!(gen_ids0, nxt0)
        push!(cur, nxt0 + 1)
        println("    pas $step/12 : token=$(nxt0)  (\"", decode_ids([nxt0]), "\")  temps=$(round(dt,digits=3))s")
        flush(stdout)
    end
    println("  Continuation complète : ", repr(decode_ids(gen_ids0)))
    flush(stdout)
end

println("\nTerminé.")
