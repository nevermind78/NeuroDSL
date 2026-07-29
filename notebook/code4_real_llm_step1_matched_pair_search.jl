# ══════════════════════════════════════════════════════════════════════════════
# code4_real_llm_step1_matched_pair_search.jl — ÉTAPE 1 (recherche, pas encore
# de causalité) du test des affirmations de code4.tex sur le VRAI
# Qwen2.5-1.5B-Instruct.
#
# But : trouver une "paire appariée" (Definition~\ref{def:pair}, code4.tex) qui
# se produit dans du langage réel, pas construite comme la tâche synthétique
# (marker task format-A/format-B) -- deux entrées $(x_A,x_B)$ partageant un
# axe de contraste, un même sélecteur fixe $s(x):=\mathrm{logit}(c_1)-
# \mathrm{logit}(c_2)$ (MÊME paire de candidats $c_1,c_2$ pour les deux), tel
# que $g_A:=s(x_A)>0>s(x_B)=:g_B$.
#
# GRAPHE UTILISÉ : recalcul complet (`ns_load`, comme `qwen2_parity_check.jl`),
# PAS le graphe de cache KV (`ns_chat`) -- consigne explicite du coordinateur :
# `capture_activations!`/`patch_node!` n'ont aucune notion de l'état temporel
# du cache et le risque n'a jamais été résolu, seulement différé. Cette étape
# ne fait que mesurer des logits, pas encore de patch -- mais le graphe
# construit ici est celui qui sera réutilisé pour l'étape 2 (causalité), donc
# recalcul complet dès maintenant.
#
# 4 FAMILLES DE CANDIDATS, ENGAGÉES À L'AVANCE (pas de recherche ouverte) :
#   1. IOI (échange donneur/destinataire, cf. Wang et al. 2022)
#   2. Indice de sentiment (aimé/détesté -> adjectif positif/négatif)
#   3. Ordre temporel (qui est parti/arrivé en premier)
#   4. Comparatif marqueur (plus grand/plus petit -- le plus proche de la
#      tâche synthétique, "conçu" plutôt que "trouvé naturellement" -- gardé
#      en repli, signalé comme tel dans le verdict).
# 5 instances lexicales par famille (noms/nombres/adjectifs variés) pour tester
# la robustesse, pas un seul cas chanceux. Verdict "propre" = TOUTES les
# instances de la famille satisfont g_A>0>g_B avec une marge saine
# (|g|>MARGIN_THRESHOLD), pas seulement le bon signe de justesse.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON

const T0 = time()
_log(msg) = (println("[t+$(round(time()-T0,digits=1))s] ", msg); flush(stdout))

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6
const MARGIN_THRESHOLD = 1.0   # logits -- seuil pour dire qu'une marge est "saine", pas juste du bon signe

_log("══ ÉTAPE 1 : recherche d'une paire appariée dans du langage réel (Qwen2.5-1.5B) ══")

# ── Graphe : recalcul complet uniquement (ns_load), PAS de cache KV ──────────
dev = NeuroDSL.Backend.CUDADevice()
ns = :qwen2_probe
g = NeuroDSL.NeuroGraph(namespace=ns, device=dev)
NeuroDSL.set!(g, :token_ids, ones(Int, 8); atom_type=NeuroDSL.Datom, namespace=ns)
tok_emb = NeuroDSL.Embedding(VOCAB_SIZE, DIM)(g, :token_ids, :tok; namespace=ns)
out_sym = NeuroDSL.LlamaModel(N_LAYERS, DIM, N_HEADS, HIDDEN_DIM;
                               batched_attn=true, n_kv_heads=N_KV_HEADS,
                               qkv_bias=true, use_rope=true, rope_theta=ROPE_THETA)(g, tok_emb; namespace=ns)
final_norm = NeuroDSL.LayerNorm(DIM; eps=RMS_EPS)(g, out_sym, :final_norm; namespace=ns)
logits_sym = NeuroDSL.Linear(DIM, VOCAB_SIZE, bias=false)(g, final_norm, :lm_head; namespace=ns)
_log("Graphe construit. Chargement des poids réels (qwen2_neurodsl.json/.bin)...")
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
_log("Poids chargés.")

function run_forward!(g, ns, tokens1idx::Vector{Int})
    NeuroDSL.set!(g, :token_ids, tokens1idx; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))
end

# ── Tokenizer : serveur persistant, action "encode" BRUTE (pas de ChatML) ──
const PYTHON_ENV = raw"C:\Users\Nevermind\anaconda3\envs\neurodsl_llm_check\python.exe"
const TOKENIZER_HELPER = joinpath(@__DIR__, "qwen2_tokenizer_helper.py")
_log("Démarrage du serveur tokenizer persistant...")
const TOK_PROC = open(`$PYTHON_ENV -u $TOKENIZER_HELPER --serve`, "r+")
JSON.parse(readline(TOK_PROC))  # ligne "ready"
function tok_call(req::Dict)
    println(TOK_PROC, JSON.json(req))
    flush(TOK_PROC)
    return JSON.parse(readline(TOK_PROC))
end
encode_raw(text::String) = Int.(tok_call(Dict("action"=>"encode", "text"=>text))["ids"])  # 0-indexé HF
_log("Serveur tokenizer prêt.")

"""Renvoie l'id (0-indexé HF) du PREMIER token du mot `w` (avec espace de tête
si besoin) et signale si `w` occupe plus d'un token (approximation standard
en interprétabilité -- on compare le premier token, mais on le dit)."""
function first_token_id(w::String)
    ids = encode_raw(w)
    return ids[1], length(ids)
end

"""Mesure s(x) = logit(cand1) - logit(cand2) à la DERNIÈRE position de `prompt`."""
function selector(prompt::String, cand1_id0::Int, cand2_id0::Int)
    ids0 = encode_raw(prompt)
    ids1 = ids0 .+ 1
    out = run_forward!(g, ns, ids1)
    row = out[end, :]
    return Float64(row[cand1_id0+1] - row[cand2_id0+1]), length(ids0)
end

struct Instance
    label::String
    prompt_A::String
    prompt_B::String
    cand1::String   # correct pour A (s doit être >0)
    cand2::String   # correct pour B (s doit être <0)
end

families = [
    ("1. IOI (échange donneur/destinataire)", [
        Instance("Mary/John",  "When Mary and John went to the store, John gave the bag to",
                                "When Mary and John went to the store, Mary gave the bag to", " Mary", " John"),
        Instance("Alice/Bob",  "When Alice and Bob went to the park, Bob gave the ball to",
                                "When Alice and Bob went to the park, Alice gave the ball to", " Alice", " Bob"),
        Instance("Sarah/Tom",  "When Sarah and Tom went to the market, Tom gave the flowers to",
                                "When Sarah and Tom went to the market, Sarah gave the flowers to", " Sarah", " Tom"),
        Instance("Anna/Mike",  "When Anna and Mike went to the cafe, Mike gave the book to",
                                "When Anna and Mike went to the cafe, Anna gave the book to", " Anna", " Mike"),
        Instance("Lucy/Sam",   "When Lucy and Sam went to the beach, Sam gave the towel to",
                                "When Lucy and Sam went to the beach, Lucy gave the towel to", " Lucy", " Sam"),
    ]),
    ("2. Indice de sentiment (aimé/détesté)", [
        Instance("film 1", "The critics loved the film; they said the acting was absolutely",
                            "The critics hated the film; they said the acting was absolutely", " great", " terrible"),
        Instance("film 2", "Everyone loved the new restaurant and said the food was truly",
                            "Everyone hated the new restaurant and said the food was truly", " excellent", " awful"),
        Instance("book 1", "The reviewers loved the novel and called the ending",
                            "The reviewers hated the novel and called the ending", " brilliant", " terrible"),
        Instance("trip 1", "We loved the vacation; the hotel service was",
                            "We hated the vacation; the hotel service was", " wonderful", " horrible"),
        Instance("show 1", "Fans loved the concert and said the performance was",
                            "Fans hated the concert and said the performance was", " amazing", " dreadful"),
    ]),
    ("3. Ordre temporel (parti en premier)", [
        Instance("Mary/John",  "First Mary left the house, then John arrived. The person who left first was",
                                "First John left the house, then Mary arrived. The person who left first was", " Mary", " John"),
        Instance("Alice/Bob",  "First Alice left the office, then Bob arrived. The person who left first was",
                                "First Bob left the office, then Alice arrived. The person who left first was", " Alice", " Bob"),
        Instance("Sarah/Tom",  "First Sarah left the room, then Tom arrived. The person who left first was",
                                "First Tom left the room, then Sarah arrived. The person who left first was", " Sarah", " Tom"),
        Instance("Anna/Mike",  "First Anna left the station, then Mike arrived. The person who left first was",
                                "First Mike left the station, then Anna arrived. The person who left first was", " Anna", " Mike"),
        Instance("Lucy/Sam",   "First Lucy left the party, then Sam arrived. The person who left first was",
                                "First Sam left the party, then Lucy arrived. The person who left first was", " Lucy", " Sam"),
    ]),
    ("4. Comparatif marqueur (plus grand/plus petit) [repli, conçu pas trouvé]", [
        Instance("47/12", "Between 47 and 12, the larger number is",
                          "Between 47 and 12, the smaller number is", " 47", " 12"),
        Instance("83/19", "Between 83 and 19, the larger number is",
                          "Between 83 and 19, the smaller number is", " 83", " 19"),
        Instance("56/31", "Between 56 and 31, the larger number is",
                          "Between 56 and 31, the smaller number is", " 56", " 31"),
        Instance("94/7",  "Between 94 and 7, the larger number is",
                          "Between 94 and 7, the smaller number is", " 94", " 7"),
        Instance("62/28", "Between 62 and 28, the larger number is",
                          "Between 62 and 28, the smaller number is", " 62", " 28"),
    ]),
]

results = Dict{String,Any}[]
winning_family = nothing

for (fname, instances) in families
    _log("")
    _log("═"^70)
    _log("Famille $fname -- $(length(instances)) instances")
    family_clean = true
    family_rows = Dict{String,Any}[]
    for inst in instances
        c1_id, c1_ntok = first_token_id(inst.cand1)
        c2_id, c2_ntok = first_token_id(inst.cand2)
        gA, nA = selector(inst.prompt_A, c1_id, c2_id)
        gB, nB = selector(inst.prompt_B, c1_id, c2_id)
        is_matched = gA > 0 && gB < 0
        is_healthy = is_matched && abs(gA) > MARGIN_THRESHOLD && abs(gB) > MARGIN_THRESHOLD
        family_clean &= is_healthy
        note = c1_ntok > 1 || c2_ntok > 1 ? "  (⚠ candidat multi-token, on compare le 1er token)" : ""
        _log("  [$(inst.label)] g_A=$(round(gA,digits=3))  g_B=$(round(gB,digits=3))  " *
             "apparié=$is_matched  marge_saine=$is_healthy$note")
        push!(family_rows, Dict("label"=>inst.label, "g_A"=>gA, "g_B"=>gB,
                                 "matched"=>is_matched, "healthy"=>is_healthy,
                                 "cand1"=>inst.cand1, "cand2"=>inst.cand2))
    end
    n_healthy = count(r->r["healthy"], family_rows)
    _log("  -- résumé famille : $n_healthy/$(length(instances)) instances avec marge saine (|g|>$MARGIN_THRESHOLD)")
    push!(results, Dict("family"=>fname, "rows"=>family_rows, "n_healthy"=>n_healthy, "n_total"=>length(instances)))
    if family_clean && winning_family === nothing
        global winning_family = fname
        _log("  ✅ FAMILLE PROPRE -- toutes les instances satisfont le critère de paire appariée avec marge saine.")
    end
end

close(TOK_PROC)

println("\n", "═"^70)
println("VERDICT ÉTAPE 1")
println("═"^70)
for r in results
    println("  $(r["family"]) : $(r["n_healthy"])/$(r["n_total"]) marge saine")
end
if winning_family !== nothing
    println("\n  RÉSULTAT : famille retenue pour l'étape 2 -> $winning_family")
else
    println("\n  RÉSULTAT : AUCUNE des $(length(families)) familles engagées à l'avance n'est entièrement propre.")
    println("  Ceci est en soi un résultat réel à rapporter honnêtement -- pas à forcer.")
end
flush(stdout)

open(joinpath(MODEL_DIR, "code4_step1_results.json"), "w") do io
    JSON.print(io, Dict("results"=>results, "winning_family"=>winning_family, "margin_threshold"=>MARGIN_THRESHOLD))
end
println("\nÉcrit -> ", joinpath(MODEL_DIR, "code4_step1_results.json"))
