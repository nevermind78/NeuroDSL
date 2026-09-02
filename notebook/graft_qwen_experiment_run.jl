# ══════════════════════════════════════════════════════════════════════════════
# graft_qwen_experiment_run.jl — expérience principale du pré-enregistrement
# (graft_qwen_experiment_preregistration.md) : corriger le suivi d'instruction
# sous distraction (contrainte de format ignorée dès qu'une clause "explique/
# justifie" est ajoutée) via une greffe Gradient Shadowing gelée à
# :layer_25_out sur Qwen2.5-1.5B-Instruct réel.
#
# UN SEUL process : construit le graphe, insère la greffe, évalue AVANT
# entraînement sur held-out A/B + témoin négatif, entraîne 150 pas AdamW sur
# les 6 exemples d'entraînement, réévalue APRÈS, écrit le verdict contre les
# critères pré-enregistrés.
# ══════════════════════════════════════════════════════════════════════════════

using NeuroDSL, JSON, Printf, Random

const MODEL_DIR = joinpath(@__DIR__, "qwen2.5-1.5b-instruct")
const DIM, N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN_DIM, VOCAB_SIZE = 1536, 28, 12, 2, 8960, 151936
const ROPE_THETA, RMS_EPS = 1_000_000.0, 1e-6
const GRAFT_SITE = :layer_25_out
const GRAFT_PREFIX = :qwen_shadow_fix
const GRAFT_HEADS, GRAFT_HIDDEN = 4, 384
const LR = 3f-3
const N_STEPS = 150

# ── Décroissance de LR déclenchée par plateau + arrêt anticipé (fix de la
# "divergence tardive", voir §8.2 du pré-enregistrement) ────────────────────
# Diagnostic (log complet `graft_qwen_experiment_results_run3.json`, un des
# 5 SUCCES_PARTIEL déjà observés) : la perte descend sous 0.1 vers le pas
# ~26-30 (confirmé aussi sur run1/run2 et le 1er run archivé : tous < 0.1
# avant le pas 35) puis reste plate ~120 pas de plus, MAIS avec des pics
# transitoires récurrents de 4-6x l'amplitude du plateau (ex. run3 : pas 127
# loss=0.212, pas 134 loss=0.277, contre un plateau ~0.03-0.05) qui se
# résorbent d'eux-mêmes. Mécanisme plausible : une fois le gradient quasi
# nul en régime de plateau, l'estimateur du 2e moment d'Adam (m2, le
# dénominateur) devient minuscule, donc le pas normalisé
# m1/(sqrt(m2)+eps) peut devenir disproportionné même après le clip de
# gradient PAR ÉLÉMENT déjà actif (clip=1.0 passé à `adamw_step_batched!`,
# noyau `_multi_adamw_kernel!` dans src/kernels.jl) -- ce clip borne g_val
# AVANT l'accumulation dans m1/m2, il ne borne PAS le pas final normalisé
# qui en résulte. Un pic un peu plus grand que ceux déjà observés suffit à
# franchir un bord de non-finitude Float32 -- exactement le mode de
# défaillance "divergence tardive" (pas 93/150 dans le run qui a divergé).
#
# FIX (ne change PAS le protocole pré-enregistré : LR initial 3e-3, budget
# de 150 pas, jeu de données -- tous inchangés ; seul le LR APRÈS détection
# d'un plateau change, et l'entraînement peut s'arrêter avant 150 pas UNE
# FOIS le plateau confirmé stable) : une moyenne mobile courte de la perte
# (fenêtre PLATEAU_WINDOW pas) est comparée à la meilleure moyenne mobile
# observée jusqu'ici (best_avg, ratchet monotone -- PAS une comparaison à
# une fenêtre décalée fixe, qui s'est avérée trop bruitée en simulation
# hors-ligne sur run1/run2/run3 : le bruit du plateau produit parfois une
# "amélioration" apparente de >2% par pur hasard, ce qui remettait le
# compteur à zéro indéfiniment et ne déclenchait JAMAIS de décroissance sur
# run1). Si la moyenne courante n'améliore pas best_avg d'au moins
# PLATEAU_REL_IMPROVEMENT pendant PLATEAU_PATIENCE vérifications
# consécutives, le LR est divisé par PLATEAU_DECAY_FACTOR. Un garde-fou dur
# MIN_STEP_BEFORE_DECAY=40 empêche TOUTE décroissance avant ce pas, avec une
# marge confortable au-delà du pas ~26-35 où la convergence réelle est
# observée sur les 4 lancements archivés -- donc les 30-50 premiers pas
# (là où l'apprentissage réel a lieu) ne sont jamais affectés. Après
# PLATEAU_MAX_DECAYS décroissances (LR final = LR / 4^3 ≈ LR/64, quasi nul),
# l'entraînement s'arrête -- combine décroissance de LR ET arrêt anticipé
# en un seul mécanisme, comme suggéré. Vérifié par simulation hors-ligne
# (rejeu de ce détecteur sur les `loss_history` déjà enregistrées de run1/
# run2/run3, sans aucun calcul GPU) : le premier déclenchement tombe entre
# les pas 55 et 68 selon le run (jamais avant le pas 40, jamais pendant la
# descente initiale), et les 3 décroissances se terminent entre les pas 75
# et 94 -- AVANT le pas 93 où la seule divergence tardive documentée a eu
# lieu, ce qui est le comportement recherché mais reste une marge, pas une
# garantie (voir le run de re-vérification réel pour la mesure honnête).
const PLATEAU_WINDOW = 10
const PLATEAU_REL_IMPROVEMENT = 0.15f0
const PLATEAU_PATIENCE = 10
const MIN_STEP_BEFORE_DECAY = 40
const PLATEAU_DECAY_FACTOR = 4f0
const PLATEAU_MAX_DECAYS = 3
const PYTHON_ENV = get(ENV, "NEURODSL_LLM_PYTHON", raw"C:\Users\Nevermind\anaconda3\envs\neurodsl_llm_check\python.exe")
const TOKENIZER_HELPER = joinpath(@__DIR__, "qwen2_tokenizer_helper.py")

println("── Expérience principale : correction du suivi d'instruction sous distraction ──")
flush(stdout)

# ── 1. Graphe + poids réels ──────────────────────────────────────────────────
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
println("Chargement des poids réels..."); flush(stdout)
NeuroDSL.load_graph!(g, ns, joinpath(MODEL_DIR, "qwen2_neurodsl"); overwrite=true)
println("Poids chargés."); flush(stdout)

# ── 2. Pont tokenizer (serveur persistant, gabarit ChatML réel) ─────────────
println("Démarrage du pont tokenizer..."); flush(stdout)
const TOKENIZER_PROC = open(`$PYTHON_ENV -u $TOKENIZER_HELPER --serve`, "r+")
const _tok_ready = JSON.parse(readline(TOKENIZER_PROC))
println("Tokenizer prêt : ", _tok_ready); flush(stdout)
function _call_helper(req::Dict)
    println(TOKENIZER_PROC, JSON.json(req))
    flush(TOKENIZER_PROC)
    return JSON.parse(readline(TOKENIZER_PROC))
end
encode_chat(messages) = Int.(_call_helper(Dict("action"=>"encode_chat", "messages"=>messages))["ids"])
encode_raw(text) = Int.(_call_helper(Dict("action"=>"encode", "text"=>text))["ids"])
decode_ids(ids::Vector{Int}) = _call_helper(Dict("action"=>"decode", "ids"=>ids))["text"]
const EOS_ID = _call_helper(Dict("action"=>"eos_id"))["id"]
println("EOS_ID = ", EOS_ID); flush(stdout)

# ── 3. Greffe + gel du backbone ──────────────────────────────────────────────
Random.seed!(4242)
NeuroDSL.Backend.CUDA_AVAILABLE && NeuroDSL.CUDA.seed!(4242)
println("Insertion de la greffe à :$(GRAFT_SITE)..."); flush(stdout)
new_out, handle = NeuroDSL.graft_shadow_block!(g, ns, GRAFT_SITE, DIM, GRAFT_HEADS, GRAFT_HIDDEN;
                                                alpha0=0f0, zero_out_proj=false, prefix=GRAFT_PREFIX)
println("Greffe insérée -> $new_out ; alpha_sym=$(handle.alpha_sym)"); flush(stdout)

function freeze_backbone!(g::NeuroDSL.NeuroGraph, ns::Symbol, keep_prefix::Symbol)
    n_frozen = 0
    for (sym, nd) in g.nodes[ns]
        nd.is_param || continue
        startswith(String(sym), String(keep_prefix)) && continue
        nd.is_param = false
        n_frozen += 1
    end
    return n_frozen
end
n_frozen = freeze_backbone!(g, ns, GRAFT_PREFIX)
ps = NeuroDSL.params(g; namespace=ns)
@printf("Backbone gelé : %d nœuds -> is_param=false. params(g;ns) = %d tenseurs entraînables (la greffe).\n",
        n_frozen, length(ps))
flush(stdout)

alpha_before_any = Float64(Array(NeuroDSL.node(g, handle.alpha_sym; namespace=ns).value)[1])
@printf("alpha avant tout entraînement = %.10f (attendu 0.0)\n", alpha_before_any)
flush(stdout)

# ── 4. Jeux de données (§2 du pré-enregistrement) ────────────────────────────
oneword_prompt(q) = "Answer in exactly one word: $q Also, briefly explain your reasoning."
yesno_prompt(q)   = "Reply with only 'yes' or 'no': $q Please justify your answer in detail."

train_set = [
    (oneword_prompt("what is the capital of France?"), "Paris"),
    (oneword_prompt("what color is the sky on a clear day?"), "Blue"),
    (oneword_prompt("what is the largest planet in the solar system?"), "Jupiter"),
    (oneword_prompt("who wrote the play Romeo and Juliet?"), "Shakespeare"),
    (oneword_prompt("what gas do humans need to breathe to survive?"), "Oxygen"),
    (oneword_prompt("what season comes after winter?"), "Spring"),
]

heldout_A = [  # même contrainte, contenu disjoint
    (oneword_prompt("what is the capital of Japan?"), "Tokyo", :oneword),
    (oneword_prompt("what is the tallest mountain on Earth?"), "Everest", :oneword),
    (oneword_prompt("what is the chemical symbol for gold?"), "Au", :oneword),
    (oneword_prompt("how many continents are there on Earth?"), "Seven", :oneword),
]
heldout_B = [  # contrainte disjointe, contenu disjoint
    (yesno_prompt("is the Earth flat?"), "No", :yesno),
    (yesno_prompt("is water made of hydrogen and oxygen?"), "Yes", :yesno),
    (yesno_prompt("is Paris the capital of Germany?"), "No", :yesno),
]
neg_control = [  # pas de contrainte de format -- doit rester verbeux
    "What is the capital of Italy? Also, briefly explain why.",
    "Is 12 an even number? Please justify your answer in detail.",
    "What is the largest ocean on Earth? Also, briefly explain your reasoning.",
]

# ── 5. Génération gloutonne (recalcul complet, cohérent avec les checks) ────
function run_forward!(g, ns, tokens::Vector{Int})
    NeuroDSL.set!(g, :token_ids, tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:length(tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    return Array(NeuroDSL.demand!(g, logits_sym; namespace=ns))
end
function generate_chat(prompt::AbstractString; max_new_tokens::Int=40)
    ids0 = encode_chat([Dict("role"=>"user", "content"=>String(prompt))])
    cur = ids0 .+ 1
    gen0 = Int[]
    for _ in 1:max_new_tokens
        o = run_forward!(g, ns, cur)
        nxt0 = argmax(o[end, :]) - 1
        push!(gen0, nxt0)
        nxt0 == EOS_ID && break
        push!(cur, nxt0 + 1)
    end
    return decode_ids(gen0)
end

# ── 6. Vérificateurs de conformité (§5 du pré-enregistrement) ───────────────
strip_trailing_punct(s) = strip(replace(s, r"[\.\!\?,;:]+$" => ""))
function is_oneword_compliant(resp::AbstractString)
    t = strip_trailing_punct(strip(resp))
    isempty(t) && return false
    return length(split(t)) == 1
end
function is_yesno_compliant(resp::AbstractString)
    t = lowercase(strip_trailing_punct(strip(resp)))
    return t == "yes" || t == "no"
end
is_verbose_enough(resp::AbstractString) = length(split(strip(resp))) >= 5
function content_matches(resp::AbstractString, target::AbstractString)
    t = strip_trailing_punct(strip(resp))
    return occursin(lowercase(target), lowercase(t)) || occursin(lowercase(t), lowercase(target))
end

function eval_set(items; kind_field=true)
    results = []
    for it in items
        if kind_field
            prompt, target, kind = it
        else
            prompt, target, kind = it, "", :verbose
        end
        resp = generate_chat(prompt)
        compliant = kind == :oneword ? is_oneword_compliant(resp) :
                    kind == :yesno   ? is_yesno_compliant(resp) :
                    is_verbose_enough(resp)
        matches = kind in (:oneword, :yesno) ? content_matches(resp, target) : missing
        push!(results, Dict("prompt"=>prompt, "target"=>target, "kind"=>String(kind),
                             "response"=>resp, "compliant"=>compliant, "content_matches"=>matches))
        @printf("    [%s] compliant=%s  content_ok=%s\n      Q: %s\n      A: %s\n",
                kind, compliant, matches, prompt, repr(resp))
        flush(stdout)
    end
    return results
end

println("\n", "═"^78); println("ÉVALUATION AVANT ENTRAÎNEMENT (alpha=0, doit être bit-exact au modèle nu)")
println("═"^78); flush(stdout)
println("-- Held-out A (même contrainte, contenu disjoint) --"); flush(stdout)
before_A = eval_set(heldout_A)
println("-- Held-out B (contrainte disjointe, contenu disjoint) --"); flush(stdout)
before_B = eval_set(heldout_B)
println("-- Témoin négatif (pas de contrainte, verbeux attendu) --"); flush(stdout)
before_neg = eval_set(neg_control; kind_field=false)

# ── 7. Entraînement : 150 pas AdamW, ordre cyclique fixe sur les 6 exemples ─
println("\n", "═"^78); println("ENTRAÎNEMENT -- ", N_STEPS, " pas AdamW, LR=", LR); println("═"^78)
flush(stdout)

NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [logits_sym, :labels], :cross_entropy; namespace=ns))

m1s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]
m2s = [NeuroDSL.Backend.zeros32(dev, size(p.value)...) for p in ps]

function train_example!(g, ns, prompt, target, t, cur_lr::Float32)
    prefix_ids = encode_chat([Dict("role"=>"user", "content"=>prompt)])
    answer_ids = encode_raw(target)
    full0 = vcat(prefix_ids, answer_ids, [EOS_ID])   # 0-indexé
    full1 = full0 .+ 1                                # 1-indexé
    input_tokens = full1[1:end-1]
    label_tokens = full1[2:end]
    NeuroDSL.set!(g, :token_ids, input_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :pos_ids, collect(1:length(input_tokens)); atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.set!(g, :labels, label_tokens; atom_type=NeuroDSL.Datom, namespace=ns)
    NeuroDSL.invalidate_all!(g; namespace=ns)
    loss_val = NeuroDSL.demand!(g, :loss; namespace=ns)
    NeuroDSL.backward_graph!(g, :loss; namespace=ns, prune_frozen=true)
    NeuroDSL.adamw_step_batched!(dev, [p.value for p in ps], [p.gradient for p in ps],
                                  m1s, m2s, cur_lr, 0.9f0, 0.999f0, 1f-8, t, 1f0, 0f0)
    return Float64(sum(Array(loss_val)))
end

# CORRECTIF découvert APRÈS le premier run réussi de ce script (2026-08-31,
# hors notebook, log/JSON de ce premier run archivés séparément -- voir
# graft_qwen_experiment_run.log) : un DEUXIÈME lancement, seed et code
# strictement identiques, a divergé (loss -> NaN à partir du pas 93,
# `alpha` NaN, génération dégénérée en "!!!!..."). NOTE (2026-09-01) : la
# phrase originale ici attribuait cela à "LR=3e-3 sans clip de gradient" --
# **inexact**, corrigé plutôt que laissé tel quel : un clip de gradient PAR
# ÉLÉMENT (clip=1.0) est bel et bien actif depuis le début, dans le noyau
# CUDA `_multi_adamw_kernel!` (src/kernels.jl) appelé par
# `adamw_step_batched!` ci-dessous (l'argument `1f0` avant le dernier `0f0`
# de weight_decay). Ce clip borne g_val AVANT l'accumulation dans les
# moments Adam -- il ne borne PAS le pas final normalisé m1/sqrt(m2+eps),
# ce qui laisse la voie ouverte à l'instabilité de plateau documentée
# au-dessus (PLATEAU_WINDOW etc.), la vraie cause probable désormais ciblée
# par une décroissance de LR plutôt qu'un clip plus agressif. Le protocole
# pré-enregistré (LR initial, budget de pas, jeu de données) N'EST PAS
# changé -- seule une garde de non-finitude est ajoutée ci-dessous : elle
# arrête l'entraînement au premier pas non-fini (perte de calcul inutile sur
# un modèle déjà corrompu), n'influence AUCUN critère de succès/partiel/
# échec, et le résultat est rapporté comme une divergence numérique plutôt
# que forcé dans un JSON invalide (l'échec original de ce notebook :
# `JSON.print` sérialise NaN en `null`, ce que `Printf.@printf` ne peut pas
# formatter -- une vraie erreur de robustesse, indépendante du résultat
# scientifique).
const RUN_ID = get(ENV, "GRAFT_QWEN_RUN_ID", "")
results_suffix = isempty(RUN_ID) ? "" : "_$(RUN_ID)"
results_path = joinpath(@__DIR__, "graft_qwen_experiment_results$(results_suffix).json")

loss_history = Float64[]
alpha_history = Float64[]
lr_history = Float64[]
diverged = false
diverged_at_step = 0
early_stopped = false
early_stopped_at_step = 0
decay_events = []   # Vector of Dict(step=>.., decay_idx=>.., new_lr=>..)
cur_lr = LR
n_decays = 0
plateau_stall = 0
plateau_best_avg = Inf32
for t in 1:N_STEPS
    ex_idx = ((t - 1) % length(train_set)) + 1
    prompt, target = train_set[ex_idx]
    l = train_example!(g, ns, prompt, target, t, cur_lr)
    a = Float64(Array(NeuroDSL.node(g, handle.alpha_sym; namespace=ns).value)[1])
    if !isfinite(l) || !isfinite(a)
        global diverged = true
        global diverged_at_step = t
        @printf("  pas %3d/%d  ex=%d  DIVERGENCE NUMERIQUE (loss=%s, alpha=%s, lr=%.6g) -- arrêt.\n", t, N_STEPS, ex_idx, l, a, cur_lr)
        flush(stdout)
        break
    end
    push!(loss_history, l)
    push!(alpha_history, a)
    push!(lr_history, Float64(cur_lr))

    # -- Détecteur de plateau (ratchet monotone sur une moyenne mobile courte,
    # voir le commentaire au-dessus de PLATEAU_WINDOW pour la justification
    # et la simulation hors-ligne) : actif seulement une fois PLATEAU_WINDOW
    # pas de perte disponibles ET au-delà de MIN_STEP_BEFORE_DECAY.
    if t >= PLATEAU_WINDOW
        cur_avg = Float32(sum(loss_history[end-PLATEAU_WINDOW+1:end]) / PLATEAU_WINDOW)
        if t < MIN_STEP_BEFORE_DECAY
            # Pendant la fenêtre de convergence réelle : on suit best_avg
            # (pour ne pas déclencher un rattrapage brutal du compteur juste
            # après le pas 40) mais on NE COMPTE PAS de stagnation.
            global plateau_best_avg = min(plateau_best_avg, cur_avg)
        else
            if cur_avg < plateau_best_avg * (1f0 - PLATEAU_REL_IMPROVEMENT)
                global plateau_best_avg = cur_avg
                global plateau_stall = 0
            else
                global plateau_stall += 1
            end
            if plateau_stall >= PLATEAU_PATIENCE && n_decays < PLATEAU_MAX_DECAYS
                global n_decays += 1
                global cur_lr = cur_lr / PLATEAU_DECAY_FACTOR
                global plateau_stall = 0
                global plateau_best_avg = cur_avg
                push!(decay_events, Dict("step"=>t, "decay_idx"=>n_decays, "new_lr"=>Float64(cur_lr)))
                @printf("  pas %3d/%d  PLATEAU DÉTECTÉ (%d/%d) -- LR décroît -> %.6g\n",
                        t, N_STEPS, n_decays, PLATEAU_MAX_DECAYS, cur_lr)
                flush(stdout)
                if n_decays >= PLATEAU_MAX_DECAYS
                    global early_stopped = true
                    global early_stopped_at_step = t
                    @printf("  pas %3d/%d  ARRÊT ANTICIPÉ -- %d décroissances de LR atteintes, plateau confirmé stable.\n",
                            t, N_STEPS, PLATEAU_MAX_DECAYS)
                    flush(stdout)
                end
            end
        end
    end

    if t % 10 == 0 || t == 1
        @printf("  pas %3d/%d  ex=%d  loss=%.4f  alpha=%.6f  lr=%.6g\n", t, N_STEPS, ex_idx, l, a, cur_lr)
        flush(stdout)
    end
    early_stopped && break
end

if diverged
    println("\n", "="^80)
    println("VERDICT FINAL : DIVERGENCE_NUMERIQUE")
    @printf("  Entraînement interrompu au pas %d/%d (perte/alpha non finis).\n", diverged_at_step, N_STEPS)
    println("  Pas d'évaluation held-out APRÈS -- un modèle corrompu ne produit aucune")
    println("  information utile (voir la note de correctif ci-dessus).")
    println("="^80)
    open(results_path, "w") do io
        JSON.print(io, Dict(
            "alpha_before_any" => alpha_before_any, "alpha_after_training" => nothing,
            "n_frozen" => n_frozen, "n_graft_params" => length(ps),
            "loss_history" => loss_history, "alpha_history" => alpha_history,
            "lr_history" => lr_history, "decay_events" => decay_events,
            "n_decays" => n_decays, "final_lr" => Float64(cur_lr),
            "before_A" => before_A, "before_B" => before_B, "before_neg" => before_neg,
            "diverged" => true, "diverged_at_step" => diverged_at_step,
            "verdict" => "DIVERGENCE_NUMERIQUE",
        ), 2)
    end
    println("Résultats écrits -> ", results_path)
    close(TOKENIZER_PROC)
    println("Terminé (divergence).")
    exit(0)
end

alpha_after_training = alpha_history[end]
n_steps_run = length(loss_history)
@printf("\nalpha après %d pas = %.6f (attendu != 0.0)\n", n_steps_run, alpha_after_training)
@printf("Perte : premier pas = %.4f, dernier pas = %.4f, moyenne des 10 derniers = %.4f\n",
        loss_history[1], loss_history[end], sum(loss_history[end-9:end])/10)
if early_stopped
    @printf("ARRÊT ANTICIPÉ au pas %d/%d (%d décroissances de LR, plateau confirmé) -- LR final = %.6g\n",
            early_stopped_at_step, N_STEPS, n_decays, cur_lr)
elseif n_decays > 0
    @printf("%d décroissance(s) de LR déclenchée(s) sans atteindre le plafond -- LR final = %.6g\n", n_decays, cur_lr)
else
    println("Aucun plateau détecté (aucune décroissance de LR déclenchée) -- entraînement mené aux 150 pas au LR initial.")
end
flush(stdout)

# ── 8. Évaluation APRÈS entraînement ─────────────────────────────────────────
println("\n", "═"^78); println("ÉVALUATION APRÈS ENTRAÎNEMENT"); println("═"^78); flush(stdout)
println("-- Held-out A --"); flush(stdout)
after_A = eval_set(heldout_A)
println("-- Held-out B --"); flush(stdout)
after_B = eval_set(heldout_B)
println("-- Témoin négatif --"); flush(stdout)
after_neg = eval_set(neg_control; kind_field=false)

# ── 9. Verdict contre les critères pré-enregistrés (§6) ─────────────────────
n_compliant_before = count(r -> r["compliant"], vcat(before_A, before_B))
n_compliant_after  = count(r -> r["compliant"], vcat(after_A, after_B))
n_heldout = length(heldout_A) + length(heldout_B)
n_A_after = count(r -> r["compliant"], after_A)
n_B_after = count(r -> r["compliant"], after_B)
n_neg_after = count(r -> r["compliant"], after_neg)

frac_after = n_compliant_after / n_heldout
verdict = if frac_after >= 5/7 && n_neg_after >= 2
    "SUCCES_COMPLET"
elseif (frac_after >= 2/7 && n_neg_after >= 2) ||
       (n_A_after >= 3 && n_B_after <= 1) ||
       (frac_after >= 5/7 && n_neg_after <= 1)
    "SUCCES_PARTIEL"
else
    "ECHEC"
end

println("\n", "="^80)
println("VERDICT FINAL")
@printf("  Conformité held-out AVANT : %d/%d\n", n_compliant_before, n_heldout)
@printf("  Conformité held-out APRÈS : %d/%d (A=%d/%d, B=%d/%d)\n",
        n_compliant_after, n_heldout, n_A_after, length(heldout_A), n_B_after, length(heldout_B))
@printf("  Témoin négatif encore verbeux APRÈS : %d/%d\n", n_neg_after, length(neg_control))
@printf("  alpha : %.10f -> %.6f\n", alpha_before_any, alpha_after_training)
@printf("  Pas réellement exécutés : %d/%d (arrêt anticipé=%s, %d décroissance(s) de LR, LR final=%.6g)\n",
        n_steps_run, N_STEPS, early_stopped, n_decays, cur_lr)
println("  VERDICT : ", verdict)
println("="^80)

open(results_path, "w") do io
    JSON.print(io, Dict(
        "alpha_before_any" => alpha_before_any, "alpha_after_training" => alpha_after_training,
        "n_frozen" => n_frozen, "n_graft_params" => length(ps),
        "loss_history" => loss_history, "alpha_history" => alpha_history,
        "lr_history" => lr_history, "decay_events" => decay_events,
        "n_decays" => n_decays, "final_lr" => Float64(cur_lr),
        "early_stopped" => early_stopped, "early_stopped_at_step" => early_stopped_at_step,
        "n_steps_run" => n_steps_run,
        "diverged" => false,
        "before_A" => before_A, "before_B" => before_B, "before_neg" => before_neg,
        "after_A" => after_A, "after_B" => after_B, "after_neg" => after_neg,
        "n_compliant_before" => n_compliant_before, "n_compliant_after" => n_compliant_after,
        "n_heldout" => n_heldout, "n_A_after" => n_A_after, "n_B_after" => n_B_after,
        "n_neg_after" => n_neg_after, "n_neg_total" => length(neg_control),
        "verdict" => verdict,
    ), 2)
end
println("Résultats écrits -> ", results_path)

close(TOKENIZER_PROC)
println("Terminé.")
