# ══════════════════════════════════════════════════════════════════════════════
# kv_cache_tokenizer_persistent_check.jl — vérifie que le mode serveur
# persistant de qwen2_tokenizer_helper.py (`--serve`) produit EXACTEMENT les
# mêmes résultats que le mode un-coup historique (déjà utilisé toute la
# session), pour les 3 actions réellement utilisées par qwen2.ipynb
# (encode_chat, decode, eos_id) -- avant de le brancher dans le notebook.
# Correctif de performance motivé par le coût mesuré ce soir (~13.5s/appel à
# chat(), tokenizer relancé à froid à chaque fois) -- mais correction avant
# vitesse, même discipline que tout le reste de cette session.
# ══════════════════════════════════════════════════════════════════════════════

using JSON

println("── Vérification correction : serveur tokenizer persistant vs mode un-coup ──"); flush(stdout)

const PYTHON_ENV = raw"C:\Users\Nevermind\anaconda3\envs\neurodsl_llm_check\python.exe"
const TOKENIZER_HELPER = joinpath(@__DIR__, "qwen2_tokenizer_helper.py")

# -- Mode un-coup (référence, code inchangé, déjà utilisé toute la session) --
function call_oneshot(req::Dict)
    input_json = JSON.json(req)
    out = IOBuffer()
    run(pipeline(`$PYTHON_ENV $TOKENIZER_HELPER`, stdin=IOBuffer(input_json), stdout=out))
    return JSON.parse(String(take!(out)))
end

# -- Mode serveur persistant (nouveau) --
println("Démarrage du process serveur persistant..."); flush(stdout)
t0 = time()
proc = open(`$PYTHON_ENV -u $TOKENIZER_HELPER --serve`, "r+")
ready = JSON.parse(readline(proc))
t_startup = time() - t0
println("  serveur prêt en $(round(t_startup,digits=2))s (charge transformers + tokenizer UNE fois) : ", ready); flush(stdout)

function call_serve(proc, req::Dict)
    println(proc, JSON.json(req))
    flush(proc)
    return JSON.parse(readline(proc))
end

# -- Test 1 : ping --
r = call_serve(proc, Dict("action"=>"ping"))
println("ping -> ", r); flush(stdout)

# -- Test 2 : eos_id, un-coup vs serveur --
eos_oneshot = call_oneshot(Dict("action"=>"eos_id"))
eos_serve = call_serve(proc, Dict("action"=>"eos_id"))
println("eos_id : un-coup=", eos_oneshot, "  serveur=", eos_serve, "  identiques=", eos_oneshot==eos_serve); flush(stdout)

# -- Test 3 : encode_chat, plusieurs historiques réalistes --
histories = [
    [Dict("role"=>"user","content"=>"What is the capital of Egypt?")],
    [Dict("role"=>"user","content"=>"What is the capital of Egypt?"),
     Dict("role"=>"assistant","content"=>"The capital of Egypt is Cairo."),
     Dict("role"=>"user","content"=>"And what is a famous food from that city?")],
    [Dict("role"=>"user","content"=>"Write a haiku about the ocean.")],
]
all_encode_match = true
for (i,h) in enumerate(histories)
    e1 = call_oneshot(Dict("action"=>"encode_chat","messages"=>h))
    e2 = call_serve(proc, Dict("action"=>"encode_chat","messages"=>h))
    m = e1 == e2
    global all_encode_match &= m
    println("  historique $i : un-coup=$(length(e1["ids"])) tokens  serveur=$(length(e2["ids"])) tokens  identiques=$m")
    flush(stdout)
end

# -- Test 4 : decode, plusieurs suites d'IDs --
id_lists = [[791,6864,315,15212,374,29055],[15947,596,13061,2136,11]]
all_decode_match = true
for (i,ids) in enumerate(id_lists)
    d1 = call_oneshot(Dict("action"=>"decode","ids"=>ids))
    d2 = call_serve(proc, Dict("action"=>"decode","ids"=>ids))
    m = d1 == d2
    global all_decode_match &= m
    println("  ids $i : un-coup=", repr(d1["text"]), "  serveur=", repr(d2["text"]), "  identiques=", m)
    flush(stdout)
end

# -- Test 5 : timing -- N appels consécutifs, serveur vs un-coup, même requête --
const N_REPEAT = 5
t_oneshot = @elapsed for _ in 1:N_REPEAT
    call_oneshot(Dict("action"=>"encode_chat","messages"=>histories[1]))
end
t_serve = @elapsed for _ in 1:N_REPEAT
    call_serve(proc, Dict("action"=>"encode_chat","messages"=>histories[1]))
end
println("\n$N_REPEAT appels encode_chat répétés :")
println("  mode un-coup (relance Python+transformers+tokenizer à chaque fois) : $(round(t_oneshot,digits=2))s total, $(round(t_oneshot/N_REPEAT,digits=2))s/appel")
println("  mode serveur (process déjà vivant, tokenizer déjà chargé)          : $(round(t_serve,digits=2))s total, $(round(t_serve/N_REPEAT,digits=3))s/appel")
flush(stdout)

close(proc)

println("\n══════════════ VERDICT ══════════════")
println("  eos_id identique : ", eos_oneshot==eos_serve)
println("  encode_chat identique (tous historiques) : ", all_encode_match)
println("  decode identique (toutes suites) : ", all_decode_match)
if eos_oneshot==eos_serve && all_encode_match && all_decode_match
    println("  PASS -- le serveur persistant produit des résultats identiques au mode un-coup.")
else
    println("  FAIL -- divergence réelle, ne pas brancher dans le notebook.")
end
flush(stdout)
