#!/usr/bin/env python
# ══════════════════════════════════════════════════════════════════════════════
# qwen2_tokenizer_helper.py — pont tokenizer pour qwen2.ipynb.
#
# RÉVISION 2026-07-29 -- mode SERVEUR PERSISTANT ajouté (`--serve`), en plus du
# mode un-coup historique (défaut, inchangé). Motif : mesuré ce soir (utilisateur
# + coordinateur, notebook réel) que le mode un-coup relance un interpréteur
# Python + réimporte `transformers` + recharge le tokenizer depuis le disque À
# CHAQUE appel (`encode_chat`, `decode_ids`) -- ~13.5s de coût fixe PAR appel à
# `chat(...)` dans qwen2.ipynb, TOUJOURS présent, indépendant du cache KV, et
# désormais le plus gros poste de temps une fois le JIT CUDA amorti. En mode
# `--serve`, le tokenizer est chargé UNE SEULE FOIS au démarrage, puis le
# process reste vivant et traite une requête JSON par ligne lue sur stdin
# (JSON Lines -- un objet JSON complet par ligne, PAS de retour à la ligne
# À L'INTÉRIEUR d'un objet -- `json.dumps` par défaut n'en émet jamais),
# écrit la réponse (une ligne JSON) sur stdout, et flush -- exactement le
# même contrat de messages que le mode un-coup, juste répété sans jamais
# quitter le process. AUCUNE nouvelle dépendance Julia dans les deux cas :
# toujours du pipe de process externe (`run`/`open`), jamais un paquet Julia
# HTTP/RPC.
#
# Mode un-coup (défaut, INCHANGÉ -- compatibilité arrière pour tout script qui
# appelle encore ce fichier sans `--serve`, ex. les scripts de mesure isolée
# de ce soir) :
#   Lit UN objet JSON sur stdin, écrit UN objet JSON sur stdout, quitte.
#
# Actions (identiques dans les deux modes) :
#   {"action": "encode_chat", "messages": [{"role":..., "content":...}, ...]}
#     -> {"ids": [...]}  -- applique le VRAI template ChatML du tokenizer
#        (`apply_chat_template`, add_generation_prompt=true), pas une
#        réimplémentation manuelle de la chaîne de gabarit.
#   {"action": "decode", "ids": [...]}
#     -> {"text": "..."}  -- skip_special_tokens=true (n'affiche pas
#        <|im_end|> etc. dans la réponse montrée à l'utilisateur).
#   {"action": "eos_id"}
#     -> {"id": 151645}  -- id du token de fin de tour (<|im_end|>), lu depuis
#        le tokenizer réel, pas codé en dur côté Julia.
#   {"action": "ping"}  (mode serveur seulement, pour vérifier que le process
#        est bien vivant sans dépenser un vrai appel tokenizer)
#     -> {"pong": true}
# ══════════════════════════════════════════════════════════════════════════════
import sys, json, os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen2.5-1.5b-instruct")

def _handle(req, tok):
    action = req["action"]
    if action == "encode":
        # Tokenisation BRUTE, SANS gabarit ChatML -- ajouté le 2026-07-29 pour
        # la recherche de paires appariées sur du texte "nu" (complétion de
        # langage, pas un tour de chat) -- add_special_tokens=False pour ne
        # PAS insérer <|endoftext|>/BOS implicite, comportement le plus
        # prévisible pour comparer des continuations mot à mot.
        ids = tok(req["text"], add_special_tokens=False)["input_ids"]
        return {"ids": [int(i) for i in ids]}
    elif action == "encode_chat":
        enc = tok.apply_chat_template(req["messages"], tokenize=True, add_generation_prompt=True)
        try:
            ids = enc["input_ids"]        # BatchEncoding (dict-like), versions récentes
        except (TypeError, KeyError):
            ids = enc                      # liste brute d'entiers, versions plus anciennes
        return {"ids": [int(i) for i in ids]}
    elif action == "decode":
        text = tok.decode(req["ids"], skip_special_tokens=True)
        return {"text": text}
    elif action == "eos_id":
        return {"id": tok.convert_tokens_to_ids("<|im_end|>")}
    elif action == "ping":
        return {"pong": True}
    else:
        return {"error": f"action inconnue : {action}"}

def main_oneshot():
    from transformers import AutoTokenizer
    req = json.loads(sys.stdin.read())
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    sys.stdout.write(json.dumps(_handle(req, tok)))

def main_serve():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    # Ligne de synchronisation : le premier octet écrit sur stdout signale à
    # l'appelant Julia que le chargement (transformers + tokenizer depuis le
    # disque, le VRAI coût fixe qu'on cherche à payer une seule fois) est
    # terminé -- sans ça, l'appelant n'a aucun moyen de savoir quand la
    # première vraie requête peut être envoyée sans bloquer indéfiniment.
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = _handle(req, tok)
        except Exception as e:
            resp = {"error": str(e)}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    if "--serve" in sys.argv:
        main_serve()
    else:
        main_oneshot()
