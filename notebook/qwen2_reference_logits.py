#!/usr/bin/env python
# ══════════════════════════════════════════════════════════════════════════════
# qwen2_reference_logits.py — référence INDÉPENDANTE (HuggingFace transformers,
# environnement conda isolé neurodsl_llm_check) pour la porte de vérification
# numérique de code4/plan LLM réel : mêmes séquences de token IDs que celles
# passées au graphe NeuroDSL chargé, logits + continuation gloutonne dumpés en
# JSON pour un diff numérique fait ailleurs (pas de comparaison ici -- ce
# script ne fait QUE produire la référence).
#
# float32 explicite partout (le checkpoint est en bf16, NeuroDSL calcule tout
# en Float32) -- upcast immédiat après chargement pour que la comparaison ne
# mélange pas deux précisions différentes côté référence.
# ══════════════════════════════════════════════════════════════════════════════
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "c:/Users/Nevermind/Desktop/NeuroDSL/notebook/qwen2.5-1.5b-instruct"

print("Chargement du tokenizer et du modèle (CPU, float32)...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
model.eval()

PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time, there was a",
]

results = []
with torch.no_grad():
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids
        out = model(ids)
        logits = out.logits[0]  # (seqlen, vocab)
        # continuation gloutonne (argmax), 8 tokens, recalcul complet à chaque
        # pas -- même style que generate_text de real_llm.ipynb côté NeuroDSL,
        # PAS model.generate() (qui utiliserait un cache KV -- on veut la même
        # opération, "un passage avant complet par token", des deux côtés).
        cur = ids.clone()
        gen_ids = []
        for _ in range(8):
            o = model(cur)
            nxt = int(o.logits[0, -1].argmax())
            gen_ids.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)

        results.append({
            "prompt": prompt,
            "token_ids": ids[0].tolist(),
            "logits_last_position": logits[-1].tolist(),
            "logits_all_positions_argmax": logits.argmax(dim=-1).tolist(),
            "greedy_continuation_ids": gen_ids,
            "greedy_continuation_text": tok.decode(gen_ids),
        })
        print(f"  prompt={prompt!r}  n_tokens={len(ids[0])}  top1_last={tok.decode([logits[-1].argmax()])!r}"
              f"  continuation={tok.decode(gen_ids)!r}", flush=True)

with open(MODEL_DIR + "/reference_logits.json", "w", encoding="utf-8") as f:
    json.dump(results, f)
print("Écrit ->", MODEL_DIR + "/reference_logits.json")
