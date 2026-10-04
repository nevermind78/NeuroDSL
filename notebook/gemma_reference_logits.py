"""gemma_reference_logits.py -- référence INDÉPENDANTE (transformers, float32, CPU, attention eager pour appliquer le
plafonnement des scores) pour valider le portage NeuroDSL de Gemma-2-2B. Env : conda neurodsl_llm_check.
Écrit gemma-2-2b/reference.json : ids (avec <bos>), logits du dernier token, états cachés au dernier token."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
D = "gemma-2-2b"
tok = AutoTokenizer.from_pretrained(D)
model = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float32, attn_implementation="eager").eval()
prompts = ["The capital of France is", "When Mary and John went to the store, John gave a drink to",
           "The opposite of up is", "apple chair river lamp apple chair river", "Better late than"]
out = []
with torch.no_grad():
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        o = model(ids, output_hidden_states=True)
        out.append({"prompt": p, "token_ids": ids[0].tolist(), "logits_last": o.logits[0, -1].tolist(),
                    "hidden_last": [h[0, -1].tolist() for h in o.hidden_states],
                    "top1": int(o.logits[0, -1].argmax()), "top1_str": tok.decode([int(o.logits[0, -1].argmax())])})
        print(repr(p), ids.shape[1], "tokens ->", repr(out[-1]["top1_str"]))
json.dump(out, open(f"{D}/reference.json", "w"))
print("écrit", f"{D}/reference.json")
