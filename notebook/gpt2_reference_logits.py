"""gpt2_reference_logits.py -- référence INDÉPENDANTE (HuggingFace transformers, float32, CPU) pour valider le portage
NeuroDSL de GPT-2. Env : conda neurodsl_llm_check. Écrit gpt2/reference.json : ids, logits du dernier token,
états cachés (sortie de chaque bloc) au dernier token."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
D = "gpt2"
tok = AutoTokenizer.from_pretrained(D)
model = AutoModelForCausalLM.from_pretrained(D, torch_dtype=torch.float32).eval()
prompts = ["The capital of France is", "When Mary and John went to the store, John gave a drink to",
           "The opposite of up is", "apple chair river lamp apple chair river", "Better late than"]
out = []
with torch.no_grad():
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        o = model(ids, output_hidden_states=True)
        hs = [h[0, -1].tolist() for h in o.hidden_states]          # 13 : embedding + sortie de chaque bloc (la dernière après ln_f)
        out.append({"prompt": p, "token_ids": ids[0].tolist(), "logits_last": o.logits[0, -1].tolist(),
                    "hidden_last": hs, "top1": int(o.logits[0, -1].argmax()), "top1_str": tok.decode([int(o.logits[0, -1].argmax())])})
        print(repr(p), ids.shape[1], "tokens ->", repr(out[-1]["top1_str"]))
json.dump(out, open(f"{D}/reference.json", "w"))
print("écrit gpt2/reference.json")
