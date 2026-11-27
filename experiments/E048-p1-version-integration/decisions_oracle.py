"""Independent oracle for /v1/decisions: recompute label logprobs with HF transformers."""
import json
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/var/tmp/lfm3"
resp_path = sys.argv[1]
resp = json.load(open(resp_path))

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, dtype=torch.float16
).to("cuda").eval()


def label_logprobs(prompt_ids, label_ids):
    ids = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits[0, -1].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    return [float(logprobs[i]) for i in label_ids]


def softmax(values):
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = math.fsum(exps)
    return [e / total for e in exps]


ok = True
for qid, ans in resp["answers"].items():
    lps = label_logprobs(ans["prompt_token_ids"], ans["label_token_ids"])
    probs = softmax(lps)
    mass = math.fsum(math.exp(v) for v in lps)
    names = list(ans["probabilities"].keys())
    server = list(ans["probabilities"].values())
    d_prob = max(abs(a - b) for a, b in zip(server, probs))
    d_mass = abs(ans["label_mass"] - mass)
    passed = d_prob < 2e-2 and d_mass < 2e-2
    ok &= passed
    print(f"{qid:4s} type={ans['type']:8s} "
          f"server={[round(p, 6) for p in server]} oracle={[round(p, 6) for p in probs]} "
          f"max|dprobs|={d_prob:.2e} server_mass={ans['label_mass']:.6f} "
          f"oracle_mass={mass:.6f} max|dmass|={d_mass:.2e} -> {'OK' if passed else 'MISMATCH'}")
    if ans["type"] == "choice":
        print(f"      server choice={ans['choice']} oracle argmax={names[probs.index(max(probs))]}")
print("ORACLE_RESULT", "PASS" if ok else "FAIL")
