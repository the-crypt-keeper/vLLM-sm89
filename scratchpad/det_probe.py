"""Isolate the forward pass: max_tokens=1, compare first-token logprobs.

Comparing generated text is a thresholded readout -- it only shows a difference
once a perturbation flips an argmax. Logprobs are the continuous underlying
signal, so this measures the perturbation directly and quantifies it in nats.
Same method bug #3 used (max_tokens=1 isolates prefill: no decode, no KV reuse,
no sampling).

usage: det_probe.py <base_url> <model> [repeats]
"""
import json, sys, os, urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 6
MODELDIR = "/media/4TBNVME/models/DeepSeek-V4-Flash-0731"
LENGTHS = ([int(x) for x in os.environ["LENGTHS"].split(",")]
           if os.environ.get("LENGTHS")
           else [256, 1024, 1792, 2304, 3072, 4096, 6144, 8192, 16384, 24000])

from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{MODELDIR}/tokenizer.json")

src = []
root = "/home/cirrascale/vllm-ds4/moet-fork/overlay/vllm/vllm"
for dirpath, _, names in sorted(os.walk(root)):
    for n in sorted(names):
        if n.endswith(".py"):
            try:
                src.append(open(os.path.join(dirpath, n), encoding="utf-8").read())
            except Exception:
                pass
ALL = tok.encode("\n\n".join(src), add_special_tokens=False).ids
Q = "\n\nIn one sentence, summarise what the code above does."

def probe(text):
    """Return (top_token, [(tok, logprob), ...]) for the single next token."""
    payload = {"model": MODEL, "messages": [{"role": "user", "content": text}],
               "temperature": 0, "max_tokens": 1, "seed": 0,
               "logprobs": True, "top_logprobs": 20,
               "reasoning_effort": "low"}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        body = json.load(r)
    lp = body["choices"][0]["logprobs"]["content"][0]
    return lp["token"], [(e["token"], e["logprob"]) for e in lp["top_logprobs"]]

print(f"{MODEL}, {R} repeats, max_tokens=1 (forward pass only)\n")
hdr = f"{'ctx tok':>8} | {'argmax':>7} | {'lp sets':>7} | {'max |dlogprob|':>14} | verdict"
print(hdr); print("-" * 72)
rows = []
for L in LENGTHS:
    text = tok.decode(ALL[:L]) + Q
    tops, vecs = [], []
    for _ in range(R):
        t, v = probe(text)
        tops.append(t); vecs.append(v)
    n_argmax = len(set(tops))
    n_sets = len(set(json.dumps(v) for v in vecs))
    # max absolute logprob delta on the shared top-1..20 by token string
    maxd = 0.0
    base = dict(vecs[0])
    for v in vecs[1:]:
        for t, p in v:
            if t in base:
                maxd = max(maxd, abs(p - base[t]))
    verdict = ("bit-identical" if n_sets == 1
               else ("logprobs differ, argmax stable" if n_argmax == 1
                     else f"ARGMAX FLIPS ({n_argmax})"))
    print(f"{L:>8} | {n_argmax:>7} | {n_sets:>7} | {maxd:>14.3e} | {verdict}")
    rows.append({"ctx": L, "argmax_variants": n_argmax, "logprob_sets": n_sets,
                 "max_abs_dlogprob": maxd})

json.dump({"model": MODEL, "repeats": R, "rows": rows},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "det_probe.json"), "w"), indent=2)
