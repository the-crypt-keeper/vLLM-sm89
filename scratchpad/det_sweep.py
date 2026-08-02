"""Where does greedy determinism break? Same prompt, R repeats, sweeping length.

Bug #3 made greedy reproducible -- verified on short prompts. pc0 shows it is
NOT reproducible at 24k with caching off and det-moe on. This finds the length
where it breaks. Prior boundaries in this model all sat at 2048 (index_topk=512
vs the ratio-4 compressed candidate count), so the sweep straddles it.

usage: det_sweep.py <base_url> <model> [repeats]
"""
import json, sys, os, urllib.request, collections

BASE, MODEL = sys.argv[1], sys.argv[2]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MODELDIR = "/media/4TBNVME/models/DeepSeek-V4-Flash-0731"
LENGTHS = [256, 1024, 1792, 2304, 4096, 8192, 16384, 24000]

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

def ask(text):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": text}],
               "temperature": 0, "max_tokens": 64, "seed": 0,
               "reasoning_effort": "low"}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        body = json.load(r)
    ch = body["choices"][0]["message"]
    return (ch.get("content") or "").strip(), (body.get("usage") or {}).get("prompt_tokens")

print(f"{MODEL}, {R} repeats per length\n")
hdr = f"{'ctx tok':>8} | {'prompt':>7} | {'distinct':>8} | verdict"
print(hdr); print("-" * 58)
out = []
for L in LENGTHS:
    text = tok.decode(ALL[:L]) + Q
    seen = collections.Counter()
    ptok = None
    for _ in range(R):
        t, ptok = ask(text)
        seen[t] += 1
    d = len(seen)
    verdict = "reproducible" if d == 1 else f"NONDETERMINISTIC ({d} outputs)"
    print(f"{L:>8} | {ptok:>7} | {d:>8} | {verdict}")
    out.append({"ctx": L, "prompt_tokens": ptok, "distinct": d,
                "samples": list(seen.keys())})

json.dump({"model": MODEL, "repeats": R, "sweep": out},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"det_sweep_{MODEL}.json"), "w"), indent=2)
