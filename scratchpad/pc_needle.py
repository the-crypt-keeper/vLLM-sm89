"""Correctness gate for prefix caching: needles with unique verifiable answers.

The pc_ab.py test maximises sensitivity to numerical perturbation ("name one
constant" has many valid answers, so 1 ULP flips it). This one does the
opposite: every question has exactly ONE right answer, so a difference is a
WRONG answer, not a different-but-valid one. That is what a cache-correctness
bug would look like -- reused blocks would corrupt retrieval at specific depths.

Shares the agent access pattern: one long prefix, many queries against it.

usage: pc_needle.py <base_url> <model> <out.json> [prefix_tokens] [n_needles]
"""
import json, sys, os, time, urllib.request

BASE, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PREFIX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 24000
N = int(sys.argv[5]) if len(sys.argv) > 5 else 10
MODELDIR = "/media/4TBNVME/models/DeepSeek-V4-Flash-0731"

from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{MODELDIR}/tokenizer.json")

# deterministic pseudo-random codes, no RNG (keeps runs comparable)
CODES = [f"{(i * 7919 + 104729) % 100000:05d}" for i in range(N)]

src = []
root = "/home/cirrascale/vllm-ds4/moet-fork/overlay/vllm/vllm"
for dirpath, _, names in sorted(os.walk(root)):
    for n in sorted(names):
        if n.endswith(".py"):
            try:
                src.append(open(os.path.join(dirpath, n), encoding="utf-8").read())
            except Exception:
                pass
ids = tok.encode("\n\n".join(src), add_special_tokens=False).ids[:PREFIX_TOKENS]

# splice needles at evenly spaced depths
chunk = len(ids) // (N + 1)
pieces, needles = [], []
for i, code in enumerate(CODES):
    seg = tok.decode(ids[i * chunk:(i + 1) * chunk])
    key = f"ALPHA-{i:02d}"
    pieces.append(seg + f"\n# CHECKPOINT {key} VALUE IS {code}\n")
    needles.append((key, code, round(100 * (i + 1) / (N + 1))))
pieces.append(tok.decode(ids[N * chunk:]))
PREFIX = "\n".join(pieces)
print(f"prefix {len(tok.encode(PREFIX, add_special_tokens=False).ids)} tok, {N} needles")

def ask(q):
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": PREFIX + "\n\n" + q}],
               "temperature": 0, "max_tokens": 64, "seed": 0,
               "reasoning_effort": "low"}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        body = json.load(r)
    ch = body["choices"][0]["message"]
    u = body.get("usage") or {}
    return (ch.get("content") or "").strip(), (u.get("prompt_tokens_details") or {}).get("cached_tokens"), time.time() - t0

rows, hits = [], 0
for key, code, depth in needles:
    text, cached, wall = ask(
        f"What is the value of CHECKPOINT {key}? Reply with the 5-digit number only.")
    ok = code in text
    hits += ok
    rows.append({"key": key, "depth_pct": depth, "expect": code,
                 "got": text[:40], "ok": ok, "cached": cached,
                 "wall_s": round(wall, 2)})
    print(f"  {key} @{depth:>3}% expect={code} got={text[:20]!r} "
          f"cached={cached} {'OK' if ok else 'MISS'}")

json.dump({"model": MODEL, "prefix_tokens": PREFIX_TOKENS,
           "hits": hits, "n": N, "rows": rows}, open(OUT, "w"), indent=2)
print(f"\n{MODEL}: {hits}/{N} needles retrieved -> {OUT}")
