"""Prefix-caching A/B gate, shaped like mini-swe-agent's access pattern.

Long prefix, then a growing conversation: append assistant turn, append user
turn, re-send everything. Records each turn's output, token counts and wall
time. Run once per server config; compare the JSONs with pc_cmp.py.

usage: pc_ab.py <base_url> <model> <out.json> [prefix_tokens] [turns]
"""
import json, sys, time, os, urllib.request

BASE, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PREFIX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 24000
TURNS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
MODELDIR = "/media/4TBNVME/models/DeepSeek-V4-Flash-0731"

from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{MODELDIR}/tokenizer.json")
ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)

# --- long, realistic prefix from actual source files ------------------------
src = []
root = "/home/cirrascale/vllm-ds4/moet-fork/overlay/vllm/vllm"
for dirpath, _, names in sorted(os.walk(root)):
    for n in sorted(names):
        if n.endswith(".py"):
            p = os.path.join(dirpath, n)
            try:
                src.append(f"### FILE: {p}\n" + open(p, encoding="utf-8").read())
            except Exception:
                pass
ids = tok.encode("\n\n".join(src), add_special_tokens=False).ids[:PREFIX_TOKENS]
PREFIX = tok.decode(ids)
print(f"prefix: {ntok(PREFIX)} tokens, {TURNS} turns, model={MODEL}")

QUESTIONS = [
    "In one sentence: what is the most common purpose of the files shown?",
    "Name one function defined in the excerpt. Just the name.",
    "Name one Python module imported anywhere in the excerpt. Just the name.",
    "Is there any class definition in the excerpt? Answer yes or no.",
    "Name one ALL_CAPS constant in the excerpt. Just the name.",
    "In one short sentence, what does the first file appear to do?",
    "Name a second function defined in the excerpt. Just the name.",
    "Reply with exactly the word: ACKNOWLEDGED",
]

def post(payload, timeout=3600):
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t0

# the agent loop: prefix + Q0, then append (assistant, user) each turn
messages = [{"role": "user", "content": PREFIX + "\n\n" + QUESTIONS[0]}]
results = []
for i in range(TURNS):
    body, wall = post({
        "model": MODEL, "messages": messages,
        "temperature": 0, "max_tokens": 1024, "seed": 0,
        "reasoning_effort": "low",
    })
    ch = body["choices"][0]["message"]
    text = (ch.get("content") or "").strip()
    # the reasoning parser routes thinking into reasoning_content; when
    # max_tokens truncates inside the think block, content is empty and all the
    # signal is over there. Compare both or the gate compares '' to ''.
    reasoning = (ch.get("reasoning_content") or "").strip()
    usage = body.get("usage") or {}
    det = usage.get("prompt_tokens_details") or {}
    rec = {"turn": i, "prompt_tokens": usage.get("prompt_tokens"),
           "completion_tokens": usage.get("completion_tokens"),
           "cached_tokens": det.get("cached_tokens"),
           "finish_reason": body["choices"][0].get("finish_reason"),
           "wall_s": round(wall, 2), "text": text, "reasoning": reasoning}
    results.append(rec)
    print(f"  turn {i}: prompt={rec['prompt_tokens']:>7} out={rec['completion_tokens']:>5} "
          f"wall={rec['wall_s']:>6}s :: c={text[:40]!r} r={len(reasoning)}ch")
    messages.append({"role": "assistant", "content": text})
    if i + 1 < TURNS:
        messages.append({"role": "user", "content": QUESTIONS[(i + 1) % len(QUESTIONS)]})

json.dump({"base": BASE, "model": MODEL, "prefix_tokens": PREFIX_TOKENS,
           "turns": results}, open(OUT, "w"), indent=2)
print(f"wrote {OUT}  total_wall={sum(r['wall_s'] for r in results):.1f}s")
