"""Compare two pc_ab.py runs: does prefix caching change what the model says?

usage: pc_cmp.py <a.json> <b.json>
"""
import json, sys, difflib

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
A, B = a["turns"], b["turns"]

print(f"A = {a['model']}   B = {b['model']}   prefix={a['prefix_tokens']} tok\n")

hdr = f"{'turn':>4} | {'prompt':>7} | {'cached A':>9} | {'cached B':>9} | {'wall A':>7} | {'wall B':>7} | text"
print(hdr); print("-" * len(hdr))
same = 0
for x, y in zip(A, B):
    ok = x["text"] == y["text"] and x.get("reasoning", "") == y.get("reasoning", "")
    same += ok
    print(f"{x['turn']:>4} | {x['prompt_tokens']:>7} | {str(x['cached_tokens']):>9} | "
          f"{str(y['cached_tokens']):>9} | {x['wall_s']:>7} | {y['wall_s']:>7} | "
          f"{'IDENTICAL' if ok else 'DIFFERS'}")

print(f"\nidentical turns: {same}/{len(A)}")
wa, wb = sum(t["wall_s"] for t in A), sum(t["wall_s"] for t in B)
print(f"total wall:  A {wa:.1f}s   B {wb:.1f}s   speedup {wb/wa:.2f}x")
ca = sum(t["cached_tokens"] or 0 for t in A)
cb = sum(t["cached_tokens"] or 0 for t in B)
pa = sum(t["prompt_tokens"] for t in A)
print(f"cached tok:  A {ca:,} / {pa:,} ({100*ca/pa:.1f}%)   B {cb:,} / {pa:,} ({100*cb/pa:.1f}%)")

for x, y in zip(A, B):
    if x["text"] != y["text"]:
        print(f"\n--- turn {x['turn']} content diff ---")
        for line in difflib.unified_diff(
                x["text"].splitlines(), y["text"].splitlines(),
                "A", "B", lineterm="", n=1):
            print(" ", line)
    if x.get("reasoning", "") != y.get("reasoning", ""):
        print(f"--- turn {x['turn']} reasoning differs "
              f"(A {len(x.get('reasoning',''))}ch, B {len(y.get('reasoning',''))}ch) ---")
