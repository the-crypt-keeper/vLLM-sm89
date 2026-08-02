"""Element-wise diff of the selected-index vectors across two identical runs.

The whole question: same SET, different ORDER? The oracle's `sel` field cannot
answer it -- exact:true only means the set is right.

usage: sel_diff.py <oracle3.rankN.jsonl>
"""
import json, sys, collections

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
pre = [r for r in recs if r.get("phase") == "prefill" and "sel_idx" in r]
print(f"{len(recs)} records, {len(pre)} with raw sel_idx")
if not pre:
    sys.exit("no sel_idx -- was VLLM_DSV4_TOPK_ORACLE_RAW=1 set?")

g = collections.defaultdict(list)
for r in pre:
    g[(r["layer"], r["seq_len"])].append(r)
pairs = {k: v for k, v in g.items() if len(v) >= 2}
print(f"{len(pairs)} (layer, seq_len) groups seen at least twice\n")

same_all = set_same_order_diff = set_diff = 0
first_example = None
for (layer, seq), v in sorted(pairs.items()):
    a, b = v[0]["sel_idx"], v[1]["sel_idx"]
    if a == b:
        same_all += 1
        continue
    if sorted(a) == sorted(b):
        set_same_order_diff += 1
        if first_example is None:
            npos = sum(1 for x, y in zip(a, b) if x != y)
            first_example = (layer, seq, a, b, npos)
    else:
        set_diff += 1
        if first_example is None:
            first_example = (layer, seq, a, b, -1)

n = len(pairs)
print(f"identical vectors (same set, same order) : {same_all}/{n}")
print(f"SAME SET, DIFFERENT ORDER                : {set_same_order_diff}/{n}")
print(f"different set entirely                   : {set_diff}/{n}")

print("\nverdict:", end=" ")
if set_same_order_diff and not set_diff:
    print("CONFIRMED -- the selector emits the same candidates in a varying "
          "ORDER.\n          Attention accumulates in that order -> ~1 ULP -> "
          "the whole cascade.\n          Fix: canonicalise (sort) the indices "
          "before attention consumes them.")
elif set_diff:
    print("the SET itself varies -- not just ordering. Different bug than "
          "predicted.")
else:
    print("selection is fully identical -- ordering is NOT the source, look "
          "elsewhere.")

if first_example:
    layer, seq, a, b, npos = first_example
    print(f"\nexample: {layer}  seq_len={seq}")
    print(f"  positions differing: {npos}/{len(a)}")
    print(f"  A[:16] {a[:16]}")
    print(f"  B[:16] {b[:16]}")
    print(f"  sorted(A)==sorted(B): {sorted(a)==sorted(b)}")
    print(f"  set(A)==set(B)      : {set(a)==set(b)}   |A|={len(a)} |set|={len(set(a))}")
