"""Diff two forward traces in execution order.

The culprit is the FIRST module whose INPUT fingerprint matches across the two
runs but whose OUTPUT fingerprint does not. Everything downstream of it differs
for free and is not evidence.

usage: fwd_diff.py <fwd.rankN.jsonl> [n_requests]
"""
import json, sys, collections

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
NREQ = int(sys.argv[2]) if len(sys.argv) > 2 else 2
print(f"{len(recs)} records total")

# split into equal halves by record count; verify the module sequences align
half = len(recs) // NREQ
runs = [recs[i * half:(i + 1) * half] for i in range(NREQ)]
a, b = runs[0], runs[1]
mism = sum(1 for x, y in zip(a, b) if x["mod"] != y["mod"])
print(f"run A {len(a)} records, run B {len(b)}; module-sequence mismatches: {mism}")
if mism:
    sys.exit("sequences do not align -- were exactly 2 identical requests sent?")

def fp_eq(p, q):
    if p is None or q is None:
        return p == q
    return (p["shape"] == q["shape"] and p["sum"] == q["sum"]
            and p["abssum"] == q["abssum"] and p["absmax"] == q["absmax"])

first = None
in_same_out_diff = []
for idx, (x, y) in enumerate(zip(a, b)):
    i_same = fp_eq(x["in"], y["in"])
    o_same = fp_eq(x["out"], y["out"])
    if i_same and not o_same:
        in_same_out_diff.append((idx, x, y))
        if first is None:
            first = (idx, x, y)

print(f"\nmodules with INPUT identical but OUTPUT differing: {len(in_same_out_diff)}")

if first:
    idx, x, y = first
    print(f"\n=== FIRST such module (execution position {idx}) ===")
    print(f"  module : {x['mod']}")
    print(f"  class  : {x['cls']}")
    print(f"  in     : {x['in']}")
    print(f"  out A  : {x['out']}")
    print(f"  out B  : {y['out']}")
    if x["out"] and y["out"]:
        d = abs(x["out"]["sum"] - y["out"]["sum"])
        print(f"  |dsum| : {d:.6e}")

    print("\n  next few, by execution order:")
    for idx2, x2, y2 in in_same_out_diff[1:8]:
        print(f"    [{idx2}] {x2['cls']:<28} {x2['mod']}")

    print("\n  culprit classes, ranked by how often they inject a divergence:")
    for cls, n in collections.Counter(
            x2["cls"] for _, x2, _ in in_same_out_diff).most_common(8):
        print(f"    {n:>5}  {cls}")
else:
    # everything that differs at all, to show where the first difference is
    diff = [(i, x) for i, (x, y) in enumerate(zip(a, b)) if not fp_eq(x["out"], y["out"])]
    print("no module had identical input with differing output.")
    if diff:
        i, x = diff[0]
        print(f"first differing output at all: [{i}] {x['cls']} {x['mod']}")
        print("  -> its input already differed, so look earlier / widen the trace")
