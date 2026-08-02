"""Segment the forward trace into passes and diff the two matching requests.

Splits on model.embed_tokens (one firing per forward pass), identifies the two
runs by matching pass shapes, then finds the FIRST module whose INPUT
fingerprint matches across runs but whose OUTPUT does not.

usage: fwd_diff2.py <fwd.rankN.jsonl>
"""
import json, sys, collections

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
starts = [i for i, r in enumerate(recs) if r["mod"] == "model.embed_tokens"]
passes = [recs[s:e] for s, e in zip(starts, starts[1:] + [len(recs)])]
print(f"{len(recs)} records -> {len(passes)} forward passes")
for i, p in enumerate(passes):
    shp = p[0]["in"]["shape"] if p[0]["in"] else None
    print(f"  pass {i}: {len(p):>4} records, embed input shape {shp}")

# the two requests are the last 6 passes (3 prefill chunks each)
if len(passes) < 6:
    sys.exit("need at least 6 passes (2 requests x 3 chunks)")
runA, runB = passes[-6:-3], passes[-3:]

def fp_eq(p, q):
    if p is None or q is None:
        return p == q
    return (p["shape"] == q["shape"] and p["sum"] == q["sum"]
            and p["abssum"] == q["abssum"] and p["absmax"] == q["absmax"])

for ci, (pa, pb) in enumerate(zip(runA, runB)):
    if len(pa) != len(pb):
        print(f"\nchunk {ci}: length mismatch {len(pa)} vs {len(pb)}, skipping")
        continue
    if pa[0]["in"]["shape"] != pb[0]["in"]["shape"]:
        print(f"\nchunk {ci}: shape mismatch, skipping")
        continue
    culprits = [(i, x, y) for i, (x, y) in enumerate(zip(pa, pb))
                if x["mod"] == y["mod"]
                and fp_eq(x["in"], y["in"]) and not fp_eq(x["out"], y["out"])]
    anydiff = [(i, x) for i, (x, y) in enumerate(zip(pa, pb))
               if not fp_eq(x["out"], y["out"])]
    print(f"\n=== chunk {ci} (tokens {pa[0]['in']['shape']}) ===")
    print(f"  modules with differing output      : {len(anydiff)}")
    print(f"  of those, with IDENTICAL input     : {len(culprits)}")
    if not culprits:
        if anydiff:
            i, x = anydiff[0]
            print(f"  first differing output: [{i}] {x['cls']} {x['mod']}"
                  f"  (input already differed -> inherited)")
        else:
            print("  chunk is bit-identical across runs")
        continue
    i, x, y = culprits[0]
    print(f"  FIRST INJECTOR -> [{i}] {x['mod']}")
    print(f"    class : {x['cls']}")
    print(f"    in    : {x['in']}")
    print(f"    out A : {x['out']}")
    print(f"    out B : {y['out']}")
    print(f"    |dsum|: {abs(x['out']['sum'] - y['out']['sum']):.6e}")
    print("    next injectors:")
    for i2, x2, _ in culprits[1:6]:
        print(f"      [{i2}] {x2['cls']:<26} {x2['mod']}")
    print("    injector classes:")
    for cls, n in collections.Counter(x2["cls"] for _, x2, _ in culprits).most_common(6):
        print(f"      {n:>4}  {cls}")
