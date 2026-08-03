"""Diff the sparse-MLA input/output fingerprints across two identical requests.

Answers, per call: were ALL inputs identical and the output different (kernel
nondeterministic in-engine), or did some input differ -- and which one?

usage: io_diff.py <mlaio.rankN.jsonl> [calls_per_request]
"""
import json, sys, collections

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"{len(recs)} calls traced")

# split into runs: the call index restarts nothing, so segment by equal halves
# after dropping any warmup. Identify request boundaries by the q shape pattern.
shapes = [tuple(r["q"]["shape"]) for r in recs]
print("q shapes in order:", collections.Counter(shapes).most_common(6))

N = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if not N:
    # a request = one contiguous block of calls; assume the last 2 blocks are
    # the two identical requests and each block has the same length
    N = len(recs) // 2
A, B = recs[-2 * N:-N], recs[-N:]
print(f"comparing last two blocks of {N} calls each\n")

FIELDS = ["q", "idx_main", "lens_main", "idx_extra", "lens_extra", "sinks",
          "kv_main_rows", "kv_extra_rows"]

first = None
counts = collections.Counter()
for i, (a, b) in enumerate(zip(A, B)):
    if tuple(a["q"]["shape"]) != tuple(b["q"]["shape"]):
        continue
    diff_in = [f for f in FIELDS if a.get(f) != b.get(f)]
    out_diff = a["out"] != b["out"]
    if out_diff:
        counts["out differs"] += 1
        key = ",".join(diff_in) if diff_in else "(ALL INPUTS IDENTICAL)"
        counts[key] += 1
        if first is None:
            first = (i, a, b, diff_in)

print("across calls whose output differed, which inputs also differed:")
for k, n in counts.most_common(12):
    print(f"  {n:>4}  {k}")

if first:
    i, a, b, diff_in = first
    print(f"\n=== FIRST call with a differing output: call block-index {i} ===")
    print(f"  q shape {a['q']['shape']}")
    print(f"  inputs differing: {diff_in or 'NONE -- kernel nondeterministic in-engine'}")
    for f in FIELDS + ["out"]:
        fa, fb = a.get(f), b.get(f)
        mark = "  <-- DIFFERS" if fa != fb else ""
        print(f"    {f:<14} A={fa}")
        if fa != fb:
            print(f"    {'':<14} B={fb}{mark}")
else:
    print("\nno call had a differing output in these two blocks.")
