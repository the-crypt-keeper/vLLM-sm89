"""Split the oracle jsonl into per-run passes and diff them.

Two identical requests were sent. Each prefill emits one record per layer per
run. The question: across the two runs, do the LOGITS differ (upstream race) or
are the logits identical while the SELECTION differs (order-dependent
tie-breaking)?

usage: oracle_cmp.py <oracle.rankN.jsonl>
"""
import json, sys, collections

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
pre = [r for r in recs if r.get("phase") == "prefill"]
print(f"{len(recs)} records, {len(pre)} prefill\n")
if not pre:
    sys.exit("no prefill records -- was PREFILL=1 set and caching off?")

# group by (layer, seq_len); each identical request should contribute one
# record per layer, so group members are repeats of the same computation
groups = collections.defaultdict(list)
for r in pre:
    groups[(r["layer"], r["seq_len"])].append(r)

multi = {k: v for k, v in groups.items() if len(v) > 1}
print(f"{len(groups)} (layer, seq_len) groups, {len(multi)} seen more than once\n")
if not multi:
    print("groups seen once each -- sample:")
    for k, v in list(groups.items())[:5]:
        r = v[0]
        print(f"  {k[0]} seq={k[1]} max={r['finite_max']!r} ties={r['ties_at_thresh']} "
              f"sel={r['sel']}")
    sys.exit(0)

logit_diff = sel_diff = both_same = 0
ties_seen = 0
examples = []
for (layer, seq), v in sorted(multi.items()):
    mx = {r["finite_max"] for r in v}
    mn = {r["finite_min"] for r in v}
    sels = {json.dumps(r["sel"], sort_keys=True) for r in v}
    ties = max(r["ties_at_thresh"] for r in v)
    ties_seen += ties > 1
    lg_same = len(mx) == 1 and len(mn) == 1
    sl_same = len(sels) == 1
    if not lg_same:
        logit_diff += 1
        if len(examples) < 6:
            examples.append((layer, seq, sorted(mx), sorted(mn), ties, sels))
    elif not sl_same:
        sel_diff += 1
        if len(examples) < 6:
            examples.append((layer, seq, sorted(mx), sorted(mn), ties, sels))
    else:
        both_same += 1

n = len(multi)
print(f"logits DIFFER across repeats            : {logit_diff}/{n}")
print(f"logits identical but SELECTION differs  : {sel_diff}/{n}")
print(f"fully identical                         : {both_same}/{n}")
print(f"groups with a tie at the cut line       : {ties_seen}/{n}")

print("\nverdict:", end=" ")
if logit_diff:
    print("LOGITS are not reproducible -> upstream race, selection only amplifies it")
elif sel_diff:
    print("logits reproducible, selection varies -> order-dependent TIE-BREAKING")
else:
    print("everything reproducible at this layer/length -- widen MIN or repeats")

for layer, seq, mx, mn, ties, sels in examples:
    print(f"\n  {layer}  seq_len={seq}  ties_at_thresh={ties}")
    print(f"    finite_max: {mx}")
    print(f"    finite_min: {mn}")
    for s in sels:
        print(f"    sel: {s}")
