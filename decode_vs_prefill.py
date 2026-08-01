#!/usr/bin/env python3
"""Decode-path vs prefill-path self-consistency at depth.

Hypothesis under test: DS4's compressor keeps INCREMENTAL STATE during decode
(DeepSeek's reference registers kv_state / score_state buffers and updates them
one token at a time). Prefill recomputes the same quantity in one shot. If the
incremental decode path drifts, the two disagree, and the disagreement grows
with generated depth -- which matches the observed collapse profile (0% garble
below 1024 generated tokens, 67.6% at 2048-4096).

This needs no external oracle: it scores ONE identical token sequence two ways
on the SAME server.

  step 1  generate N tokens  -> per-token logprob from the DECODE path
  step 2  re-feed prompt+generation with prompt_logprobs
                             -> per-token logprob from the PREFILL path
  compare |decode_lp - prefill_lp| bucketed by depth

A flat profile exonerates the incremental decode path. A profile that grows
with depth localises the defect to it.

usage: decode_vs_prefill.py BASE MODEL OUT.json [--max-tokens 3000] [--temp 0]
"""
import argparse, json, urllib.request, statistics, sys


def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from {url}: {e.read()[:600].decode(errors='replace')}")


PROMPT = ("Given a small set of sentences about a particular date, answer the "
          "provided question. Respond only with the final date in MM/DD/YYYY "
          "format.\nEmma was born on the last day of February in 2000. Today is "
          "her 23-year-old birthday. What is today's date in MM/DD/YYYY?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--temp", type=float, default=1.0)
    a = ap.parse_args()

    # ---- step 1: generate, capturing DECODE-path logprobs -------------------
    gen = post(f"{a.base}/v1/chat/completions", {
        "model": a.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": a.max_tokens,
        "temperature": a.temp, "top_p": 0.95,
        "logprobs": True, "top_logprobs": 1,
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"},
    })
    ch = gen["choices"][0]
    msg = ch["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    steps = (ch.get("logprobs") or {}).get("content") or []
    decode_lp = [s["logprob"] for s in steps]
    print(f"generated {len(decode_lp)} tokens (finish={ch['finish_reason']}), "
          f"prompt_tokens={gen['usage']['prompt_tokens']}", flush=True)
    if not decode_lp:
        sys.exit("no decode logprobs returned -- server did not honour logprobs")

    # ---- step 2: re-score the SAME sequence through PREFILL -----------------
    # tokenize prompt-only and prompt+generation so we know the split point
    def tok(payload):
        return post(f"{a.base}/tokenize", payload)

    base_ids = tok({"model": a.model,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "chat_template_kwargs": {"thinking": True,
                                             "reasoning_effort": "max"}})["tokens"]
    full_ids = tok({"model": a.model, "prompt": text})["tokens"]
    ids = base_ids + full_ids
    print(f"prefill re-score over {len(ids)} tokens "
          f"({len(base_ids)} prompt + {len(full_ids)} generated)", flush=True)

    def rescore():
        pf = post(f"{a.base}/v1/completions", {
            "model": a.model, "prompt": ids,
            "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0,
        })
        plp = pf["choices"][0]["prompt_logprobs"]
        # prompt_logprobs[i] is a dict {token_id: {...}} for position i (None at 0)
        out = []
        for i in range(len(base_ids), len(ids)):
            e = plp[i]
            if not e:
                out.append(None); continue
            tid = str(ids[i])
            out.append(e[tid]["logprob"] if tid in e else None)
        return out

    prefill_lp = rescore()
    # CONTROL: identical request again. Any disagreement here is the stack's
    # own nondeterminism floor, not a decode-vs-prefill effect.
    prefill_lp2 = rescore()

    # ---- compare, bucketed by depth ----------------------------------------
    n = min(len(decode_lp), len(prefill_lp))
    rows = []
    for i in range(n):
        if prefill_lp[i] is None or prefill_lp2[i] is None:
            continue
        rows.append((i, decode_lp[i], prefill_lp[i],
                     abs(decode_lp[i] - prefill_lp[i]),
                     abs(prefill_lp[i] - prefill_lp2[i])))

    print(f"\ncomparable positions: {len(rows)}\n")
    buckets = [(0, 128), (128, 512), (512, 1024), (1024, 2048),
               (2048, 4096), (4096, 100000)]
    summary = []
    print(f"{'depth bucket':>16} {'n':>6} | {'DECODE vs PREFILL':>30} | "
          f"{'CONTROL prefill vs prefill':>30}")
    print(f"{'':>16} {'':>6} | {'median':>9} {'p90':>9} {'max':>9} | "
          f"{'median':>9} {'p90':>9} {'max':>9}")
    for lo, hi in buckets:
        sel = [r for r in rows if lo <= r[0] < hi]
        if not sel:
            continue
        def st(idx):
            d = sorted(r[idx] for r in sel)
            return statistics.median(d), d[int(0.9 * (len(d) - 1))], d[-1]
        dm, dp, dx = st(3)
        cm, cp, cx = st(4)
        summary.append({"lo": lo, "hi": hi, "n": len(sel),
                        "decode_vs_prefill": [dm, dp, dx],
                        "control": [cm, cp, cx]})
        print(f"{f'{lo}-{hi}':>16} {len(sel):>6} | {dm:>9.5f} {dp:>9.5f} {dx:>9.4f} | "
              f"{cm:>9.5f} {cp:>9.5f} {cx:>9.4f}")

    json.dump({"rows": rows, "summary": summary, "text_tail": text[-800:]},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
