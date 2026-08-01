#!/usr/bin/env python3
"""Locate the first NaN logprob in a long DS4 generation.

The non-streaming endpoint returns HTTP 400 ("Out of range float values are not
JSON compliant: nan") the moment ANY logprob in the response is NaN, which tells
us a NaN exists but not where. Streaming serialises per chunk, so we can count
tokens up to the first bad one and see the surrounding text.

A NaN logprob means the model's logits contained NaN/all -inf at that step ->
softmax degenerates -> sampling picks arbitrary tokens. That is a direct
mechanism for the observed mid-stream collapse into token salad.

usage: nan_hunt.py BASE MODEL [--trials 6] [--max-tokens 8192]
"""
import argparse, json, urllib.request, math, sys

PROMPT = ("Given a small set of sentences about a particular date, answer the "
          "provided question. Respond only with the final date in MM/DD/YYYY "
          "format.\nEmma was born on the last day of February in 2000. Today is "
          "her 23-year-old birthday. What is today's date in MM/DD/YYYY?")


def run(base, model, max_tokens, pad=0):
    prompt = PROMPT
    if pad:
        filler = ("Context note {i}: this sentence is padding and carries no "
                  "information relevant to the question.\n")
        prompt = "".join(filler.format(i=i) for i in range(pad)) + PROMPT
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0, "top_p": 0.95,
        "logprobs": True, "top_logprobs": 5,
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"},
        "stream": True,
    }
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    n = 0
    first_nan = None
    text = []
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)   # python json accepts bare NaN
                except Exception:
                    print(f"    unparseable chunk at token {n}: {payload[:200]}")
                    if first_nan is None:
                        first_nan = n
                    continue
                ch = d["choices"][0]
                lp = ch.get("logprobs") or {}
                for s in (lp.get("content") or []):
                    n += 1
                    v = s.get("logprob")
                    tops = [t.get("logprob") for t in (s.get("top_logprobs") or [])]
                    bad = (v is None or (isinstance(v, float) and math.isnan(v))
                           or any(isinstance(t, float) and math.isnan(t) for t in tops))
                    text.append(s.get("token", ""))
                    if bad and first_nan is None:
                        first_nan = n
                        print(f"    FIRST NaN at generated token #{n}: "
                              f"token={s.get('token')!r} logprob={v} tops={tops}")
                        ctx = "".join(text[max(0, n - 40):n])
                        print(f"    context: ...{ctx!r}")
    except Exception as e:
        print(f"    stream aborted after {n} tokens: {type(e).__name__} {e}")
    return n, first_nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--pad", type=int, default=0)
    a = ap.parse_args()

    hits = 0
    for t in range(a.trials):
        print(f"trial {t}:", flush=True)
        n, fn = run(a.base, a.model, a.max_tokens, a.pad)
        if fn is not None:
            hits += 1
            print(f"  -> {n} tokens, first NaN at {fn}", flush=True)
        else:
            print(f"  -> {n} tokens, clean", flush=True)
    print(f"\nNaN observed in {hits}/{a.trials} generations")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
