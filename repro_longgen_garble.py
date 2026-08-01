#!/usr/bin/env python3
"""Standalone repro: DeepSeek-V4-Flash output collapses into token salad once a
generation runs past roughly 1-2K tokens on the sm89 vLLM-Moet stack.

Observed 2026-08-01 on 8x L40S (TP8, vLLM 0.25.1 + iSevenDays/vLLM-Moet @5949a1c,
MXFP4->Marlin experts, fp8_ds_mla KV, Triton sparse-MLA port). The same prompts
through DeepSeek's cloud API produce clean output in 0/1344 samples; locally
4.2% of an eval run garbles, rising to 67.6% of samples that reach 2-4K tokens.

The failure is NOT truncation: finish_reason is "stop", is_truncated is false,
and </think> is emitted. Generation simply degenerates mid-stream into
multilingual token salad after a coherent opening.

usage:
  python3 repro_longgen_garble.py [--base URL] [--model NAME] [--n 8]

exit 1 if any sample garbles (i.e. the bug reproduced).
"""
import argparse, json, re, sys, urllib.request

# Prompts chosen because they force long chains-of-thought (leap-year edge cases
# and multi-step countdowns). From the `dates` eval, scenarios birthday_leap_year
# and consumable_countdown, which had the highest local-vs-cloud failure rates.
PROMPTS = [
    "Emma was born on the last day of February in 2000. Today is her 23-year-old "
    "birthday. What is today's date in MM/DD/YYYY?",
    "Peter was born on the last day of February in 1996. Today is her 17-year-old "
    "birthday. What is the date 3 days from today in MM/DD/YYYY?",
    "Emma was born on the last day of February in 1992. Today is her 22-year-old "
    "birthday. What is the date tomorrow in MM/DD/YYYY?",
    "Today is the palindrome day of 2010, because the MMDDYYYY format of the date "
    "is the same backwards as forwards. What is the date a month ago from today "
    "in MM/DD/YYYY?",
]
PREAMBLE = ("Given a small set of sentences about a particular date, answer the "
            "provided question. Respond only with the final date in MM/DD/YYYY "
            "format.\n")

# CJK / Cyrillic / Thai / Devanagari / Hangul / kana. An English-language date
# task has no business emitting these; >=5 of them means the stream collapsed.
FOREIGN = re.compile(r'[぀-ヿ一-鿿가-힯'
                     r'Ѐ-ӿ֐-׿؀-ۿ'
                     r'฀-๿ऀ-ॿ]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--n", type=int, default=8, help="samples per prompt")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--show", action="store_true", help="print a garbled tail")
    ap.add_argument("--pad", type=int, default=0,
                    help="prepend N filler sentences so the prompt already "
                         "exceeds 2048 tokens (the mitigation: the 2048 "
                         "boundary is then crossed in prefill, not decode)")
    a = ap.parse_args()

    pad_text = "".join(
        "Context note {}: this sentence is padding and carries no "
        "information relevant to the question.\n".format(i)
        for i in range(a.pad))
    total = bad = 0
    worst = None
    for p in PROMPTS:
        for s in range(a.n):
            body = {
                "model": a.model,
                "messages": [{"role": "user", "content": pad_text + PREAMBLE + p}],
                "max_tokens": a.max_tokens,
                "temperature": 1.0, "top_p": 0.95,
                "chat_template_kwargs": {"thinking": True,
                                         "reasoning_effort": "max"},
            }
            req = urllib.request.Request(
                f"{a.base}/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            ch = d["choices"][0]
            msg = ch["message"]
            text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
            ntok = d["usage"]["completion_tokens"]
            hits = len(FOREIGN.findall(text))
            garbled = hits >= 5
            total += 1
            bad += garbled
            if garbled and worst is None:
                worst = text
            print(f"  {p[:38]:38} #{s} tokens={ntok:5} finish={ch['finish_reason']:6} "
                  f"foreign_chars={hits:4} {'<-- GARBLED' if garbled else ''}",
                  flush=True)

    print(f"\ngarbled {bad}/{total} samples ({100*bad/total:.1f}%)")
    if worst and a.show:
        print("\n--- tail of a garbled sample ---")
        print(worst[-600:])
    if bad:
        print("\nREPRODUCED: generation degenerates past ~1-2K tokens.")
        return 1
    print("\nnot reproduced in this run (try --n larger; rate is ~5% overall "
          "but ~68% among samples reaching 2-4K tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
