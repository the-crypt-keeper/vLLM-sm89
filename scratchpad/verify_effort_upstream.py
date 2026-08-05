"""Empirical re-verification of the DS4 reasoning-effort ladder after syncing to
upstream's #50580 + #48780.  Pure tokenizer, no GPU, no server."""

import sys

from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/media/4TBNVME/models/DeepSeek-V4-Flash-0731"
tok = DeepseekV4Tokenizer.from_pretrained(MODEL)
MSGS = [{"role": "user", "content": "Hello!"}]


def n(**kw):
    return len(tok.apply_chat_template(MSGS, tokenize=True, **kw))


base = n(thinking=True)
print(f"{'request':<12} {'tokens':>7} {'delta':>7}   tier")
print("-" * 44)
for eff in ["<omitted>", "none", "minimal", "low", "medium", "high", "xhigh", "max", "bogus"]:
    kw = {"thinking": True}
    if eff != "<omitted>":
        kw["reasoning_effort"] = eff
    try:
        t = n(**kw)
    except Exception as e:  # noqa: BLE001
        print(f"{eff:<12} {'RAISED':>7}         {type(e).__name__}: {e}")
        continue
    # identify which prompt landed by re-encoding each tier explicitly
    tier = next(
        (
            name
            for name in ("low", "high", "max")
            if t == n(thinking=True, reasoning_effort=name)
        ),
        "?",
    )
    print(f"{eff:<12} {t:>7} {t - base:>+7}   {tier}")

print()
print("thinking-mode default (the DEFAULT_CTK question):")
for kw in ({}, {"thinking": False}, {"enable_thinking": False}, {"thinking": True}):
    text = tok.apply_chat_template(MSGS, tokenize=False, **kw)
    closed = "</think>" in text
    print(f"  kwargs={str(kw):<28} pre-closed </think>: {closed}  -> "
          f"{'chat (thinking OFF)' if closed else 'thinking ON'}")
