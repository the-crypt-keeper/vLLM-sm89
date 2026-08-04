# DeepSeek-V4-Flash on Ada (sm89) — validated, not just running

This fork serves
**[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)**
— the GA checkpoint, 304B total / ~12.7B activated — on **NVIDIA L40S / Ada
(sm_89)** using its **stock MXFP4 experts through Marlin**: no exotic
quantization, no 2-bit codebooks, no hand-written SASS. Base is official vLLM
**v0.25.1** at commit `752a3a504` plus generated per-file patches from `patches/`.

The distinguishing claim is not that it runs. It is that the output has been
**measured against the DeepSeek cloud API and found statistically
indistinguishable** — after five correctness bugs that were invisible to every
self-test in the stack were found and fixed.

## Benchmarks

8x L40S TP8, [`DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), 16K context, fp8 KV, MXFP4 experts
via Marlin, **no speculative decoding**, prefix caching off, and
`VLLM_DSV4_DETERMINISTIC_MOE=1` — i.e. the ~5.3% determinism cost is *included*,
not benchmarked around. `vllm bench serve`, random dataset, `--ignore-eos`,
measured 2026-08-02. Every run completed with zero failures.

**Decode** — 1K in / 2K out:

| concurrency | output tok/s | per stream | median TTFT | median TPOT |
|---:|---:|---:|---:|---:|
| 1 | **54.6** | 54.6 | 324 ms | 18.2 ms |
| 2 | **105.6** | 52.8 | 498 ms | 18.7 ms |
| 4 | **188.6** | 47.2 | 1,060 ms | 20.7 ms |
| 16 | **517.4** | 32.3 | 1,232 ms | 30.1 ms |
| 64 | **1,067.1** | 16.7 | 1,582 ms | 58.5 ms |
| 128 | **1,500.3** | 11.7 | 3,017 ms | 82.6 ms |

**Prefill** — 8K in / 4K out:

| concurrency | prefill tok/s | total tok/s | output tok/s | median TTFT |
|---:|---:|---:|---:|---:|
| 1 | **3,499** | 152.1 | 50.7 | 2,341 ms |
| 2 | **5,175** | 289.0 | 96.3 | 3,166 ms |
| 4 | **8,566** | 506.7 | 168.9 | 3,825 ms |

Decode scales close to linearly to 4 streams (54.6 → 188.6 tok/s) and keeps
climbing to **1,500 tok/s at 128 concurrent** — 27x the single-stream rate.
Prefill is `concurrency x 8192 / median TTFT`; single-stream **3.5K tok/s**,
**8.6K tok/s** at 4 concurrent. Decode here is bandwidth-bound: L40S is GDDR6
at 864 GB/s, and TP8 aggregate bandwidth is what carries the batch.

Raw `vllm bench serve` JSON and the sweep script are committed under
[`docs/benchmarks/l40s-tp8-2026-08-02/`](docs/benchmarks/l40s-tp8-2026-08-02/).

### Second sparse-MLA backend, and speculative decoding (2026-08-04)

Two optional levers landed after the numbers above. Both default **off**, so the
table above is still what you get out of the box.

**`BACKEND=flashinfer`** routes sparse MLA through
[yhfgyyf/vllm-deepseek-v4-sm89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89)'s
FlashInfer build instead of vLLM's Triton port. There is no native sm89 sparse-MLA
kernel anywhere; what that project did is implement the Ada path *inside*
FlashInfer's SM120 kernel, substituting an available primitive for every
SM90+-only one — unscaled FP8 MMA plus a software UE8M0 rescale (exact, since
block scaling is one scale per k=32 block), `cp.async` for TMA,
`mbarrier.test_wait.parity` for the SM90 barrier.

**`SPEC=dspark SPEC_TOKENS=3`** enables the DSpark drafter that ships *inside* the
GA 0731 checkpoint (`mtp.0/1/2`, `markov_head`, `confidence_head`) — no separate
download. `SPEC_TOKENS=7`, the model card's figure, is far too deep here.

`MTP` is the third option — the classic 1-layer DeepSeek head, which only the
**preview** checkpoint ships. Extract it with `extract_mtp_head.py` and point
`SPEC_MODEL` at it. Cross-revision drafting is a *speed* risk only: rejection
sampling preserves the target distribution regardless of draft quality.

512 in / 2048 out, `--ignore-eos`, output tok/s (mean TPOT in parens):

| conc | Triton, no spec | Triton + DSpark k=3 | FlashInfer, no spec | FI + DSpark k=3 | FI + MTP k=1 | FI + MTP k=2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 52.96 (18.7 ms) | 69.46 (14.2) | 65.59 (15.1) | **118.47 (8.3)** | 99.84 (9.8) | 86.80 (11.3) |
| 4 | 163.62 (24.2) | 190.88 (19.0) | 197.28 (20.0) | **215.45 (17.2)** | 204.12 (16.8) | 170.49 (22.1) |
| 8 | 258.81 (30.5) | 210.24 (34.8) | **295.24 (26.7)** | 252.22 (26.9) | 290.37 (24.3) | 221.78 (31.8) |

Backend and speculation are **synergistic**: at c1 DSpark is worth +31% on the
Triton port but +81% on FlashInfer, beating the product of the independent gains.
Each decode step carries `k+1` query tokens, and that multi-query shape is where
the Triton port is weakest. Best single-stream config is **2.24x** the default.

**Pick your `k` by measuring, not by the model card.** `SPEC_TOKENS` is a
throughput knob with a sharp optimum, and the card's DSpark figure of 7 is far
past it here:

| method | k | draft acceptance | mean accept length | verdict |
|---|---:|---:|---:|---|
| DSpark | 7 | 12–17% | 1.84–2.19 of 8 | far too deep |
| DSpark | **3** | 66–73% | 2.78–3.18 of 4 | best peak throughput |
| MTP | **1** | 25.8% | 1.26 | best at concurrency |
| MTP | 2 | 15.6% | 1.31 | dominated by k=1 everywhere |

The k=7 "15% acceptance" is a denominator artifact, not a broken drafter —
accepting ~1.2 of 7 reads as 16% while accepting ~2 of 3 reads as 67%.

**DSpark wins on peak, MTP k=1 wins on flatness.** DSpark k=3 is the fastest
single-stream option (+81%) but pays 15% at c8; MTP k=1 gives +52% single-stream
and is within 2% of no-spec at c8, so it degrades far more gracefully. If one
server has to cover both interactive and batch traffic, MTP k=1 is the safer
default. Speculation crosses over between **c4 and c8 on both backends**, so that
boundary is a property of the workload, not the kernel.

**The backend gain collapses at high concurrency.** Batch-serving config — 16K
context, `NUM_SEQS=128`, prefix caching off, no speculation, 1K in / 2K out:

| conc | backend | output tok/s | median TPOT | median TTFT |
|---:|---|---:|---:|---:|
| 64 | Triton | 1,080.56 | 57.95 ms | **1,512.74 ms** |
| 64 | FlashInfer | **1,112.41** (+2.9%) | **56.36 ms** | 1,608.63 ms |
| 128 | Triton | 1,498.53 | 82.47 ms | **3,039.79 ms** |
| 128 | FlashInfer | **1,533.65** (+2.3%) | **80.72 ms** | 3,136.91 ms |

+14% at c1 becomes +2.9% at c64 and +2.3% at c128 — once decode is bandwidth
bound, FP8 MMA efficiency stops paying. TTFT is consistently *worse* on
FlashInfer at these batch sizes. For calibration, the Triton figures reproduce
the 2026-08-02 run above to 1.3% and 0.1%, so run-to-run noise is around 1% and
the 2-3% gap is real but marginal.

**So: use FlashInfer for interactive/low-concurrency serving, and the Triton port
for batch evals** — where the gain is inside the noise floor and the Triton path
is the one with the full correctness record behind it.

> **Graph the spec batch shape or every number is wrong.** With speculation the
> captured decode shape is `max_num_seqs x (SPEC_TOKENS+1)`, not `max_num_seqs`.
> A ladder of `1,2,4,8` against `NUM_SEQS=8 SPEC_TOKENS=3` (shape 32) made DSpark
> measure as a **3.2x loss**; the correct ladder recovered 2.7x of that. vLLM's
> own default will not save you — it caps at `min(max_num_seqs*2, 512)`, and the
> `x2` assumes a 1-token draft. `serve_l40s_ds4_tp8.sh` now derives this.

Install steps for both optional paths are under
[Quickstart](#optional-the-flashinfer-sm89-backend).

Caveats worth stating plainly: that sm89 fallback was **wrong upstream until
2026-08-02**, so it is young code; it is emulation, so FP8 accumulate range
differs from true block-scaled hardware; and its own `__CUDA_ARCH__ < 900` gate
also covers sm80, which we cannot test. The Triton port remains the default and
the validated path.

---

## Validation

Ada had no working DSV4 sparse-MLA kernel, so the port routes attention through
vLLM's Triton port. That path was *nearly* right, which is the dangerous kind of
wrong: the model was coherent, benchmarks looked plausible, and the defects only
showed up as a slightly elevated rate of runaway generations.

Head-to-head against the DeepSeek cloud API, matched prompts and sampler
([ReasonScape](https://github.com/the-crypt-keeper/reasonscape) `dates`):

| arm | n | invalid | trunc | score |
|---|---:|---:|---:|---|
| this fork, 8x L40S TP8 | 1341 | 0 | 0.0022 | **0.972 ± 0.009** |
| DeepSeek cloud API | 1336 | 0 | 0.0060 | 0.975 ± 0.008 |

Difference **−0.003 ± 0.012** (z = 0.49). Prompt tokens were **149.2 on both
arms** — the encoder renders byte-identical prompts to cloud.

Full local sweep, **26,901 prompts**, invalid rate 0.0008:

| base task | n | trunc | score | completion |
|---|---:|---:|---|---:|
| **all** | 26901 | 0.0124 | **0.954 ± 0.003** | 1691.6 |
| tables | 3456 | 0 | 0.990 ± 0.003 | 463.2 |
| arithmetic | 2078 | 0.0048 | 0.979 ± 0.006 | 2598.6 |
| letters | 1727 | 0.0006 | 0.979 ± 0.007 | 1857.7 |
| dates | 1338 | 0.0045 | 0.976 ± 0.008 | 645.2 |
| shuffle | 3356 | 0.0012 | 0.976 ± 0.006 | 1188.0 |
| objects | 2301 | 0.0013 | 0.965 ± 0.007 | 722.4 |
| boolean | 2342 | 0.0242 | 0.963 ± 0.011 | 3751.4 |
| cars | 2300 | 0.0017 | 0.945 ± 0.011 | 890.9 |
| brackets | 2301 | 0.0013 | 0.942 ± 0.010 | 2287.1 |
| sort | 2014 | 0.0010 | 0.940 ± 0.010 | 1172.8 |
| shapes | 1710 | 0.0372 | 0.917 ± 0.013 | 1349.4 |
| sequence | 1978 | 0.0843 | 0.837 ± 0.016 | 4053.5 |

`sequence` is the model's weak task, not a port defect: cloud scores
**0.837 ± 0.016** on it too (n=1963), with *higher* truncation (9.1% vs 8.4%)
and longer completions. Identical to three decimals.

Evaluation is [ReasonScape](https://github.com/the-crypt-keeper/reasonscape). Method and the full investigation record: [`docs/l40s-8x-tp8-investigation.md`](docs/l40s-8x-tp8-investigation.md).

**What this does not cover.** These prompts average 366 tokens. Nothing in the
suite reaches the L=2048 boundary where two of the four bugs lived, so the
**long-context path is not yet validated** — a 2–14k prompt sweep is pending.
Treat long-context behaviour as untested rather than working.

---

## What was broken

Five correctness bugs, each found by comparing measured *rates* against a cloud
reference and then bisected with instrumentation. None of them threw an error;
none were caught by the existing self-tests.

**1 — decode top-k selection past context 2048.** DSV4's lightning indexer
selects `index_topk=512` candidates. The ratio-4 layers cross that threshold at
exactly L=2048 (compressed count L/4 = 512). The radix top-k selectors were
being handed an unclamped `k_select`; below 2048 selection is a no-op so nothing
showed, above it the model selected badly.

**2 — the indexer KV cache was read with the wrong byte layout.** The writer
stores each block **segregated** — `[block_size×D keys][block_size×4 scale
bytes]` — while the sm89 decode kernel read it **interleaved**, as `[D+4]` per
token. Every decode candidate score was garbage (−1e27 … 1e33, NaN) while
prefill over the same tokens scored 1.1–3.8. Below 2048 the top-k is a no-op, so
garbage scores cost nothing; past 2048 the model selected 512 essentially
arbitrary candidates. Fixing it moved truncations from 11/40 to 2/40 (p=0.0129)
and "reached 6500 tokens then finished" from 0.214 to 0.882 (p=0.00026),
landing on the cloud reference (p=0.428, null).

**Why the self-test passed:** the kernel, the torch reference and the reference
packer all shared the same wrong layout. The test compared three implementations
of the same mistake. It is now checked against the **real writer** with f32
ground truth computed pre-quantization — a test that can actually fail.

**3 — greedy decoding was not deterministic.** `moe_align_block_size` assigns
each token's slot within its expert with an `atomicAdd` return value, so the
bucketing is thread-arrival-ordered and varies run to run.

**4 — and that ordering changed the numbers.** It should not have: DeepSeek's
reference MoE has no bucketing at all, and per-row dot products are permutation
invariant. But Marlin schedules tiles as DP + two-tile stream-K, and
`slice_count` — how the fp32 partial sums are *grouped* before the bf16 store —
is derived from the global tile index, which is keyed on the block index. A row
that moves between blocks of the same expert gets a different, equally valid
summation grouping. Measured over 172 calls/rank: differing rows were **always**
a subset of rows that changed block (`diff_not_moved == 0`, every call), and
every difference was **≤ 1 bf16 ULP**.

One ULP is enough. Flipping the low mantissa bit of every 6th expert-output row
— for reasons unrelated to ordering — moves prompt logprobs by median 0.0935
nats, indistinguishable from what reordering does (0.0920). 43 layers plus a
discrete top-512 selection amplify one ULP into tenths of a nat.

Consequence: the ordering cannot be made inert without changing Marlin's
schedule, so **`VLLM_DSV4_DETERMINISTIC_MOE=1` is the fix**, not a workaround.
It is on by default in the launcher and costs ~5.3% decode.

**Bonus — `reasoning_effort` was shifted a rung.** vLLM shipped one constant
holding DeepSeek's **high** text, emitted only for `"max"`. So `high` was
silently identical to `low`, `max` delivered high, and the real `max` level was
unreachable from any request. The official three-level ladder is restored from
the encoder that ships with the checkpoint, verified byte-identical at
`low`/`high`/`max`.

**5 — the response said `reasoning`, every client read `reasoning_content`.**
Not an sm89 bug at all — this one hits anyone self-hosting V4-Flash agentically
on vLLM 0.25.1. vLLM renamed the response field to `reasoning` (the OpenAI
spelling), while DeepSeek's API and every client honouring their thinking-mode
contract read `reasoning_content`. The model's monologue was returned all along
under a name nothing looked for, so harnesses dropped it on every tool-carrying
turn and the model re-derived its state from scratch each step.

The request path was never broken — vLLM accepts either spelling inbound and
renders both back into the assistant turn. Only the two response serializers
omitted the alias, which is exactly why it was invisible: probing
`message["reasoning_content"]` and getting `None` is what a *healthy* server
looks like on this version. Dump the whole message object, not one key.

Measured on deep-swe, 15 tasks, seed 0, mini-swe-agent, identical sampler:

| | before | after | DeepSeek cloud |
|---|---:|---:|---:|
| PASS | 2/15 | **7/15** | 7/15 |
| terminated | 7/15 | 12/15 | 15/15 |
| "Tool call error" exceptions | 10 | 3 | 0 |

The per-step token profile converges on cloud's — 95,521 → 132,504 input tokens
per step (cloud 136,561) and 854 → 655 output (cloud 636). The broken arm was
*lighter* on input because it discarded the monologue, and *heavier* on output
because the model kept re-deriving what it had already worked out. The remaining
three failures are wall-clock timeouts, not crashes.

---

## Hardware

Validated on **8x L40S** (sm_89, 48 GB, ECC off) and **4x L40S** (ECC on).
Anything Ada with enough VRAM should work; the checkpoint is 156 GB on disk, so
plan ~40 GB/card at TP4 or ~21 GB/card at TP8.

**Don't read the disk size as a parameter count.** The routed experts are MXFP4
stored two values per byte (`I8` container, `[2048, 2048]` for a logically
`[2048, 4096]` matrix, with an `F8_E8M0` scale per 32 elements), so bytes are
roughly *half* the parameters. Counted from the safetensors headers:

| | params |
|---|---:|
| main experts (43 layers x 257 experts) | 278.11 B |
| main attention / indexer / norms | 5.17 B |
| embeddings + lm_head | 1.06 B |
| **main model** | **284.33 B** |
| DSpark draft module (`mtp.*`, 19.40 B of it experts) | 19.85 B |
| **total** | **304.18 B** |

Which matches the model card's 304B. Per token, 6 of 256 routed experts plus the
shared one activate: **~12.7 B**. The DSpark draft module is loaded but unused
here — `SPEC=none` is the default, and speculative decoding is unverified on the
Marlin path.

| | 4x L40S (ECC on) | 8x L40S (ECC off) |
|---|---|---|
| weights resident | ~40 GB/card | ~21 GB/card |
| served context | 32K | 16K |
| KV pool at util 0.95 | 2.43 GiB / 158,769 tok | **22.8 GiB / 851,099 tok** |
| concurrency at served context | 4.85x | **52x** |

(The two columns were measured at different `--max-model-len`, so read the pool
sizes rather than the concurrency multiples. ECC accounts for ~3 GiB/card.)

Topology matters for TP8: this box is two PIX islands (0-3 \| 4-7) joined by SYS
across NUMA, so every all-reduce crosses the host bridge and custom all-reduce
stays off. Check yours with `nvidia-smi topo -m`.

Driver with CUDA 13.0 (tested 580.95.05). ECC off roughly doubles the KV pool;
that is a per-box integrity call.

## Quickstart

```bash
git clone https://github.com/the-crypt-keeper/vLLM-sm89 && cd vLLM-sm89
uv venv --python 3.12 venv
uv pip install --python venv/bin/python vllm==0.25.1     # the patch baseline

# FlashInfer pin swap (mandatory: 0.6.13 lacks swa_topk_lens)
uv pip uninstall --python venv/bin/python flashinfer-cubin
uv pip install --python venv/bin/python flashinfer-python==0.6.14
uv pip install --python venv/bin/python --index-url https://flashinfer.ai/whl/cu130 \
  "flashinfer-jit-cache==0.6.14+cu130"

# apply the patches FROM THE REPO ROOT (see the gotcha below)
git apply --directory=venv/lib/python3.12/site-packages --verbose patches/*.patch

hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir /path/to/model
MODEL=/path/to/model NUM_SEQS=128 ./serve_l40s_ds4_tp8.sh
```

The launcher defaults to `TP=8`; on a four-card box use `TP=4 MAXLEN=32768`.
Every knob is an env override — `MODEL PORT TP MAXLEN UTIL NUM_SEQS
BATCHED_TOKENS KV_DTYPE BLOCK_SIZE PREFIX_CACHING BACKEND SPEC SPEC_TOKENS
SPEC_MODEL` — and the cudagraph capture ladder is derived from `NUM_SEQS` **and
the speculation depth**, so it always reaches your real decode batch shape.

### Optional: the FlashInfer sm89 backend

Only needed for `BACKEND=flashinfer`. It JIT-compiles at boot, so unlike the
Triton path it needs a compiler present:

```bash
# from https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases
uv pip install --python venv/bin/python flashinfer_python-0.6.14+sm89.1-py3-none-any.whl
sudo apt install ninja-build      # or: uv pip install --python venv/bin/python ninja
BACKEND=flashinfer ./serve_l40s_ds4_tp8.sh
```

Three things that will bite you, all handled by the launcher:

1. `flashinfer-jit-cache` refuses to pair with a differently-versioned
   `flashinfer-python`, so `FLASHINFER_DISABLE_VERSION_CHECK=1` is required. Keep
   the jit-cache — every *other* kernel still comes from it.
2. That jit-cache ships a prebuilt **sm120-only** `sparse_mla_sm120.so` which
   would shadow the JIT build. `FLASHINFER_SPARSE_MLA_FORCE_SM89_PRIMS=1` renames
   the module to `sparse_mla_sm120_sm89_prims` and sidesteps it.
3. `ninja` and `nvcc` must be on `PATH` (`CUDA_HOME` too). The launcher fails
   fast if either is missing rather than dying mid-boot.

Rollback is `uv pip install --python venv/bin/python flashinfer-python==0.6.14`.
Note the venv has no `pip` — always go through `uv pip --python venv/bin/python`,
and keep the wheel's full versioned filename or `uv` rejects it.

### Optional: the MTP draft head

```bash
python3 extract_mtp_head.py    # needs the PREVIEW checkpoint; symlinks shard 46
SPEC=mtp SPEC_TOKENS=1 SPEC_MODEL=/path/to/DeepSeek-V4-Flash-MTP ./serve_l40s_ds4_tp8.sh
```

The 0731 GA checkpoint ships a 3-layer DSpark head (`mtp.0/1/2` +
`confidence_head`), which vLLM's `deepseek_mtp` method cannot load — it wants the
clean 1-layer head that only the preview checkpoint has. Both configs claim
`num_nextn_predict_layers: 1`, so GA's config understates its own head; that
mismatch is the `KeyError` you will otherwise hit. DSpark needs none of this — its
weights are already inside the GA checkpoint.

**Patch gotcha:** if site-packages sits inside a git work tree (a venv at the
repo root does), running `git apply` *from* site-packages silently skips every
patch — "Skipped patch", exit 0, and `--check` passes vacuously. Apply from the
repo root with `--directory`, and confirm the count:

```bash
ls patches/*.patch | wc -l    # expected count
git apply --directory=venv/lib/python3.12/site-packages patches/*.patch 2>&1 | grep -c "^Applied"
```

Boot checklist — grep the log for all four:

1. `Using MarlinExperts` (not TRTLLM, not DeepGEMM)
2. `DeepSeek V4 o_proj: using native SM89 block-scaled FP8 grouped matmul`
3. `DSv4 sparse-MLA Triton port self-test` passes (or, with `BACKEND=flashinfer`,
   `routing decode AND prefill to FlashInfer`)
4. `Available KV cache memory:` is **positive**

Full runbooks: [`docs/l40s-4x-runbook.md`](docs/l40s-4x-runbook.md) (4x box) and
[`docs/l40s-8x-tp8-investigation.md`](docs/l40s-8x-tp8-investigation.md) (8x box
+ the bug hunt).

## Knobs worth knowing

| env | default | what |
|---|---|---|
| `VLLM_DSV4_DETERMINISTIC_MOE` | `1` (launcher) | pins the MoE bucketing order. `=2` is the order-invariance regression gate, `=0` is stock nondeterministic behaviour. |
| `VLLM_DSV4_DETERMINISTIC_TOPK` | `0` | canonicalises the top-k index order (bug #5). Off by default because the throughput cost is unmeasured; without it, first-token argmax can flip run-to-run above ~2048 context. |
| `VLLM_MOE_W2` | `0` | must stay 0. `=1` reroutes experts to the inherited 2-bit path, which is not what this fork validates. The launcher refuses to start if it leaks in. |
| `VLLM_DSV4_SPARSE_MLA_SELFTEST` | `1` | keep on. |
| `VLLM_DSV4_SPARSE_MLA_FORCE_TRITON` | `0` | forces the Triton port on any arch, including ones FlashInfer covers. A/B lever. |
| `VLLM_DSV4_SPARSE_MLA_FORCE_FLASHINFER` | `0` | the mirror image — routes to FlashInfer on an arch the resolver would otherwise refuse. Honoured **only** if FlashInfer's own resolver claims the device, and raises otherwise; it never silently falls back, because that would make an A/B measure the kernel you did not select. Set by `BACKEND=flashinfer`. |
| `VLLM_DSV4_MARLIN_ORDER_CHECK` | off | diagnostics, see the investigation doc. |

Launcher-level:

| env | default | what |
|---|---|---|
| `BACKEND` | `triton` | `triton` \| `flashinfer`. Selects the sparse-MLA implementation and exports the three env vars the FlashInfer path needs. |
| `SPEC` | `none` | `none` \| `dspark` \| `mtp`. |
| `SPEC_TOKENS` | `7` | draft depth. Measure it — see the table above; `3` for DSpark, `1` for MTP. |
| `SPEC_MODEL` | — | separate draft checkpoint, `mtp` only. |
| `CUDAGRAPH_SIZES` | derived | powers of two reaching `NUM_SEQS*(SPEC_TOKENS+1)`. Override only if you know why. |

## Scope — what this fork is not

The upstream lineage carries a large body of Blackwell/SM120 work: 2-bit expert
codebooks with FP4 recovery, hand-written SASS kernels, tiered NVMe expert
residency, and support for GLM-5.2 and Kimi-K2.7. **None of that is exercised,
tested, or supported here.** It is inherited, left in place, and out of scope.
The SM120 cubins never load on Ada. If you want that work, go to the upstream
repos below.

This fork supports exactly one thing: **DeepSeek-V4-Flash on sm_89 via the stock
MXFP4→Marlin path.** Prefix caching is off by default as the correctness
baseline. Speculative decoding (DSpark, MTP) now works and is measured above, but
stays off by default.

## Lineage

The credit splits cleanly, and two independent Ada efforts converged:

- **[kacper-daftcode/vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet)** —
  the original: 2-bit experts, FP4 delta, SM120 SASS toolchain. Blackwell only.
- **[iSevenDays/vLLM-Moet](https://github.com/iSevenDays/vLLM-Moet)** — the Ada
  port, and the hard engineering here. FlashInfer's
  `trtllm_batch_decode_sparse_mla_dsv4` resolves its backend from device arch
  and refuses anything it has no kernel for, which on Ada means a hard failure
  at decode graph capture. That author wrote an **arch-portable Triton
  reimplementation of FlashInfer's SM120 sparse-MLA backend** — same call
  surface, semantics lifted from flashinfer 0.6.14 — plus a native SM89 FP8
  grouped matmul for the attention output projection. The port stalled at a
  validation wall, not an engineering one.
- **[yhfgyyf/vllm-deepseek-v4-sm89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89)** —
  an independent Ada lineage that solved the same kernel gap the other way. There
  is no native sm89 sparse-MLA kernel anywhere; that author implemented the Ada
  path *inside* FlashInfer's SM120 kernel, substituting an available primitive
  for every SM90+-only one — unscaled FP8 MMA plus a software UE8M0 rescale
  (exact, since block scaling is one scale per k=32 block), `cp.async` for TMA,
  `mbarrier.test_wait.parity` for the SM90 barrier. Also ports the DSv4 aux ops
  to CuTe DSL and carries an SM80 build. Optional here via `BACKEND=flashinfer`;
  we keep our own Triton indexer, since that is the code path we debugged.
- **[guqiong96/Lvllmds4-x](https://github.com/guqiong96/Lvllmds4-x)** — forks the
  above and adds `lk_moe`, a CPU-GPU hybrid MoE engine doing NUMA-aware expert
  compute in system RAM. Aimed at boxes that cannot hold the weights in VRAM;
  not used here, where all 156 GB is resident at 21 GB/card. Noted for people
  arriving with less VRAM than this box has.
- **this fork** — broke through the validation wall: five correctness fixes, the
  reasoning-effort ladder, the `reasoning_content` response alias, deterministic
  MoE, the runtime backend switch, and the measurement to prove all of it.

## Repository layout

- **`overlay/vllm/`** — complete modified vLLM files. **Edit these.**
- **`patches/`** — generated per-file patches. Never hand-edit; regenerate with
  `python3 tools/gen_patches.py` and verify with `--verify`.
- **`vllm/`** — read-only v0.25.1 baseline at `752a3a504` (gitignored).
- **`serve_l40s_ds4_tp8.sh`** — the TP8 launcher these results were measured with.
- **`docs/l40s-*.md`** — the runbooks and the investigation record.
- **`docs/ada-sm89-port.md`** — the inherited Ada port notes.
- **`kernels/`, `docker/`, `bench/`** — inherited from upstream; SM120-oriented
  and not exercised by this fork.
