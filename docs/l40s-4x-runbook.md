# CLAUDE.md — DS4-Flash on 4x L40S via vLLM-Moet (NO 2-bit quant)

## Status: WORKS — field-tested 2026-07-23 on 4x L40S
First boot succeeded on the first attempt. All boot-checklist items green, zero
tracebacks, correct outputs, benchmarked. This doc is the full replication
runbook for a fresh machine. Copy `CLAUDE.md` + `serve_l40s_ds4.sh` into the
fresh clone (both are untracked; the serve command is also inlined below).

## Mission
Serve **DeepSeek-V4-Flash** on **4x NVIDIA L40S (sm89, 48 GB each, 192 GB total)**
using the **stock MXFP4→Marlin expert path** — deliberately NOT the repo's
headline 2-bit "W2" quantization. No Docker: venv + patched official wheel.

## Repo layering
1. `vllm-project/vllm` — real base. DS4 model code, sparse-MLA, MXFP4 backend
   selection all live upstream. Patch baseline: **v0.25.1, commit `752a3a504`**
   (recorded in `patches/SOURCE.txt`; matches the PyPI wheel exactly — all 70
   patches applied clean).
2. `kacper-daftcode/vLLM-Moet` — original Moet (SM120/Blackwell only: 2-bit W2,
   FP4 delta, SASS cubins). Nothing sm89 in it; NVFP4 content is this lineage.
3. `iSevenDays/vLLM-Moet` — the fork we use (https://github.com/iSevenDays/vLLM-Moet,
   branch `main`; field-tested at `5949a1c`). Adds the Ada/sm89 port:
   - `overlay/vllm/.../moe_w2_sm89.py` — Triton W2 emulation (unused by us)
   - `overlay/vllm/.../triton_ada_fp8_bmm.py` + `o_proj.py` — native SM89 FP8
     output projection (capability-gated, active regardless of W2 → we NEED this)
   - `overlay/vllm/vllm/v1/attention/ops/triton_sparse_mla_dsv4.py` — sparse-MLA
     Triton port (FlashInfer has no <SM100 kernel for DSV4 sparse MLA → NEED this)
   - `patches/` = 70 per-file diffs generated from `overlay/` (source of truth)

## Why the no-quant path works (all confirmed at runtime)
- DS4-Flash official checkpoint is **mixed**: dense stack = block-FP8 (ue8m0
  scales), routed experts = **MXFP4**. 149 GB total, 46 shards.
- `VLLM_MOE_W2` defaults to `"0"` (`moe_w2_cubit.py`, `is_w2_layer()` gate).
  Left unset, patched `mxfp4.py` falls through to stock
  `select_deepseek_v4_mxfp4_moe_backend` → FlashInfer-TRTLLM (sm100+, rejected)
  → DEEPGEMM_MXFP4 (sm90+, rejected) → **MarlinExperts** (weight-only dequant →
  BF16 tensor cores; no FP4 hardware needed). Observed: `Using MarlinExperts`.
- Memory: ~149 GB weights / TP4 ≈ 40 GB/card allocated, 43.5 GB/card serving.
- KV cost: the 584 B/token fp8_ds_mla packed row is PER LAYER; whole-stack cost
  is ~16.4 KB/token (43 layers, SWA layers + indexer average cheaper). With ECC
  on (46 GiB visible/card) and this config: pool = 2.43 GiB = **158,769 tokens**
  = 4.85x concurrency at 32K. ECC off (`nvidia-smi -e 0` + reset) would roughly
  double it (~3 GiB/card reclaimed; integrity tradeoff, decide per box).

## Fresh machine replication

### 0. Prerequisites
- 4x sm89 48 GB GPUs; NVIDIA driver with CUDA 13.0 (tested 580.95.05).
- Check topology: `nvidia-smi topo -m` — all-PIX (one PCIe switch) is what we
  validated; x16 links (idle cards show Gen1 — they retrain under load).
- ~170 GB disk for weights + ~10 GB venv; `uv` installed; internet.

### 1. Clone + venv + wheel
```bash
git clone https://github.com/iSevenDays/vLLM-Moet.git moet-fork && cd moet-fork
uv venv --python 3.12 venv
uv pip install --python venv/bin/python vllm==0.25.1   # official wheel = patch baseline
```

### 2. FlashInfer pin swap (MANDATORY)
The wheel ships flashinfer 0.6.13, which lacks `swa_topk_lens` — vLLM 0.25.1's
DSV4 sparse call sites pass it → decode-warmup TypeError. Engine also probes the
signature at init and refuses loudly.
```bash
uv pip uninstall --python venv/bin/python flashinfer-cubin
uv pip install --python venv/bin/python flashinfer-python==0.6.14
uv pip install --python venv/bin/python --index-url https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache==0.6.14+cu130"
```

### 3. Apply the 70 patches
GOTCHA (burned us 2026-07-23): if site-packages sits INSIDE a git work tree
(venv at the repo root does), `git apply` run from site-packages silently SKIPS
every patch — "Skipped patch 'vllm/...'", exit 0, and `--check` passes
vacuously. Apply from the repo root with `--directory`:
```bash
git apply --directory=venv/lib/python3.12/site-packages --verbose patches/*.patch 2>&1 | grep -cE "^Applied"   # expect 70
```
The tests/csrc/tools patches are new-file diffs — they create inert files next
to the wheel (csrc is Blackwell NVFP4-KV only, never compiled; harmless).

### 4. Verify before burning GPU time
```bash
venv/bin/python -c "from vllm.model_executor.layers.quantization.utils import moe_w2_sm89; print('gate OK')"
venv/bin/python -c "import vllm; print(vllm.__version__)"          # 0.25.1
venv/bin/python -c "import flashinfer; print(flashinfer.__version__)"  # 0.6.14
grep -rl swa_topk_lens venv/lib/python3.12/site-packages/flashinfer/mla/
ls venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py
```

### 5. Weights
```bash
hf download deepseek-ai/DeepSeek-V4-Flash --local-dir ~/models/DeepSeek-V4-Flash
# 149 GB, 46 shards. Done when: 46 *.safetensors present and
# .cache/huggingface/download/ has no *.incomplete files.
```

### 6. Serve
Use `serve_l40s_ds4.sh` (next to this file), or the full command below. Leave
`VLLM_MOE_W2` UNSET (the script refuses if =1). NO `--speculative-config`
(no MTP: unwanted + Marlin+MTP+DS4 unverified). Prefix caching OFF is the
correctness baseline; flip only after long-prompt answers verified identical.
Launch DETACHED (tmux/systemd/setsid) — a Claude-session-tied background
process dies when the session exits (learned 2026-08-01).
```bash
export VLLM_ENGINE_READY_TIMEOUT_S=1800   # first boot JITs all Triton kernels
mkdir -p ~/.cache/moet-l40s/triton ~/.cache/moet-l40s/torchinductor ~/.cache/torch/kernels
export TRITON_CACHE_DIR=~/.cache/moet-l40s/triton TORCHINDUCTOR_CACHE_DIR=~/.cache/moet-l40s/torchinductor
venv/bin/vllm serve ~/models/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash auto \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 1024 --max-num-seqs 4 \
  --no-scheduler-reserve-full-isl \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8]}' \
  --port 8001
```

### 7. Boot checklist — exact lines observed on the good boot
Cold boot ≈ 8.5 min (page-cache-warm weights; engine init 111 s of it).
1. `sm_89 routes to DeepseekV4FlashInferSM120Attention (fp8_ds_mla layout;
   decode+prefill via vLLM's Triton sparse-MLA port ...)` + `DeepSeek V4
   expert_dtype resolved to 'fp4'` + `Using DeepSeek's fp8_ds_mla KV cache format.`
2. `DSv4 sparse-MLA Triton port self-test on sm_89 ... worst_row_rel=7.634e-03`
   (expected ~1e-2 for the bf16 path; printed by all 4 ranks).
3. `Using MarlinExperts` ← the make-or-break line. If TRTLLM/DeepGEMM: wrong.
   If `NotImplementedError: No MXFP4 MoE backend supports ...`: inspect
   `is_supported_config` rejections; selector honors explicit non-auto backend.
4. `DeepSeek V4 o_proj: using native SM89 block-scaled FP8 grouped matmul`
   — if absent on sm89, patches didn't land.
5. `Available KV cache memory: 2.43 GiB` / `GPU KV cache size: 158,769 tokens`
   — must be POSITIVE. Negative = weights+graphs+activations ate the budget →
   shrink cudagraph sizes / batched tokens first, raise util last.
6. `Application startup complete.` Benign noise: SymmMemCommunicator "not
   supported" warnings (sm89); a torch "kernel cache directory could not be
   created" mkdir race (pre-create ~/.cache/torch/kernels).

### 8. Acceptance numbers (vllm bench serve, this config, ECC on)
```bash
venv/bin/vllm bench serve --backend vllm --base-url http://localhost:8001 \
  --model deepseek-v4-flash --tokenizer ~/models/DeepSeek-V4-Flash \
  --tokenizer-mode deepseek_v4 --trust-remote-code --dataset-name random \
  --random-input-len 1024 --random-output-len 256 --num-prompts 16 \
  --max-concurrency 4 --ignore-eos
```
| Scenario (in→out, conc) | Measured 2026-07-23 |
|---|---|
| 128→512, c1 | 57 tok/s decode (TPOT 17.5 ms), median TTFT 238 ms |
| 1024→256, c4 | 161 tok/s out (peak 200), TPOT 22 ms, TTFT 708 ms, 806 tok/s total |
| 16384→128, c4 | 3,280 tok/s total; ~2.9K tok/s/req prefill; 16K TTFT 5.6 s median |

Sanity: the "all but 9 sheep" riddle + 127×43=5461 both answered correctly.
Quick check: `curl localhost:8001/v1/models`.

## Tuning headroom (deliberately not yet applied)
- `--max-model-len`: 32768 is conservative; pool is 158,769 tokens and the
  model is 1M-native. Raise freely within pool ÷ expected concurrency.
- Prefix caching: flip `--no-enable-prefix-caching` off only after verifying a
  repeated long agent prompt gives identical answers (docker launcher §4).
- ECC off ≈ 2x KV pool. MTP: unverified with Marlin — test before trusting.

## Known risks / caveats
- The combo (Marlin experts + Ada patches, W2 off) is now field-tested by US
  (this box, 2026-07-23) but still undocumented by the fork authors. No formal
  quality eval yet — only spot checks.
- Repo is young/0-star but auditable: official wheel + 70 readable diffs, no
  binary blobs on the sm89 path (cubins are SM120-only, never load).
- If TP4 throughput is bad, check PCIe link width: `nvidia-smi topo -m`.
- Marlin hidden-size round-up to 256 is automatic; fine for DS4 shapes.
- Fallback if quality/perf disappoints: W2 GPU-residency mode (`VLLM_MOE_W2=1`,
  `RESIDENCY=gpu`, `VLLM_MOE_W2_DELTA_GB=0` — FP4 delta NOT ported to Ada).
  Docs: `docs/ada-sm89-port.md`; launcher: `docker/serve_sm89_ds4.sh` (its
  header/TECHNICAL NOTES are the deep-why reference for every knob we cribbed).
- Triage-only env knobs: `VLLM_DSV4_SPARSE_MLA_FORCE_TRITON=1` (force Triton
  MLA anywhere), `VLLM_DSV4_SPARSE_MLA_SELFTEST=0` (skip self-test; keep ON).

## Hardware notes (validated box)
Driver 580.95.05 / CUDA 13.0, 4x L40S all PIX on one switch, NUMA node 1,
503 GB RAM. L40S = AD102, 142 SMs / 18,176 cores @ 350 W, GDDR6 864 GB/s
(ECC inline: −3 GiB/card visible + some bandwidth). Decode is bandwidth-bound:
per-card a 4090D (114 SMs but GDDR6X 1008 GB/s) decodes faster; TP4 aggregate
bandwidth is why this box beats the fork's 2x4090D W2 numbers (38 tok/s no-MTP).

## Repo hygiene rule (from AGENTS.md)
Edit vLLM code only under `overlay/vllm/`, then regenerate patches:
`python3 tools/gen_patches.py && python3 tools/gen_patches.py --verify --check`.
Never hand-edit `patches/*.patch`. There is no fork branch of vLLM anywhere.
