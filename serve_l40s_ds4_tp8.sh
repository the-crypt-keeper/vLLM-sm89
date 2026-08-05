#!/usr/bin/env bash
# serve_l40s_ds4_tp8.sh — DeepSeek-V4-Flash-0731 (GA) on 8x L40S (sm89), NO 2-bit quant.
# Derived from serve_l40s_ds4.sh (4x box, preview weights). Deltas:
#   * TP=8 (was 4). Topology here is TWO PIX islands (0-3 | 4-7) joined by SYS
#     across NUMA nodes -> every all-reduce crosses the host bridge. Custom
#     all-reduce stays OFF (no P2P across the boundary).
#   * GA weights /media/4TBNVME/models/DeepSeek-V4-Flash-0731 (48 shards, 156 GB)
#     which ship a DSpark speculative-decoding module (mtp.0/1/2 + markov_head).
#   * ECC is OFF on this box (49140 MiB visible/card) -> much larger KV pool.
#   * SPEC=dspark uses the model card's official flag; target and draft weights
#     come from the same checkpoint (no separate draft model path).
#
# Boot checklist (grep the log):
#   1) MXFP4 backend selection must name MARLIN (not TRTLLM/DeepGEMM)
#   2) 'DeepSeek V4 o_proj: using native SM89 block-scaled FP8 grouped matmul'
#   3) sparse-MLA self-test pass at init
#   4) 'Available KV cache memory' POSITIVE
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-/media/4TBNVME/models/DeepSeek-V4-Flash-0731}
PORT=${PORT:-8001}
TP=${TP:-8}
MAXLEN=${MAXLEN:-16384}      # the evaluation's requirement
# fp8 was inherited from the 4-card box where KV pool size was the binding
# constraint. Here it isn't (23 GiB free, 52x concurrency at 16K), and fp8 KV
# is a prime suspect for the over-sharpened output distribution measured on
# 2026-08-01 vs llama.cpp. KV_DTYPE=auto uses the model dtype instead.
KV_DTYPE=${KV_DTYPE:-fp8}
# Manager block size. 256 was inherited from the 4-card doc; the FlashInfer
# kernel page is 64 and the fork's own dflash code documents a 256-vs-64
# mismatch silently producing all-zero context elsewhere. BLOCK_SIZE=64 makes
# manager and kernel units agree (bisect for the long-generation collapse).
BLOCK_SIZE=${BLOCK_SIZE:-256}
UTIL=${UTIL:-0.95}
BATCHED_TOKENS=${BATCHED_TOKENS:-1024}
NUM_SEQS=${NUM_SEQS:-32}
# Capture ladder must REACH the largest decode batch SHAPE. It used to be
# hard-coded to 1..32, so raising NUM_SEQS to 128 left every batch above 32 with
# no captured graph -- a measured 3.7x decode cliff, and silent (the log says
# "Graph capturing finished" either way).
#
# WITH SPECULATIVE DECODING THAT SHAPE IS NUM_SEQS*(SPEC_TOKENS+1), NOT NUM_SEQS.
# Each decode step carries the drafted tokens too, so vLLM's V2 runner computes
# max_decode_tokens = max_num_reqs * decode_query_len (cudagraph_utils.py). On
# 2026-08-04 a ladder of 1,2,4,8 against NUM_SEQS=8 SPEC_TOKENS=3 (shape 32) made
# DSpark measure as a 3.2x LOSS at c8; with 1,2,4,8,16,32 the same config gained
# 2.7x back. Do NOT rely on vLLM's own default either -- it caps at
# min(max_num_seqs*2, 512), and that x2 assumes a 1-token draft.
# Defaults pulled up from the SPEC block below so the ladder can see them; the
# assignments there are idempotent. Keep the two in sync if you change either.
SPEC=${SPEC:-none}
SPEC_TOKENS=${SPEC_TOKENS:-7}
_CG_TARGET=$NUM_SEQS
if [ "$SPEC" != "none" ]; then
  _CG_TARGET=$(( NUM_SEQS * (SPEC_TOKENS + 1) ))
fi
if [ -z "${CUDAGRAPH_SIZES:-}" ]; then
  _s=""; _n=1
  while [ "$_n" -lt "$_CG_TARGET" ]; do _s="${_s}${_s:+,}$_n"; _n=$((_n * 2)); done
  CUDAGRAPH_SIZES="${_s}${_s:+,}$_CG_TARGET"
fi

# BACKEND selects the DSv4 sparse-MLA implementation:
#   triton     (default) vLLM's Triton port -- ours, the validated baseline
#   flashinfer yhfgyyf's sm89 FlashInfer build; ~14% faster at c1 and it roughly
#              doubles what DSpark is worth, but the sm89 fallback is young.
# The three env vars below are all REQUIRED together; see the route resolver in
# vllm/utils/flashinfer.py. PATH must carry ninja (venv/bin) and nvcc because
# this path JIT-compiles at boot -- the Triton path never needed a compiler.
BACKEND=${BACKEND:-triton}
# Unconditional, NOT just on the flashinfer branch. flashinfer-jit-cache refuses
# to pair with a differently-versioned flashinfer-python, and vLLM imports
# flashinfer for things unrelated to sparse MLA -- so once the +sm89.1 wheel is
# installed, BACKEND=triton fails to BOOT without this too (measured 2026-08-04:
# "flashinfer-jit-cache version (0.6.14+cu130) does not match flashinfer version
# (0.6.14+sm89.1)" from WorkerProc init). Inert when the versions do match.
export FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-1}
case "$BACKEND" in
  triton) ;;
  flashinfer)
    export VLLM_DSV4_SPARSE_MLA_FORCE_FLASHINFER=1
    export FLASHINFER_SPARSE_MLA_FORCE_SM89_PRIMS=1
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    export PATH="$(cd "$(dirname "$0")" && pwd)/venv/bin:$CUDA_HOME/bin:$PATH"
    command -v ninja >/dev/null || { echo "BACKEND=flashinfer needs ninja on PATH" >&2; exit 1; }
    command -v nvcc  >/dev/null || { echo "BACKEND=flashinfer needs nvcc on PATH"  >&2; exit 1; }
    ;;
  *) echo "BACKEND must be triton|flashinfer (got '$BACKEND')" >&2; exit 1 ;;
esac
# FULL_AND_PIECEWISE | PIECEWISE | NONE. DSpark replays FULL graphs and captured
# only 3 sizes vs the main model's 6 on 2026-08-01; a batch shape outside those
# faulted in cudagraph_utils.run_fullgraph -> graph.replay(). Drop to PIECEWISE
# or NONE to take DSpark off the full-graph replay path.
CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}
PREFIX_CACHING=${PREFIX_CACHING:-0}  # 0 = correctness baseline
# Always pass --enable-prompt-tokens-details (below): without it vLLM reports
# usage.prompt_tokens_details = null and `cached_tokens` is invisible, so there
# is no way to tell a prefix-cache hit from a miss at the API. Pure telemetry,
# no behaviour change. Essential for agentic harnesses, which re-send a growing
# conversation every step and live or die on the hit rate.
# ENFORCE_EAGER=1 turns off BOTH torch.compile and CUDA graphs. Diagnostic
# only (it costs real throughput): nn.Module forward hooks do not fire inside
# an inductor-compiled region, so any per-module tracing needs this.
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
# PARSERS=0 serves RAW: no reasoning/tool parsers, so `content` carries the
# model's literal output including <think>...</think> instead of being split
# into reasoning_content and tool_calls. Use when the eval harness does its own
# parsing and you want zero server-side transforms between model and scorer.
PARSERS=${PARSERS:-1}
# NAME_SUFFIX tags the served model name with the experiment, e.g.
# NAME_SUFFIX=-dotf32 -> "deepseek-v4-flash-dotf32". The suffixed name is
# FIRST, so it is what the API reports back in `model` and what lands in eval
# output -- the data self-identifies instead of depending on which directory
# you filed it under. The bare name and "auto" stay registered as aliases so
# existing tooling keeps working against the same endpoint.
NAME_SUFFIX=${NAME_SUFFIX:-}
if [ -n "$NAME_SUFFIX" ]; then
  SERVED_NAMES=("deepseek-v4-flash${NAME_SUFFIX}" deepseek-v4-flash auto)
else
  SERVED_NAMES=(deepseek-v4-flash auto)
fi

# MLA_DOT=fp32 runs the sparse-MLA attention dots on tf32 tensor cores instead
# of bf16 (~11 vs ~8 mantissa bits) and halves BLOCK_N to 16 to fit Ada smem.
# Quality lever for long context; costs throughput. See the knob's comment in
# overlay/vllm/vllm/v1/attention/ops/triton_sparse_mla_dsv4.py.
MLA_DOT=${MLA_DOT:-bf16}
case "$MLA_DOT" in
  bf16) export VLLM_DSV4_MLA_DOT_FP32=0 ;;
  fp32) export VLLM_DSV4_MLA_DOT_FP32=1 ;;
  *) echo "MLA_DOT must be bf16|fp32 (got '$MLA_DOT')" >&2; exit 1 ;;
esac

# SPEC: none | dspark | mtp
#   dspark = official GA path (model card): block drafting, 7 tokens, greedy.
#   mtp    = classic DeepSeek MTP head (mtp.0); the fork measured +40% decode
#            on its W2 path. Unverified with Marlin experts.
SPEC=${SPEC:-none}
SPEC_TOKENS=${SPEC_TOKENS:-7}
# SPEC_MODEL points `mtp` at a SEPARATE draft checkpoint. Needed because the
# 0731 GA weights ship a 3-layer DSpark head (mtp.0/1/2 + confidence_head,
# 4708 tensors) that vLLM's deepseek_mtp method cannot load -- it wants the
# clean 1-layer head (1575 tensors) that only the PREVIEW checkpoint has.
# Build one with ../extract_mtp_head.py, then:
#   SPEC=mtp SPEC_TOKENS=2 SPEC_MODEL=/media/4TBNVME/models/DeepSeek-V4-Flash-MTP
# Cross-revision drafting is speed-only risk, not correctness: vLLM's rejection
# sampling keeps the TARGET distribution regardless of draft quality, so a
# mismatched head costs acceptance rate, never wrong tokens.
SPEC_MODEL=${SPEC_MODEL:-}
# Extra JSON fields spliced into the dspark speculative-config, e.g.
#   SPEC_EXTRA='"dspark_scheduler":true,"dspark_per_request":true,"dspark_pad_to_bucket":true'
# pad_to_bucket is what sets real_query_start_loc -> HAS_REAL_LENS=1 in the
# fork's dflash prepare kernel, which is the ONLY path that neutralizes the
# padded context tail. Without it, stale entries from prior steps leak into
# other requests' pages (2026-08-01: illegal memory access on the 2nd batch).
SPEC_EXTRA=${SPEC_EXTRA:-}

# This launcher is the no-quant Marlin path. A leaked VLLM_MOE_W2=1 would
# silently reroute experts to the 2-bit Triton emulation — refuse instead.
if [ "${VLLM_MOE_W2:-0}" = "1" ]; then
  echo "refusing to start: VLLM_MOE_W2=1 is set; unset it (W2 mode has its own launcher)" >&2
  exit 1
fi

EAGERARGS=""
[ "$ENFORCE_EAGER" = "1" ] && EAGERARGS="--enforce-eager"

case "$SPEC" in
  none)   SPECARGS=() ;;
  dspark) SPECARGS=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_TOKENS,\"draft_sample_method\":\"greedy\"${SPEC_EXTRA:+,$SPEC_EXTRA}}") ;;
  mtp)    SPECARGS=(--speculative-config "{\"method\":\"deepseek_mtp\",\"num_speculative_tokens\":$SPEC_TOKENS${SPEC_MODEL:+,\"model\":\"$SPEC_MODEL\"}${SPEC_EXTRA:+,$SPEC_EXTRA}}") ;;
  *) echo "SPEC must be none|dspark|mtp (got '$SPEC')" >&2; exit 1 ;;
esac

# First boot JITs every Triton kernel (sparse MLA, FP8 o_proj) — keep the
# engine-ready window wide, and persist JIT caches so later boots are fast.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}

# Deterministic MoE is ON by default here, not a debugging opt-in (bug #3/#4).
# moe_align_block_size assigns each token's slot with an atomicAdd return value,
# so the bucketing is thread-arrival-ordered; marlin's DP + stream-K schedule
# then makes the fp32 accumulation grouping a function of which block a row
# lands in, so that ordering leaks into the numerics at ~1 ULP. On this model
# 1 ULP is worth ~0.09 nats median at the prompt-logprob level, so without this
# greedy decoding is not reproducible. Costs ~5.3% decode throughput.
# Still overridable: =2 (reversed order) is the order-invariance regression
# gate, =0 restores stock nondeterministic behaviour for A/B.
export VLLM_DSV4_DETERMINISTIC_MOE=${VLLM_DSV4_DETERMINISTIC_MOE:-1}
# Bug #5: the top-k selectors return the right SET and promise no ORDER.
# top_k_per_row_prefill was measured emitting the same 512 candidates in a
# different order across two byte-identical requests, and the sparse-MLA
# attention accumulates in list order, so that permutation is worth ~1 bf16 ULP
# and compounds to 1.6-4.4 nats over 43 layers. =1 sorts each row ascending,
# which made ctx 2048..10240 bit-reproducible where none of it was before
# (verified with cudagraphs on). =2 is the descending control, the same A/B
# role it plays for the MoE knob above.
#
# DEFAULT 0, unlike DETERMINISTIC_MOE, because the cost is not yet measured
# end-to-end. The sort is launch-latency bound at ~0.13 ms/call eager
# (identical for 15 and 4096 rows), so ~5.7 ms per forward across 43 layers --
# which would be severe against a ~20 ms decode TPOT if it survived cudagraph
# capture, and near-free if it does not. Benchmark before flipping this on.
#
# Does NOT fix everything: a second, differently-shaped source remains above
# ~11k context (probabilistic, 3-6 distinct of 6 repeats, not the clean step
# that 2048 was). The ratio-128 layers take a separate path
# (attn_metadata.c128a_prefill_topk_indices) that this does not touch.
export VLLM_DSV4_DETERMINISTIC_TOPK=${VLLM_DSV4_DETERMINISTIC_TOPK:-0}
CACHEROOT=${CACHEROOT:-$HOME/.cache/moet-l40s}
export TRITON_CACHE_DIR=$CACHEROOT/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHEROOT/torchinductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HOME/.cache/torch/kernels"

PREFIXARGS=""
[ "$PREFIX_CACHING" = 1 ] || PREFIXARGS="--no-enable-prefix-caching"

# DEFAULT_CTK sets --default-chat-template-kwargs: template defaults that a
# request can still override (vLLM merges, request wins).
#
# OBSOLETE for its original purpose as of the 2026-08-05 upstream tokenizer sync
# (#50580), and left here only as a general escape hatch. It existed because DS4
# used to treat a MISSING thinking/enable_thinking as thinking-OFF -- the template
# rendered `</think>` pre-closed -- and not every client forwards the field.
# opencode via @ai-sdk/openai-compatible drops it entirely (measured:
# `"reasoning":0` on every step, ~75 output tokens), so without this an agent
# comparison silently became thinking-vs-no-thinking. Upstream now defaults that
# case to thinking ON, so such clients get thinking -- at official *high* effort.
# mini-swe-agent passes reasoning_effort top-level and was unaffected either way.
DEFAULT_CTK=${DEFAULT_CTK:-}
CTKARGS=()
[ -n "$DEFAULT_CTK" ] && CTKARGS=(--default-chat-template-kwargs "$DEFAULT_CTK")

# --enable-auto-tool-choice requires --tool-call-parser, so they drop together.
if [ "$PARSERS" = 1 ]; then
  PARSERARGS=(--tool-call-parser deepseek_v4 --enable-auto-tool-choice
              --reasoning-parser deepseek_v4)
else
  PARSERARGS=()
fi

echo "serving $MODEL  tp=$TP port=$PORT maxlen=$MAXLEN util=$UTIL batched=$BATCHED_TOKENS seqs=$NUM_SEQS graphs=[$CUDAGRAPH_SIZES] mode=$CUDAGRAPH_MODE prefix-cache=$PREFIX_CACHING spec=$SPEC(${SPEC_TOKENS}) backend=$BACKEND det-moe=$VLLM_DSV4_DETERMINISTIC_MOE mla-dot=$MLA_DOT name=${SERVED_NAMES[0]} W2=off(marlin)"

exec ./venv/bin/vllm serve "$MODEL" \
  --served-model-name "${SERVED_NAMES[@]}" \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size "$TP" --disable-custom-all-reduce \
  --kv-cache-dtype "$KV_DTYPE" --block-size "$BLOCK_SIZE" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" \
  --max-num-batched-tokens "$BATCHED_TOKENS" --max-num-seqs "$NUM_SEQS" \
  --no-scheduler-reserve-full-isl \
  --enable-prompt-tokens-details \
  "${PARSERARGS[@]}" \
  "${CTKARGS[@]}" \
  $PREFIXARGS \
  "${SPECARGS[@]}" \
  ${EAGERARGS} \
  --compilation-config '{"cudagraph_mode":"'"$CUDAGRAPH_MODE"'","custom_ops":["all"],"cudagraph_capture_sizes":['"$CUDAGRAPH_SIZES"']}' \
  --port "$PORT"
