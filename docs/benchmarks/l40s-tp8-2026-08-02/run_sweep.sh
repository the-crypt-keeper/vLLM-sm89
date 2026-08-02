#!/usr/bin/env bash
# Benchmark sweep for the sm89 / L40S TP8 README.
#   decode  baselines: 1k in / 2k out at concurrency 1,2,4,16,64,128
#   prefill baselines: 8k in / 4k out at concurrency 1,2,4
# Server must already be running with --max-num-seqs >= 128 and maxlen >= 16384.
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root
OUT=${OUT:-./bench-results}
mkdir -p "$OUT"
MODELDIR=${MODELDIR:-/path/to/DeepSeek-V4-Flash-0731}

run() {  # run <in> <out> <concurrency> <num_prompts> <tag>
  local i=$1 o=$2 c=$3 n=$4 tag=$5
  echo "=== $tag : in=$i out=$o conc=$c prompts=$n ==="
  ./venv/bin/vllm bench serve \
    --backend vllm --base-url http://localhost:8001 \
    --model deepseek-v4-flash --tokenizer "$MODELDIR" \
    --tokenizer-mode deepseek_v4 --trust-remote-code \
    --dataset-name random \
    --random-input-len "$i" --random-output-len "$o" \
    --num-prompts "$n" --max-concurrency "$c" \
    --ignore-eos --seed 1234 \
    --save-result --result-filename "$OUT/$tag.json" \
    2>&1 | grep -E "Successful|Benchmark duration|Request throughput|Output token throughput|Total Token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean ITL|Median ITL"
  echo
}

# decode baselines
run 1024 2048 1     4 decode_c1
run 1024 2048 2     8 decode_c2
run 1024 2048 4    12 decode_c4
run 1024 2048 16   32 decode_c16
run 1024 2048 64  128 decode_c64
run 1024 2048 128 256 decode_c128

# prefill baselines
run 8192 4096 1   3 prefill_c1
run 8192 4096 2   6 prefill_c2
run 8192 4096 4  12 prefill_c4

echo "SWEEP COMPLETE"
