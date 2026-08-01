#!/usr/bin/env bash
# bench_gen.sh — gen-heavy sweep (eval profile B: ~512 in / 4096 out).
cd "$(dirname "$0")"
for C in "$@"; do
  echo "########## concurrency=$C ##########"
  ./venv/bin/vllm bench serve --backend vllm --base-url http://localhost:8001 \
    --model deepseek-v4-flash --tokenizer /media/4TBNVME/models/DeepSeek-V4-Flash-0731 \
    --tokenizer-mode deepseek_v4 --trust-remote-code --dataset-name random \
    --random-input-len 512 --random-output-len 4096 \
    --num-prompts "$C" --max-concurrency "$C" --ignore-eos 2>&1 \
  | grep -E "Successful requests|Failed requests|Benchmark duration|Output token throughput|Peak output|Total token throughput|Mean TPOT|Median TPOT|Mean TTFT|Median TTFT"
done
