#!/usr/bin/env bash
# restart.sh <logname> [env assignments...] — stop any running engine, boot a
# fresh one detached, wait for readiness. Kills by GPU compute-app PID because
# the TP workers survive a pkill on the parent (learned 2026-08-01).
set -uo pipefail
cd "$(dirname "$0")"
LOG=${1:?usage: restart.sh <logname> [VAR=val ...]}; shift || true

pkill -9 -f "vllm serve" 2>/dev/null
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 "$p" 2>/dev/null; done
for i in $(seq 1 30); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break
  sleep 2
done
echo "GPUs clear; booting $LOG with: $*"
(setsid nohup env "$@" ./serve_l40s_ds4_tp8.sh > "logs/$LOG.log" 2>&1 < /dev/null &)

for i in $(seq 1 150); do
  if grep -qE "Application startup complete|Traceback \(most recent" "logs/$LOG.log" 2>/dev/null; then break; fi
  sleep 10
done
echo "=== waited ~$((i*10))s ==="
grep -nE "Using MarlinExperts|o_proj: using native|Available KV cache|GPU KV cache size|aximum concurrency|Speculative|dspark|DSpark|Application startup complete" "logs/$LOG.log" | tail -15
grep -nE "Traceback|ValueError|RuntimeError|NotImplementedError|CUDA out of memory" "logs/$LOG.log" | tail -8
