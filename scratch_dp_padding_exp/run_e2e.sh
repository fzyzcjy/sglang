#!/usr/bin/env bash
# DP-attention padding-mode A/B on 8 GPUs.
#   usage: run_e2e.sh <variant>      variant = stock | heuristic | max
set -uo pipefail

VARIANT="${1:?variant required: stock|heuristic|max}"
OUT_DIR="/scratch/dppad/out/${VARIANT}"
mkdir -p "$OUT_DIR"

export HF_HOME=/cluster-storage/models
export SGLANG_DBG_DP_LOG=1
case "$VARIANT" in
  stock)     export SGLANG_DBG_DP_PAD="" ;;
  heuristic) export SGLANG_DBG_DP_PAD="heuristic" ;;
  max)       export SGLANG_DBG_DP_PAD="max" ;;
  *) echo "bad variant"; exit 2 ;;
esac

SERVER_LOG="${OUT_DIR}/server.log"
PORT=31000

echo "=== launching server (variant=${VARIANT}, SGLANG_DBG_DP_PAD='${SGLANG_DBG_DP_PAD}') ==="
nohup python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --tp 8 --dp 8 --ep 8 --enable-dp-attention \
  --port "$PORT" \
  --mem-fraction-static 0.82 \
  --disable-radix-cache \
  --max-running-requests 128 \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "server pid ${SERVER_PID}"

for i in $(seq 1 120); do
  if grep -q "The server is fired up and ready to roll" "$SERVER_LOG" 2>/dev/null; then
    echo "server ready after ${i}0s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SERVER DIED"; tail -40 "$SERVER_LOG"; exit 1
  fi
  sleep 10
done

if ! grep -q "The server is fired up and ready to roll" "$SERVER_LOG"; then
  echo "SERVER TIMEOUT"; tail -40 "$SERVER_LOG"; kill "$SERVER_PID"; exit 1
fi

run_bench() {
  local name="$1"; shift
  echo "=== bench ${name} ==="
  python -m sglang.bench_serving --backend sglang-oai --port "$PORT" "$@" \
    > "${OUT_DIR}/bench_${name}.log" 2>&1
  grep -E "Request throughput|Input token throughput|Total token throughput|Mean TTFT|Median TTFT|Successful requests|Benchmark duration" \
    "${OUT_DIR}/bench_${name}.log"
}

# Balanced prefill: 8 in-flight requests of equal length -> one per DP rank.
run_bench balanced_2048 --dataset-name random --random-input-len 2048 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 512 --max-concurrency 8
run_bench balanced_8192 --dataset-name random --random-input-len 8192 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 192 --max-concurrency 8
run_bench balanced_c64_2048 --dataset-name random --random-input-len 2048 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 1024 --max-concurrency 64
# Skewed prefill: wide length spread -> uneven per-rank token counts.
run_bench skewed_ratio01 --dataset-name random --random-input-len 8192 --random-output-len 1 \
  --random-range-ratio 0.05 --num-prompts 512 --max-concurrency 8
# Mixed prefill+decode: long outputs so some ranks decode while others prefill.
run_bench mixed_decode --dataset-name random --random-input-len 4096 --random-output-len 256 \
  --random-range-ratio 1.0 --num-prompts 512 --max-concurrency 32

echo "=== DPPAD/DPATTN log summary ==="
grep -c "\[DPPAD\]" "$SERVER_LOG" || true
grep "\[DPPAD\]" "$SERVER_LOG" | head -40
echo "---"
grep "\[DPATTN\]" "$SERVER_LOG" | head -40

kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
echo "=== done variant=${VARIANT} ==="
