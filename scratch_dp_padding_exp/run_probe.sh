#!/usr/bin/env bash
# Per-GEMM row-count probe under DP attention.
#   usage: run_probe.sh <pad_variant> <prefill_cg_backend>
#     pad_variant        = stock | heuristic | max
#     prefill_cg_backend = disabled | breakable | tc_piecewise
set -uo pipefail

PAD="${1:?pad variant}"
PCG="${2:?prefill cuda graph backend}"
TAG="${PAD}-${PCG}"
OUT_DIR="/scratch/dppad2/out/${TAG}"
mkdir -p "$OUT_DIR"

export HF_HOME=/scratch/hf
export SGLANG_DBG_DP_LOG=1
export SGLANG_DBG_GEMM_M=1
case "$PAD" in
  stock)     export SGLANG_DBG_DP_PAD="" ;;
  heuristic) export SGLANG_DBG_DP_PAD="heuristic" ;;
  max)       export SGLANG_DBG_DP_PAD="max" ;;
  *) echo "bad pad variant"; exit 2 ;;
esac

SERVER_LOG="${OUT_DIR}/server.log"
PORT=32000

echo "=== launch tag=${TAG} (SGLANG_DBG_DP_PAD='${SGLANG_DBG_DP_PAD}', prefill cg=${PCG}) ==="
nohup python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --tp 2 --dp 2 --enable-dp-attention \
  --cuda-graph-backend-prefill "$PCG" \
  --port "$PORT" \
  --mem-fraction-static 0.80 \
  --disable-radix-cache \
  --max-running-requests 64 \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 90); do
  grep -q "The server is fired up and ready to roll" "$SERVER_LOG" 2>/dev/null && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SERVER DIED"; grep -iE "error|Exception|ValueError|assert" "$SERVER_LOG" | head -20; exit 1
  fi
  sleep 10
done
grep -q "The server is fired up and ready to roll" "$SERVER_LOG" || {
  echo "SERVER TIMEOUT"; tail -30 "$SERVER_LOG"; kill "$SERVER_PID"; exit 1; }
echo "server ready"

run_bench() {
  local name="$1"; shift
  echo "=== bench ${name} ==="
  python -m sglang.bench_serving --backend sglang-oai --port "$PORT" "$@" \
    > "${OUT_DIR}/bench_${name}.log" 2>&1
  grep -E "Input token throughput|Mean TTFT|Successful requests|Benchmark duration" \
    "${OUT_DIR}/bench_${name}.log"
}

run_bench balanced_2048 --dataset-name random --random-input-len 2048 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 256 --max-concurrency 2
run_bench balanced_c16 --dataset-name random --random-input-len 2048 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 512 --max-concurrency 16
run_bench skewed --dataset-name random --random-input-len 4096 --random-output-len 1 \
  --random-range-ratio 0.05 --num-prompts 256 --max-concurrency 4

echo "=== final GEMMPROBE dump ==="
curl -s "http://127.0.0.1:${PORT}/flush_cache" > /dev/null 2>&1
sleep 2
grep "\[GEMMPROBE\]" "$SERVER_LOG" | tail -60

kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
echo "=== done ${TAG} ==="
