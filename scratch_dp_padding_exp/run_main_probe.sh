#!/usr/bin/env bash
# Round 3: current main vs main-without-PR#10414, on 8 GPUs with DP attention.
#   usage: run_main_probe.sh <pad_variant> <prefill_cg_backend>
#     pad_variant        = main | no10414 | max
#     prefill_cg_backend = breakable | disabled
set -uo pipefail

PAD="${1:?pad variant}"
PCG="${2:?prefill cuda graph backend}"
TAG="${PAD}-${PCG}"
OUT_DIR="/scratch/dppad3/out/${TAG}"
mkdir -p "$OUT_DIR"

export HF_HOME=/cluster-storage/models
export SGLANG_DBG_DP_LOG=1
case "$PAD" in
  main)    export SGLANG_DBG_DP_PAD="" ;;        # current main: forced SUM_LEN for extend
  no10414) export SGLANG_DBG_DP_PAD="heuristic" ;;  # PR #10414 condition removed
  max)     export SGLANG_DBG_DP_PAD="max" ;;
  *) echo "bad pad variant"; exit 2 ;;
esac

SERVER_LOG="${OUT_DIR}/server.log"
PORT=33000

echo "=== launch tag=${TAG} (SGLANG_DBG_DP_PAD='${SGLANG_DBG_DP_PAD}', prefill cg=${PCG}) ==="
nohup python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --tp 8 --dp 8 --enable-dp-attention \
  --cuda-graph-backend-prefill "$PCG" \
  --port "$PORT" \
  --mem-fraction-static 0.80 \
  --disable-radix-cache \
  --max-running-requests 128 \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 100); do
  grep -q "The server is fired up and ready to roll" "$SERVER_LOG" 2>/dev/null && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SERVER DIED"; grep -iE "error|Exception|ValueError|assert|Traceback" "$SERVER_LOG" | head -25; exit 1
  fi
  sleep 10
done
grep -q "The server is fired up and ready to roll" "$SERVER_LOG" || {
  echo "SERVER TIMEOUT"; tail -30 "$SERVER_LOG"; kill "$SERVER_PID"; exit 1; }
echo "server ready"
grep -iE "prefill.*cuda graph|Capture.*prefill|capture_num_tokens" "$SERVER_LOG" | head -5

run_bench() {
  local name="$1"; shift
  echo "=== bench ${name} ==="
  python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port "$PORT" "$@" \
    > "${OUT_DIR}/bench_${name}.log" 2>&1
  grep -E "Input token throughput|Mean TTFT|Successful requests|Benchmark duration" \
    "${OUT_DIR}/bench_${name}.log"
}

# Short prompts first: these are the ones small enough to fit a captured prefill
# graph bucket, i.e. exactly the regime where the PR #10414 condition decides
# whether the prefill CUDA graph can be used at all.
run_bench short_256 --dataset-name random --random-input-len 256 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 2048 --max-concurrency 64
run_bench short_512 --dataset-name random --random-input-len 512 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 1024 --max-concurrency 64
run_bench med_1024 --dataset-name random --random-input-len 1024 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 1024 --max-concurrency 32
run_bench long_4096 --dataset-name random --random-input-len 4096 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 256 --max-concurrency 16
run_bench skewed --dataset-name random --random-input-len 2048 --random-output-len 1 \
  --random-range-ratio 0.05 --num-prompts 512 --max-concurrency 16
run_bench mixed --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --random-range-ratio 1.0 --num-prompts 512 --max-concurrency 64

echo "=== [PCG] hit counts ==="
grep -c "\[PCG\]" "$SERVER_LOG" || true
echo "=== sample [STEP]/[PCG] ==="
grep -E "\[STEP\]|\[PCG\]" "$SERVER_LOG" | head -20

kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
echo "=== done ${TAG} ==="
