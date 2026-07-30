#!/usr/bin/env bash
# Clean A/B: current main vs main-without-PR#10414, with and without the prefill
# CUDA graph. No [STEP]/[PCG] logging -- that logging makes breakable prefill
# CUDA graph capture fail, so throughput numbers must be taken without it.
# Only SGLANG_DBG_DP_PAD is set, which is a plain env read inside
# DpPaddingMode.get_dp_padding_mode and does not touch the capture path.
#
#   usage: run_ab_clean.sh <pad_variant> <prefill_cg_backend>
set -uo pipefail

PAD="${1:?pad variant}"
PCG="${2:?prefill cuda graph backend}"
TAG="${PAD}-${PCG}"
OUT_DIR="/scratch/dppad4/out/${TAG}"
mkdir -p "$OUT_DIR"

export HF_HOME=/cluster-storage/models
unset SGLANG_DBG_DP_LOG
unset SGLANG_DBG_GEMM_M
case "$PAD" in
  main)    export SGLANG_DBG_DP_PAD="" ;;
  no10414) export SGLANG_DBG_DP_PAD="heuristic" ;;
  max)     export SGLANG_DBG_DP_PAD="max" ;;
  *) echo "bad pad variant"; exit 2 ;;
esac

case "$TAG" in
  main-breakable)    PORT=41000 ;;
  no10414-breakable) PORT=42000 ;;
  main-disabled)     PORT=43000 ;;
  no10414-disabled)  PORT=44000 ;;
  max-breakable)     PORT=45000 ;;
  max-disabled)      PORT=46000 ;;
  *)                 PORT=47000 ;;
esac

SERVER_LOG="${OUT_DIR}/server.log"
echo "=== launch ${TAG} (SGLANG_DBG_DP_PAD='${SGLANG_DBG_DP_PAD}', prefill cg=${PCG}, port=${PORT}) ==="
nohup python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --tp 8 --dp 8 --enable-dp-attention \
  --cuda-graph-backend-prefill "$PCG" \
  --port "$PORT" \
  --mem-fraction-static 0.80 \
  --disable-radix-cache \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 60); do
  grep -q "fired up" "$SERVER_LOG" 2>/dev/null && break
  kill -0 "$SERVER_PID" 2>/dev/null || break
  sleep 10
done
if ! grep -q "fired up" "$SERVER_LOG" 2>/dev/null; then
  echo "SERVER NOT READY"; tail -20 "$SERVER_LOG"; kill "$SERVER_PID" 2>/dev/null; exit 1
fi
echo "server ready"

run_bench() {
  local name="$1"; shift
  echo "=== bench ${name} ==="
  python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port "$PORT" "$@" \
    > "${OUT_DIR}/bench_${name}.log" 2>&1
  grep -E "Successful requests|Input token throughput|Mean TTFT" "${OUT_DIR}/bench_${name}.log"
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "SERVER DIED during ${name}"; return 1; }
}

# 256/512/1024 fit the captured prefill buckets (max bucket is 1024), 4096 does not.
run_bench short_256 --dataset-name random --random-input-len 256 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 1024 --max-concurrency 32
run_bench short_512 --dataset-name random --random-input-len 512 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 512 --max-concurrency 32
run_bench med_1024 --dataset-name random --random-input-len 1024 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 512 --max-concurrency 16
run_bench long_4096 --dataset-name random --random-input-len 4096 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 128 --max-concurrency 8
run_bench skewed_2048 --dataset-name random --random-input-len 2048 --random-output-len 1 \
  --random-range-ratio 0.05 --num-prompts 256 --max-concurrency 8

kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
for i in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
echo "=== done ${TAG} ==="
