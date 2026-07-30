#!/usr/bin/env bash
# Round 4 (prefill CUDA graph disabled, probe on):
#   * how many rows actually reach attention under MAX_LEN vs SUM_LEN on main
#   * whether MLA's ReplicatedLinear down-projections are recomputed per attn-TP rank
#
#   usage: run_round4.sh <pad_variant> <tp> <dp>
set -uo pipefail

PAD="${1:?pad variant}"
TP="${2:?tp}"
DP="${3:?dp}"
TAG="${PAD}-tp${TP}dp${DP}"
OUT_DIR="/scratch/dppad5/out/${TAG}"
mkdir -p "$OUT_DIR"

export HF_HOME=/cluster-storage/models
export SGLANG_DBG_DP_LOG=1
export SGLANG_DBG_GEMM_M=1
case "$PAD" in
  main)    export SGLANG_DBG_DP_PAD="" ;;
  no10414) export SGLANG_DBG_DP_PAD="heuristic" ;;
  max)     export SGLANG_DBG_DP_PAD="max" ;;
  *) echo "bad pad variant"; exit 2 ;;
esac

PORT=$((48000 + TP * 100 + DP * 10))
case "$PAD" in main) PORT=$((PORT + 1));; no10414) PORT=$((PORT + 2));; max) PORT=$((PORT + 3));; esac

SERVER_LOG="${OUT_DIR}/server.log"
echo "=== launch ${TAG} (pad='${SGLANG_DBG_DP_PAD}', tp=${TP} dp=${DP}, attn_tp=$((TP / DP)), port=${PORT}) ==="
nohup python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --tp "$TP" --dp "$DP" --enable-dp-attention \
  --cuda-graph-backend-prefill disabled \
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
  grep -E "Successful requests|Input token throughput" "${OUT_DIR}/bench_${name}.log"
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "SERVER DIED during ${name}"; return 1; }
}

run_bench mixed_len --dataset-name random --random-input-len 1024 --random-output-len 1 \
  --random-range-ratio 0.2 --num-prompts 512 --max-concurrency 32
run_bench balanced_2048 --dataset-name random --random-input-len 2048 --random-output-len 1 \
  --random-range-ratio 1.0 --num-prompts 256 --max-concurrency 16

echo "=== final [PROBE] dump ==="
grep "\[PROBE\]" "$SERVER_LOG" | tail -40

kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
for i in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
echo "=== done ${TAG} ==="
