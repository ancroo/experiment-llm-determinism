#!/usr/bin/env bash
# Third model: Qwen3-1.7B, small and unquantized, many runs, on llama.cpp and on vLLM.
# All engines load the same weights (Hugging Face Qwen/Qwen3-1.7B, revision 70d244cc, BF16
# safetensors) in FP16: llama.cpp as an F16 GGUF made with llama.cpp's convert_hf_to_gguf.py,
# vLLM with --dtype float16. Unlike the Qwen3-8B comparison (Q4_K_M vs. AWQ), quantization
# does not differ between the engines. Same prompt and chat format as the Qwen3-8B tests.
# Every request records the logprob of each generated token (LOGPROBS=1) and its wall time.
#
# Usage: small_model.sh llama   (LABEL=vulkan or LABEL=cuda: names the output files)
#        small_model.sh vllm    (default mode first, then VLLM_BATCH_INVARIANT=1)
#
# Series, temperature 0, 200 tokens; llama.cpp: 1 slot, no prompt cache unless noted;
# vLLM: prefix caching at its default:
#   run    : one long-running server, RUNS sequential requests (default 300)
#   seq    : 32 sequential requests after a fresh start, 3 fresh starts
#   nolp   : 10 sequential requests after a fresh start, without logprobs (control: the texts
#            must equal the first 10 of seq if recording logprobs changes nothing)
#   noise  : 100 requests, each with 3 concurrent strangers (llama.cpp: 4-slot server)
#   slots4 : 100 sequential requests on a fresh 4-slot server (llama.cpp only, control for noise)
#   fresh  : fresh server per request, 20 requests (llama.cpp only)
# Output: results/small_<LABEL>_<series>.jsonl, results/small_vllm_{default,bi}_<series>.jsonl
# Summary: python3 small_model_summary.py results
# Configuration: MODEL (F16 GGUF), LLAMA_BIN, PORT, RUNS, LABEL; SERVE (vLLM start script that
# serves the model as "Qwen3-1.7B" in FP16, port as first argument).
set -euo pipefail
cd "$(dirname "$0")"
OUT=results
mkdir -p "$OUT"
ENGINE=${1:?usage: small_model.sh llama|vllm}
RUNS=${RUNS:-300}
export LOGPROBS=1

wait_up() {  # $1 = seconds
  for _ in $(seq 1 "$1"); do curl -sf "127.0.0.1:$PORT/health" >/dev/null && return; sleep 1; done
  echo "server did not start" >&2; exit 1
}
m() { echo "$(date +%T) $1"; shift; python3 measure.py "$@"; }

case "$ENGINE" in
llama)
  LABEL=${LABEL:?set LABEL=vulkan or LABEL=cuda}
  MODEL=${MODEL:-~/models/Qwen3-1.7B-F16.gguf}
  LLAMA_BIN=${LLAMA_BIN:-~/llama.cpp/build/bin}
  export PORT=${PORT:-8091}
  P="$OUT/small_$LABEL"
  rm -f "$P"_*.jsonl
  start() {  # $1 = number of slots
    systemctl --user stop det-test 2>/dev/null || true
    systemd-run --user --unit=det-test --collect "$LLAMA_BIN"/llama-server \
      -m "$MODEL" -ngl 99 -fa on -c 16384 --parallel "$1" \
      --host 127.0.0.1 --port "$PORT" >/dev/null 2>&1
    wait_up 120
  }
  start 1; m "run, $RUNS requests" "${P}_run.jsonl" gen "$RUNS"
  for r in 1 2 3; do start 1; m "seq, fresh start $r" "${P}_seq_$r.jsonl" gen 32; done
  start 1; echo "$(date +%T) nolp"; LOGPROBS=0 python3 measure.py "${P}_nolp.jsonl" gen 10
  echo "$(date +%T) fresh server per request"
  for _ in $(seq 1 20); do start 1; python3 measure.py "${P}_fresh.jsonl" gen 1; done
  start 4; m "noise, 4 slots, 3 concurrent strangers" "${P}_noise.jsonl" gen 100 --noise
  start 4; m "slots4, 4 slots, sequential" "${P}_slots4.jsonl" gen 100
  systemctl --user stop det-test
  ;;
vllm)
  SERVE=${SERVE:-/opt/llm/serve-qwen3-1.7b.sh}
  export PORT=${PORT:-8000} API=openai MODEL_NAME=Qwen3-1.7B
  start() {  # $1 = default | bi
    systemctl --user stop det-vllm 2>/dev/null || true
    local env=()
    [ "$1" = bi ] && env=(--setenv=VLLM_BATCH_INVARIANT=1)
    systemd-run --user --unit=det-vllm --collect -p KillSignal=SIGINT "${env[@]}" \
      "$SERVE" "$PORT" >/dev/null 2>&1
    wait_up 300
  }
  for mode in default bi; do
    P="$OUT/small_vllm_$mode"
    rm -f "$P"_*.jsonl
    start "$mode"; m "$mode: run, $RUNS requests" "${P}_run.jsonl" gen "$RUNS"
    for r in 1 2 3; do start "$mode"; m "$mode: seq, fresh start $r" "${P}_seq_$r.jsonl" gen 32; done
    start "$mode"; echo "$(date +%T) $mode: nolp"; LOGPROBS=0 python3 measure.py "${P}_nolp.jsonl" gen 10
    start "$mode"; m "$mode: noise, 3 concurrent strangers" "${P}_noise.jsonl" gen 100 --noise
  done
  systemctl --user stop det-vllm
  ;;
*) echo "usage: small_model.sh llama|vllm" >&2; exit 2 ;;
esac
echo "$(date +%T) done"
