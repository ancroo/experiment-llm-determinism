#!/usr/bin/env bash
# Does the "server memory" effect also appear without a GPU (CPU backend only)?
# Two fresh starts, 12 sequential requests each, 1 slot, temperature 0, no prompt cache.
# `-ngl 0` alone is not enough: llama.cpp still offloads ops with a batch of >= 32 tokens
# (GGML_OP_OFFLOAD_MIN_BATCH) to the GPU, and the prompt has 33 tokens. Hence also
# `--device none --no-op-offload`, so that no computation runs on the GPU.
# Output: results/cpu_seq_{1,2}.jsonl
# Configuration: MODEL, LLAMA_BIN, PORT as in run_all.sh.
set -euo pipefail
cd "$(dirname "$0")"
OUT=results
mkdir -p "$OUT"
rm -f "$OUT"/cpu_seq_*.jsonl
MODEL=${MODEL:-~/models/Qwen3-8B-Q4_K_M.gguf}
LLAMA_BIN=${LLAMA_BIN:-~/llama.cpp/build/bin}
PORT=${PORT:-8091}

start() {
  systemctl --user stop det-test 2>/dev/null || true
  systemd-run --user --unit=det-test --collect "$LLAMA_BIN"/llama-server \
    -m "$MODEL" --device none --no-op-offload -ngl 0 -fa on -c 16384 --parallel 1 \
    --host 127.0.0.1 --port "$PORT" >/dev/null 2>&1
  for _ in $(seq 1 120); do curl -sf "127.0.0.1:$PORT/health" >/dev/null && return; sleep 1; done
  echo "server did not start" >&2; exit 1
}

for r in 1 2; do start; python3 measure.py "$OUT/cpu_seq_$r.jsonl" gen 12; done
systemctl --user stop det-test
echo "done"
