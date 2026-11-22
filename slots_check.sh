#!/usr/bin/env bash
# Does the number of slots change the answer of a FRESH server?
# 5 fresh starts each with 1, 2 and 4 slots, one request per start; then the same 16-request
# sequence on a fresh 4-slot server, twice.
# Output: results/slots_first_{1,2,4}.jsonl, results/slots_seq4_{1,2}.jsonl
# Configuration: MODEL, LLAMA_BIN, PORT as in run_all.sh.
set -euo pipefail
cd "$(dirname "$0")"
OUT=results
mkdir -p "$OUT"
rm -f "$OUT"/slots_*.jsonl
MODEL=${MODEL:-~/models/Qwen3-8B-Q4_K_M.gguf}
LLAMA_BIN=${LLAMA_BIN:-~/llama.cpp/build/bin}
PORT=${PORT:-8091}

start() {  # $1 = number of slots
  systemctl --user stop det-test 2>/dev/null || true
  systemd-run --user --unit=det-test --collect "$LLAMA_BIN"/llama-server \
    -m "$MODEL" -ngl 99 -fa on -c 16384 --parallel "$1" \
    --host 127.0.0.1 --port "$PORT" >/dev/null 2>&1
  for _ in $(seq 1 120); do curl -sf "127.0.0.1:$PORT/health" >/dev/null && return; sleep 1; done
  echo "server did not start" >&2; exit 1
}

for n in 1 2 4; do
  for _ in $(seq 1 5); do start "$n"; python3 measure.py "$OUT/slots_first_$n.jsonl" gen 1; done
done
for r in 1 2; do start 4; python3 measure.py "$OUT/slots_seq4_$r.jsonl" gen 16; done
systemctl --user stop det-test
echo "done"
