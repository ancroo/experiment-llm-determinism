#!/usr/bin/env bash
# Cheaper alternatives to "restart the server before every request".
#   V1: one llama-completion process per request (no server)
#   V2: running server, slot 0 erased before every request
# Output: results/v1_cli.txt, results/v2_erase.jsonl
# Configuration: same environment variables as run_all.sh (MODEL, LLAMA_BIN, PORT); RUNS default 15.
set -euo pipefail
cd "$(dirname "$0")"
OUT=results
mkdir -p "$OUT"
rm -f "$OUT"/v1_cli.txt "$OUT"/v2_erase.jsonl
MODEL=${MODEL:-~/models/Qwen3-8B-Q4_K_M.gguf}
LLAMA_BIN=${LLAMA_BIN:-~/llama.cpp/build/bin}
PORT=${PORT:-8091}
RUNS=${RUNS:-15}
SLOTS_DIR=$(mktemp -d)

# Note: `-f` strips ONE trailing newline (common/arg.cpp), which turns the final
# token "\n\n" (271) into "\n" (198) and yields a completely different story.
# The extra "\n" at the end compensates, so the CLI gets the same 33 tokens as the server.
# (The recorded results/v1_cli.txt was produced before this fix, i.e. with the stripped prompt.)
printf '<|im_start|>user\nWrite a short story about a lighthouse keeper who finds a strange message in a bottle. Be creative.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n\n' > prompt.txt

echo "$(date +%T) V1 llama-completion, one process per run"
for _ in $(seq 1 "$RUNS"); do
  "$LLAMA_BIN"/llama-completion -m "$MODEL" -ngl 99 -fa on -c 16384 -f prompt.txt -n 200 \
    --temp 0 --no-display-prompt -no-cnv 2>/dev/null </dev/null | sha256sum | cut -c1-10 >> "$OUT"/v1_cli.txt
done

echo "$(date +%T) V2 server, slot erased before each request"
# The erase action is only available when the server has a --slot-save-path.
systemctl --user stop det-test 2>/dev/null || true
systemd-run --user --unit=det-test --collect "$LLAMA_BIN"/llama-server -m "$MODEL" -ngl 99 -fa on \
  -c 16384 --parallel 1 --slot-save-path "$SLOTS_DIR" --host 127.0.0.1 --port "$PORT" >/dev/null 2>&1
for _ in $(seq 1 90); do curl -sf "127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
for _ in $(seq 1 "$RUNS"); do
  curl -sf -X POST "127.0.0.1:$PORT/slots/0?action=erase" >/dev/null
  python3 measure.py "$OUT"/v2_erase.jsonl gen 1
done
systemctl --user stop det-test
rm -rf "$SLOTS_DIR"
echo "$(date +%T) done"
