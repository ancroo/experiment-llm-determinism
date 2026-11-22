#!/usr/bin/env bash
# Full measurement series (s1-s9). Starts its own llama-server on 127.0.0.1:$PORT
# as a transient systemd user unit (Linux), so no other users share it.
#
# Configuration via environment:
#   MODEL      path to the GGUF file   (default: ~/models/Qwen3-8B-Q4_K_M.gguf)
#   LLAMA_BIN  llama.cpp bin directory (default: ~/llama.cpp/build/bin)
#   PORT       server port             (default: 8091; also read by measure.py)
#   RUNS       runs per series         (default: 30)
set -euo pipefail
cd "$(dirname "$0")"
OUT=results
mkdir -p "$OUT"
rm -f "$OUT"/s*.jsonl
MODEL=${MODEL:-~/models/Qwen3-8B-Q4_K_M.gguf}
LLAMA_BIN=${LLAMA_BIN:-~/llama.cpp/build/bin}
PORT=${PORT:-8091}
RUNS=${RUNS:-30}

start() {  # $1 = number of slots
  systemctl --user stop det-test 2>/dev/null || true
  systemd-run --user --unit=det-test --collect "$LLAMA_BIN"/llama-server \
    -m "$MODEL" -ngl 99 -fa on -c 16384 --parallel "$1" \
    --host 127.0.0.1 --port "$PORT" >/dev/null 2>&1
  for _ in $(seq 1 90); do curl -sf "127.0.0.1:$PORT/health" >/dev/null && return; sleep 1; done
  echo "server did not start" >&2; exit 1
}
m() { echo "$(date +%T) $1"; shift; python3 measure.py "$@"; }

start 1
m "s1 temp 0.7, no seed"        "$OUT/s1_t07_noseed.jsonl" gen "$RUNS" --temp 0.7
m "s2 temp 0.7, seed 42"        "$OUT/s2_t07_seed.jsonl"   gen "$RUNS" --temp 0.7 --seed 42
m "s3 temp 0, no cache"         "$OUT/s3_t0.jsonl"         gen "$RUNS"
m "s4 temp 0, prompt cache"     "$OUT/s4_t0_cache.jsonl"   gen "$RUNS" --cache
m "s5 prompt processing only"   "$OUT/s5_probe.jsonl"      probe "$RUNS"

echo "$(date +%T) s6/s7 fresh server per run"
for _ in $(seq 1 10); do start 1; python3 measure.py "$OUT/s6_t0_fresh.jsonl" gen 1; done
for _ in $(seq 1 5);  do start 1; python3 measure.py "$OUT/s7_t07_seed_fresh.jsonl" gen 1 --temp 0.7 --seed 42; done

start 4
m "s8 temp 0, 3 concurrent strangers" "$OUT/s8_t0_noise.jsonl" gen "$RUNS" --noise

# Control for s8: same 4-slot server configuration, but sequential and without strangers.
# (In the recorded data, s9 was measured separately on a freshly started 4-slot server.)
start 4
m "s9 temp 0, 4 slots, sequential" "$OUT/s9_t0_4slots.jsonl" gen "$RUNS"

systemctl --user stop det-test
echo "$(date +%T) done"
