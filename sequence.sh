#!/usr/bin/env bash
# Is the server as a whole deterministic? Same sequence of requests after a fresh start,
# repeated 3 times, with and without prompt cache. If the hash sequences match run for run,
# the answer depends only on the history of requests, not on chance.
# Output: results/seq_{nocache,cache}_{1,2,3}.jsonl (16 runs each)
# Configuration: MODEL, LLAMA_BIN, PORT as in run_all.sh.
set -euo pipefail
cd "$(dirname "$0")"
OUT=results
mkdir -p "$OUT"
rm -f "$OUT"/seq_*.jsonl
MODEL=${MODEL:-~/models/Qwen3-8B-Q4_K_M.gguf}
LLAMA_BIN=${LLAMA_BIN:-~/llama.cpp/build/bin}
PORT=${PORT:-8091}
RUNS=${RUNS:-16}

start() {
  systemctl --user stop det-test 2>/dev/null || true
  systemd-run --user --unit=det-test --collect "$LLAMA_BIN"/llama-server \
    -m "$MODEL" -ngl 99 -fa on -c 16384 --parallel 1 \
    --host 127.0.0.1 --port "$PORT" >/dev/null 2>&1
  for _ in $(seq 1 90); do curl -sf "127.0.0.1:$PORT/health" >/dev/null && return; sleep 1; done
  echo "server did not start" >&2; exit 1
}

for mode in nocache cache; do
  flag=""; [ "$mode" = cache ] && flag="--cache"
  for rep in 1 2 3; do
    echo "$(date +%T) $mode, fresh start $rep"
    start
    python3 measure.py "$OUT/seq_${mode}_${rep}.jsonl" gen "$RUNS" $flag
  done
done
systemctl --user stop det-test

python3 - "$OUT" <<'EOF'
import json, sys, pathlib
for mode in ("nocache", "cache"):
    seqs = []
    for rep in (1, 2, 3):
        rows = [json.loads(l) for l in open(pathlib.Path(sys.argv[1]) / f"seq_{mode}_{rep}.jsonl")]
        seqs.append([r["hash"] for r in rows])
        print(mode, rep, " ".join(h[:4] for h in seqs[-1]),
              "| prompt_n:", sorted({r["prompt_n"] for r in rows}))
    print(mode, "identical across fresh starts:", all(s == seqs[0] for s in seqs))
EOF
echo "$(date +%T) done"
