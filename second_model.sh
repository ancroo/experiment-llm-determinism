#!/usr/bin/env bash
# Repeat the core tests with a second, very different model (default: gpt-oss-120b, MoE, MXFP4).
#   g1: one long-running server, temperature 0, 30 sequential requests ("standard")
#   g2: fresh server per request, temperature 0, 10 requests ("deterministic" recipe)
#   g3: 16 requests after a fresh start, repeated 3 times (is the server as a whole deterministic?)
# Same prompt as the Qwen3 tests, in the model's own chat format (prompts/gpt-oss.txt,
# rendered by llama-server's chat template with reasoning effort "low", date fixed).
# Output: results/g1_*.jsonl, g2_*.jsonl, g3_seq_{1,2,3}.jsonl
# Configuration: MODEL, LLAMA_BIN, PORT, PROMPT_FILE.
set -euo pipefail
cd "$(dirname "$0")"
OUT=results
mkdir -p "$OUT"
rm -f "$OUT"/g[123]_*.jsonl
MODEL=${MODEL:-~/models/gpt-oss-120b-MXFP4.gguf}
LLAMA_BIN=${LLAMA_BIN:-~/llama.cpp/build/bin}
PORT=${PORT:-8091}
export PORT PROMPT_FILE=${PROMPT_FILE:-prompts/gpt-oss.txt}

start() {
  systemctl --user stop det-test 2>/dev/null || true
  systemd-run --user --unit=det-test --collect "$LLAMA_BIN"/llama-server \
    -m "$MODEL" -ngl 99 -fa on -c 16384 --parallel 1 \
    --host 127.0.0.1 --port "$PORT" >/dev/null 2>&1
  for _ in $(seq 1 180); do curl -sf "127.0.0.1:$PORT/health" >/dev/null && return; sleep 1; done
  echo "server did not start" >&2; exit 1
}

echo "$(date +%T) g1 long-running server, 30 sequential requests"
start
python3 measure.py "$OUT/g1_t0_running.jsonl" gen 30

echo "$(date +%T) g2 fresh server per request"
for _ in $(seq 1 10); do start; python3 measure.py "$OUT/g2_t0_fresh.jsonl" gen 1; done

for rep in 1 2 3; do
  echo "$(date +%T) g3 fresh start $rep, 16 requests"
  start
  python3 measure.py "$OUT/g3_seq_${rep}.jsonl" gen 16
done
systemctl --user stop det-test

python3 - "$OUT" <<'EOF'
import json, sys, pathlib, collections
d = pathlib.Path(sys.argv[1])
load = lambda n: [json.loads(l)["hash"] for l in open(d / n)]
letters = {}
lab = lambda h: letters.setdefault(h, chr(ord("A") + len(letters)))
for name in ("g2_t0_fresh.jsonl", "g1_t0_running.jsonl"):
    hs = load(name)
    c = collections.Counter(hs)
    print(f"{name}: {len(hs)} runs, {len(c)} distinct, most frequent {c.most_common(1)[0][1]}x")
    print("   ", " ".join(lab(h) for h in hs))
seqs = [load(f"g3_seq_{r}.jsonl") for r in (1, 2, 3)]
for i, s in enumerate(seqs, 1):
    print(f"g3 fresh start {i}:", " ".join(lab(h) for h in s))
print("g3 identical across fresh starts:", all(s == seqs[0] for s in seqs))
EOF
echo "$(date +%T) done"
