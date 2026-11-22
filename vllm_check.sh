#!/usr/bin/env bash
# Does vLLM's batch-invariant mode remove the effects seen with llama.cpp?
# Same tests twice on the same machine: default mode and VLLM_BATCH_INVARIANT=1.
#   run   : fresh server, 30 sequential requests
#   noise : fresh server, 30 requests, each with 3 concurrent "stranger" requests
#   seq   : 16 sequential requests after a fresh start, repeated 3 times
# Model and server: /opt/llm/serve.sh (vLLM, Qwen3-8B-AWQ, temperature 0, prefix caching
# at vLLM's default). Same built-in Qwen3 prompt as the llama.cpp tests.
# Output: results/vllm_{default,bi}_{run,noise,seq_1,seq_2,seq_3}.jsonl
# Configuration: SERVE (start script), PORT.
set -euo pipefail
cd "$(dirname "$0")"
OUT=results
mkdir -p "$OUT"
SERVE=${SERVE:-/opt/llm/serve.sh}
export PORT=${PORT:-8000} API=openai

start() {  # $1 = default | bi
  systemctl --user stop det-vllm 2>/dev/null || true
  local env=()
  [ "$1" = bi ] && env=(--setenv=VLLM_BATCH_INVARIANT=1)
  systemd-run --user --unit=det-vllm --collect -p KillSignal=SIGINT "${env[@]}" \
    "$SERVE" "$PORT" >/dev/null 2>&1
  for _ in $(seq 1 300); do curl -sf "127.0.0.1:$PORT/health" >/dev/null && return; sleep 1; done
  echo "server did not start" >&2; exit 1
}

for mode in default bi; do
  rm -f "$OUT"/vllm_"$mode"_*.jsonl
  echo "$(date +%T) $mode: run";   start "$mode"; python3 measure.py "$OUT/vllm_${mode}_run.jsonl" gen 30
  echo "$(date +%T) $mode: noise"; start "$mode"; python3 measure.py "$OUT/vllm_${mode}_noise.jsonl" gen 30 --noise
  for rep in 1 2 3; do
    echo "$(date +%T) $mode: seq $rep"; start "$mode"
    python3 measure.py "$OUT/vllm_${mode}_seq_${rep}.jsonl" gen 16
  done
done
systemctl --user stop det-vllm

python3 - "$OUT" <<'EOF'
import json, sys, pathlib, collections
d = pathlib.Path(sys.argv[1])
load = lambda n: [json.loads(l)["hash"] for l in open(d / n)]
letters = {}
lab = lambda h: letters.setdefault(h, chr(ord("A") + len(letters)))
for mode in ("default", "bi"):
    for kind in ("run", "noise"):
        hs = load(f"vllm_{mode}_{kind}.jsonl")
        c = collections.Counter(hs)
        print(f"{mode:7} {kind:5}: {len(c)} distinct, most frequent {c.most_common(1)[0][1]}/30")
        print("   ", " ".join(lab(h) for h in hs))
    seqs = [load(f"vllm_{mode}_seq_{r}.jsonl") for r in (1, 2, 3)]
    for i, s in enumerate(seqs, 1):
        print(f"{mode:7} seq {i}:", " ".join(lab(h) for h in s))
    print(f"{mode:7} seq identical across fresh starts:", all(s == seqs[0] for s in seqs))
EOF
echo "$(date +%T) done"
