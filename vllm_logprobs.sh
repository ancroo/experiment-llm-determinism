#!/usr/bin/env bash
# Follow-up to vllm_check.sh: all texts were identical there, in both modes. Is that because the
# computation is identical, or because no token was close enough to flip? Same tests, but now
# comparing the logprob of every generated token at full precision (LOGPROBS=1).
#   run   : fresh server, 10 sequential requests
#   noise : fresh server, 10 requests with 3 concurrent strangers each
#   again : another fresh server, 5 sequential requests (compare with run)
# Output: results/vllmlp_{default,bi}_{run,noise,again}.jsonl
set -euo pipefail
cd "$(dirname "$0")"
OUT=results
mkdir -p "$OUT"
SERVE=${SERVE:-/opt/llm/serve.sh}
export PORT=${PORT:-8000} API=openai LOGPROBS=1

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
  rm -f "$OUT"/vllmlp_"$mode"_*.jsonl
  echo "$(date +%T) $mode: run";   start "$mode"; python3 measure.py "$OUT/vllmlp_${mode}_run.jsonl" gen 10
  echo "$(date +%T) $mode: noise"; start "$mode"; python3 measure.py "$OUT/vllmlp_${mode}_noise.jsonl" gen 10 --noise
  echo "$(date +%T) $mode: again"; start "$mode"; python3 measure.py "$OUT/vllmlp_${mode}_again.jsonl" gen 5
done
systemctl --user stop det-vllm

python3 - "$OUT" <<'EOF2'
import json, sys, pathlib, math
d = pathlib.Path(sys.argv[1])
L = lambda n: [json.loads(l) for l in open(d / n)]
ref = None
for mode in ("default", "bi"):
    rows = {k: L(f"vllmlp_{mode}_{k}.jsonl") for k in ("run", "noise", "again")}
    base = rows["run"][0]
    ref = ref or base
    for k, rs in rows.items():
        texts = {r["hash"] for r in rs}
        lps = {r["lp_hash"] for r in rs}
        dmax = max(max(abs(a - b) for a, b in zip(base["logprobs"], r["logprobs"])) for r in rs)
        print(f"{mode:7} {k:5}: texts {len(texts)}, logprob sets {len(lps)}, max |dlogprob| vs run#1 {dmax:.3g}")
    # closest call in the story: smallest gap between best and second-best candidate
    gaps = []
    for i, t in enumerate(base["top2"]):
        v = sorted(t.values(), reverse=True)
        if len(v) == 2:
            gaps.append((math.exp(v[0]) - math.exp(v[1]), i))
    g, i = min(gaps)
    print(f"{mode:7} closest call: token {i}, gap {100*g:.2f} percentage points")
same = ref["hash"] == L("vllmlp_bi_run.jsonl")[0]["hash"]
print("default vs bi: same text:", same,
      "| max |dlogprob|:", f'{max(abs(a-b) for a,b in zip(ref["logprobs"], L("vllmlp_bi_run.jsonl")[0]["logprobs"])):.3g}')
EOF2
echo "$(date +%T) done"
