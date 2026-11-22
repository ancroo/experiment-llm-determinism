"""Repeat one prompt-processing pass and print first-token logprobs at full precision.

If the logprobs differ between identical runs, the kernels themselves are not
run-to-run deterministic (no batching, no cache, no sampling involved).
"""
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8091/completion"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 6

body = {
    "prompt": "The lighthouse keeper opened the bottle and found",
    "n_predict": 1,
    "n_probs": 5,
    "temperature": 0,
    "cache_prompt": False,
}

for i in range(RUNS):
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    top = r["completion_probabilities"][0]["top_logprobs"]
    print(i, "  ".join(f"{t['token']!r}:{t['logprob']:.9f}" for t in top[:3]))
