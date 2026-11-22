"""Show how close the decision at the first divergent token is.

Feeds the prompt plus the first N generated tokens (taken from a recorded run) to the
server and prints the top candidates for the next token with full precision.
Usage: near_tie.py <results.jsonl> <n_tokens>
"""
import json
import math
import os
import sys
import urllib.request

PORT = os.environ.get("PORT", "8091")
BASE = f"http://127.0.0.1:{PORT}"
PROMPT = ("<|im_start|>user\nWrite a short story about a lighthouse keeper who finds a strange "
          "message in a bottle. Be creative.<|im_end|>\n<|im_start|>assistant\n"
          "<think>\n\n</think>\n\n")


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


run = json.loads(open(sys.argv[1]).readline())
n = int(sys.argv[2])
prompt_tokens = post("/tokenize", {"content": PROMPT})["tokens"]
tokens = prompt_tokens + run["tokens"][:n]

r = post("/completion", {"prompt": tokens, "n_predict": 1, "n_probs": 5,
                         "temperature": 0, "cache_prompt": False})
context = post("/detokenize", {"tokens": run["tokens"][max(0, n - 12):n]})["content"]
print(f"...{context!r} ->")
for t in r["completion_probabilities"][0]["top_logprobs"]:
    print(f"  {t['token']!r:>12}  logprob {t['logprob']:.6f}  p = {math.exp(t['logprob']) * 100:.3f} %")
