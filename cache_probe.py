"""Prompt cache on: record per run how many prompt tokens were actually (re)computed.

Prints one line per run: run number, prompt tokens processed (timings.prompt_n),
tokens taken from the cache (tokens_cached), and the output hash.
Usage: cache_probe.py [runs]
"""
import hashlib
import json
import os
import sys
import urllib.request

URL = f"http://127.0.0.1:{os.environ.get('PORT', '8091')}/completion"
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
PROMPT = ("<|im_start|>user\nWrite a short story about a lighthouse keeper who finds a strange "
          "message in a bottle. Be creative.<|im_end|>\n<|im_start|>assistant\n"
          "<think>\n\n</think>\n\n")

for i in range(1, RUNS + 1):
    body = {"prompt": PROMPT, "n_predict": 200, "temperature": 0, "cache_prompt": True}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        r = json.load(resp)
    h = hashlib.sha256(r["content"].encode()).hexdigest()[:10]
    print(i, "prompt_n =", r["timings"]["prompt_n"], "cached =", r.get("tokens_cached"), h, flush=True)
