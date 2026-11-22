"""Repeat a raw greedy generation and print a hash per run (no chat template, no cache)."""
import hashlib
import json
import os
import sys
import urllib.request

URL = f"http://127.0.0.1:{os.environ.get('PORT', '8091')}/completion"
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
PROMPT = ("<|im_start|>user\nWrite a short story about a lighthouse keeper who finds a "
          "strange message in a bottle. Be creative.<|im_end|>\n<|im_start|>assistant\n"
          "<think>\n\n</think>\n\n")

for _ in range(RUNS):
    body = {"prompt": PROMPT, "n_predict": 200, "temperature": 0, "cache_prompt": False}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req))["content"]
    print(hashlib.sha256(out.encode()).hexdigest()[:10])
