"""Determinism test against a local llama-server (temperature 0).

Scenarios:
  A  sequential, prompt cache off      -> run-to-run determinism
  B  sequential, prompt cache on       -> effect of prefix caching / request slicing
  C  concurrent with "noise" requests  -> effect of batching with strangers
"""
import hashlib
import json
import sys
import threading
import urllib.request
from collections import Counter

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8091/v1/chat/completions"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MAX_TOKENS = 200

PROMPT = ("Write a short story about a lighthouse keeper who finds a strange "
          "message in a bottle. Be creative.")
NOISE = [
    "Explain how a refrigerator works in detail.",
    "List twenty European cities and one fact about each.",
    "Write a poem about autumn rain in a big city.",
]


def ask(prompt, cache_prompt=True):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": MAX_TOKENS,
        "cache_prompt": cache_prompt,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def report(name, outputs, reference):
    c = Counter(h(o) for o in outputs)
    print(f"\n== {name}: {len(outputs)} runs, {len(c)} distinct outputs")
    for k, v in c.most_common():
        mark = " (= reference)" if k == h(reference) else ""
        print(f"   {k}: {v}x{mark}")
    for o in outputs:
        if o != reference:
            i = next((j for j, (a, b) in enumerate(zip(o, reference)) if a != b),
                     min(len(o), len(reference)))
            print(f"   first divergence at char {i}:")
            print(f"     ref: ...{reference[max(0, i-40):i+40]!r}")
            print(f"     got: ...{o[max(0, i-40):i+40]!r}")
            break


def concurrent_run():
    """Target request plus noise requests started at the same moment."""
    result = {}
    barrier = threading.Barrier(1 + len(NOISE))

    def target():
        barrier.wait()
        result["out"] = ask(PROMPT, cache_prompt=False)

    def noise(p):
        barrier.wait()
        ask(p, cache_prompt=False)

    ts = [threading.Thread(target=target)] + [
        threading.Thread(target=noise, args=(p,)) for p in NOISE]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return result["out"]


reference = ask(PROMPT, cache_prompt=False)
print("Reference output (first 300 chars):\n", reference[:300])

report("A sequential, no cache", [ask(PROMPT, False) for _ in range(N)], reference)
report("B sequential, cache on", [ask(PROMPT, True) for _ in range(N)], reference)
report("C concurrent with noise", [concurrent_run() for _ in range(N)], reference)
