"""One measurement series against the local test llama-server.

Usage: measure.py <out.jsonl> <mode> <runs> [--temp T] [--seed S] [--cache] [--noise]

mode gen   : greedy/sampled generation of a short story (200 tokens), raw /completion
mode probe : prompt processing only, logprobs of the first token
Each run appends one JSON line (hash, tokens or logprobs) to out.jsonl.
Env PROMPT_FILE: use the raw prompt from this file instead of the built-in Qwen3 prompt
(e.g. prompts/gpt-oss.txt for the second model).
Env API=openai: talk to an OpenAI-compatible server (vLLM) via /v1/completions instead of
llama.cpp's /completion (gen mode only; --cache has no effect, the server decides about
prefix caching; MODEL_NAME sets the served model name).
Env LOGPROBS=1 (gen mode, both APIs): the logprob of every generated token (and the top-2
candidates) is recorded as well, and lp_hash covers text + logprobs (full precision).
llama.cpp: n_probs=2, probabilities before sampling, read from the logits after the token
was chosen (does not change the computation as long as backend sampling is off, the default).
Gen mode also records secs: wall time of the measured request (not of the strangers).
"""
import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request

LOGPROBS = os.environ.get("LOGPROBS") == "1"

OPENAI = os.environ.get("API") == "openai"
URL = f"http://127.0.0.1:{os.environ.get('PORT', '8091')}" + (
    "/v1/completions" if OPENAI else "/completion")


def chat(user):
    return (f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n")


PROMPT = chat("Write a short story about a lighthouse keeper who finds a strange "
              "message in a bottle. Be creative.")
if os.environ.get("PROMPT_FILE"):
    with open(os.environ["PROMPT_FILE"]) as f:
        PROMPT = f.read()
NOISE = [chat(p) for p in (
    "Explain how a refrigerator works in detail.",
    "List twenty European cities and one fact about each.",
    "Write a poem about autumn rain in a big city.",
)]


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def gen(prompt, a):
    t0 = time.monotonic()
    r = gen_request(prompt, a)
    r["secs"] = round(time.monotonic() - t0, 3)
    return r


def gen_request(prompt, a):
    if OPENAI:
        body = {"model": os.environ.get("MODEL_NAME", "Qwen3-8B-AWQ"), "prompt": prompt,
                "max_tokens": 200, "temperature": a.temp, "return_token_ids": True}
        if LOGPROBS:
            body["logprobs"] = 2
        if a.seed != -1:
            body["seed"] = a.seed
        c = post(body)["choices"][0]
        lp = c.get("logprobs") or {}
        return {"content": c["text"], "tokens": c.get("token_ids"),
                "logprobs": lp.get("token_logprobs"), "top2": lp.get("top_logprobs")}
    body = {"prompt": prompt, "n_predict": 200, "temperature": a.temp,
            "seed": a.seed, "cache_prompt": a.cache, "return_tokens": True}
    if LOGPROBS:
        body["n_probs"] = 2
    r = post(body)
    if LOGPROBS:  # same shape as the OpenAI branch: list of logprobs, list of {token: logprob}
        cp = r["completion_probabilities"]
        r["logprobs"] = [p["logprob"] for p in cp]
        r["top2"] = [{t["token"]: t["logprob"] for t in p["top_logprobs"]} for p in cp]
    return r


def one_run(a):
    if a.mode == "probe":
        r = post({"prompt": PROMPT, "n_predict": 1, "n_probs": 5, "temperature": 0,
                  "cache_prompt": False})
        top = r["completion_probabilities"][0]["top_logprobs"][:3]
        key = json.dumps([(t["token"], t["logprob"]) for t in top])
        return {"hash": hashlib.sha256(key.encode()).hexdigest()[:10], "top": top}

    if not a.noise:
        r = gen(PROMPT, a)
    else:
        res, barrier = {}, threading.Barrier(1 + len(NOISE))

        def target():
            barrier.wait()
            res["r"] = gen(PROMPT, a)

        def noise(p):
            barrier.wait()
            gen(p, a)

        ts = [threading.Thread(target=target)] + [
            threading.Thread(target=noise, args=(p,)) for p in NOISE]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        r = res["r"]
    out = {}
    if r.get("logprobs") is not None:
        key = json.dumps([r["content"], r["logprobs"]])
        out = {"lp_hash": hashlib.sha256(key.encode()).hexdigest()[:10],
               "logprobs": r["logprobs"], "top2": r.get("top2")}
    return out | {"hash": hashlib.sha256(r["content"].encode()).hexdigest()[:10],
            "prompt_n": r.get("timings", {}).get("prompt_n"),
            "tokens": r["tokens"], "content": r["content"], "secs": r["secs"]}


p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("mode", choices=["gen", "probe"])
p.add_argument("runs", type=int)
p.add_argument("--temp", type=float, default=0.0)
p.add_argument("--seed", type=int, default=-1)
p.add_argument("--cache", action="store_true")
p.add_argument("--noise", action="store_true")
a = p.parse_args()

with open(a.out, "a") as f:
    for _ in range(a.runs):
        f.write(json.dumps(one_run(a)) + "\n")
        f.flush()
