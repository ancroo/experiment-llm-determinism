"""Summary of small_model.sh (Qwen3-1.7B, FP16): results/small_<setup>_<series>.jsonl.

Per setup and series: distinct texts and logprob sets, the sequence of texts and of logprob
sets as letters (shared by all series of a setup), the largest logprob shift against request 1
of the "run" series (over the tokens both runs share), and the median time per request.
Then: are the fresh-start sequences identical, does the no-logprobs control match, and how
far do the setups agree with each other (first request after a fresh start; settled state).
Usage: small_model_summary.py [results_dir]
"""
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
SERIES = ["run", "seq_1", "seq_2", "seq_3", "nolp", "fresh", "noise", "slots4"]
ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def load(name):
    p = DIR / f"{name}.jsonl"
    return [json.loads(line) for line in p.open()] if p.exists() else None


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def max_shift(ref, r):
    """Largest |delta logprob| over the tokens both runs generated identically."""
    n = common_prefix(ref["tokens"], r["tokens"])
    return max((abs(x - y) for x, y in zip(ref["logprobs"][:n], r["logprobs"][:n])), default=0.0)


def letters_line(seq, lab):
    s = "".join(lab(h) for h in seq)
    return " ".join(s[i:i + 50] for i in range(0, len(s), 50))


setups = sorted({p.stem[len("small_"):].rsplit("_", 1)[0].removesuffix("_seq")
                 for p in DIR.glob("small_*.jsonl")})
firsts, settled = {}, {}
for setup in setups:
    print(f"\n=== {setup}")
    # Letters per distinct text, and separately per distinct logprob set (lower-case line);
    # beyond 62 distinct values, every further one is shown as "*".
    letters, lp_letters = {}, {}
    lab = lambda h: letters.setdefault(h, ABC[len(letters)] if len(letters) < len(ABC) else "*")
    lp_lab = lambda h: lp_letters.setdefault(
        h, ABC[len(lp_letters)] if len(lp_letters) < len(ABC) else "*")
    data = {s: load(f"small_{setup}_{s}") for s in SERIES}
    base = (data["run"] or [None])[0]  # reference for all series: request 1 of "run"
    for s, rows in data.items():
        if not rows:
            continue
        texts = Counter(r["hash"] for r in rows)
        lps = {r.get("lp_hash") for r in rows} - {None}
        line = (f"{s:6} {len(rows):3} runs: {len(texts):2} texts (top {texts.most_common(1)[0][1]}x)")
        if lps:
            ref = base or rows[0]
            line += (f", {len(lps):2} logprob sets, max |dlogprob| vs run#1 "
                     f"{max(max_shift(ref, r) for r in rows):.3g}")
        line += f", median {statistics.median(r['secs'] for r in rows):.2f} s"
        print(line)
        print("  text  ", letters_line([r["hash"] for r in rows], lab))
        if lps:
            print("  lp    ", letters_line([r["lp_hash"] for r in rows], lp_lab))
        if len(rows) >= 40:
            tail = rows[len(rows) // 2:]
            print(f"         second half: {len({r['hash'] for r in tail})} texts, "
                  f"{len({r.get('lp_hash') for r in tail})} logprob sets")
    seqs = [data[f"seq_{i}"] for i in (1, 2, 3) if data[f"seq_{i}"]]
    if seqs:
        same_t = all([r["hash"] for r in s] == [r["hash"] for r in seqs[0]] for s in seqs)
        same_lp = all([r["lp_hash"] for r in s] == [r["lp_hash"] for r in seqs[0]] for s in seqs)
        print(f"seq identical across fresh starts: texts {same_t}, logprobs {same_lp}")
        firsts[setup] = seqs[0][0]
        if data["run"]:
            settled[setup] = data["run"][-1]
        if data["run"]:
            k = len(seqs[0])
            print(f"run[:{k}] equals seq_1 (texts): {[r['hash'] for r in data['run'][:k]] == [r['hash'] for r in seqs[0]]}")
    if data["nolp"] and seqs:
        k = len(data["nolp"])
        print(f"control nolp == seq_1[:{k}] (texts): "
              f"{[r['hash'] for r in data['nolp']] == [r['hash'] for r in seqs[0][:k]]}")
    if base and base.get("top2"):
        gaps = []
        for i, t in enumerate(base["top2"]):
            v = sorted(t.values(), reverse=True)
            if len(v) >= 2:
                gaps.append((math.exp(v[0]) - math.exp(v[1]), i, t))
        g, i, t = min(gaps)
        print(f"closest call in run 1: token {i}, gap {100 * g:.2f} pp, "
              + ", ".join(f"{k!r} {100 * math.exp(v):.2f} %" for k, v in t.items()))

for title, req in (("first request after a fresh start", firsts),
                   ("last request of run (settled state)", settled)):
    if len(req) < 2:
        continue
    print(f"\n=== {title}, across setups")
    names = sorted(req)
    for a in names:
        for b in names:
            if a < b:
                n = common_prefix(req[a]["tokens"], req[b]["tokens"])
                same = req[a]["hash"] == req[b]["hash"]
                d = max_shift(req[a], req[b])
                print(f"{a:12} vs {b:12}: same text {same}, common tokens {n}, "
                      f"max |dlogprob| on them {d:.3g}")
