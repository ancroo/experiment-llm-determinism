"""Summarize the measurement series in results/*.jsonl."""
import json
import sys
from collections import Counter
from pathlib import Path

DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "results")


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


for f in sorted(DIR.glob("*.jsonl")):
    rows = [json.loads(line) for line in f.open()]
    c = Counter(r["hash"] for r in rows)
    print(f"\n== {f.stem}: {len(rows)} runs, {len(c)} distinct  "
          f"(most frequent: {c.most_common(1)[0][1]}x)")
    if "tokens" not in rows[0]:
        print("   ", rows[0]["top"])
        continue
    ref = rows[0]
    divs = [first_diff(r["tokens"], ref["tokens"]) for r in rows[1:] if r["hash"] != ref["hash"]]
    if divs:
        print(f"   first divergent token vs run 1: min {min(divs)}, max {max(divs)}, "
              f"positions {sorted(Counter(divs).items())}")
    # one sample per distinct output, shown around the earliest divergence
    seen, i0 = set(), min(divs) if divs else 0
    k = first_diff(ref["content"], next((r["content"] for r in rows if r["hash"] != ref["hash"]),
                                        ref["content"]))
    for r in rows:
        if r["hash"] in seen:
            continue
        seen.add(r["hash"])
        print(f"   {r['hash']} ({c[r['hash']]}x): ...{r['content'][max(0, k-50):k+45]!r}")
