"""Head-to-head on the two datasets where Laya claims a win over Jev: AG News and DAIR Emotion.

Laya's BENCHMARKS.md compares its own runs with Jev figures "third-party published, never
measured here", so the samples and prompts differ. This puts both on identical items.
Kept apart from the original 200-item set (items_manifest.json / calls.jsonl are unchanged).
Like that set, the item text is not committed: `sample` rebuilds it from the public
sources and checks it against results/ext_items_manifest.json.

  python ext_bench.py sample                 # -> results/ext_items.json (verified)
  python ext_bench.py run --arms jev         # Jev over OpenRouter (pennies)
  python ext_bench.py run --arms laya        # then laya-ml, laya-td, one at a time
  python ext_bench.py report
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402

OUT = bench.OUT
ITEMS = OUT / "ext_items.json"            # holds dataset text: not committed
MANIFEST = OUT / "ext_items_manifest.json"
LOG = OUT / "ext_calls.jsonl"
SEED = 20260926
PER_TASK = 100

AG = {"World": "world news, politics, international affairs",
      "Sports": "sports and athletes",
      "Business": "business, companies, markets and the economy",
      "Sci/Tech": "science and technology"}
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]  # dair-ai/emotion label order


def sample() -> None:
    rng = random.Random(SEED)
    items = []
    rows = bench._parquet("fancyzhx/ag_news", "default", "test")
    names = list(AG)
    for i, r in enumerate(rng.sample(rows, PER_TASK)):
        items.append({"id": f"agnews-{i:02d}", "task": "agnews", "type": "choice",
                      "state": {"article": r["text"]},
                      "instructions": "Which topic is `article` about?",
                      "options": AG, "gold": names[int(r["label"])]})
    rows = bench._parquet("dair-ai/emotion", "split", "test")
    for i, r in enumerate(rng.sample(rows, PER_TASK)):
        items.append({"id": f"emotion-{i:02d}", "task": "emotion", "type": "choice",
                      "state": {"text": r["text"]},
                      "instructions": "Which emotion does the author of `text` express?",
                      "options": {e: None for e in EMOTIONS}, "gold": EMOTIONS[int(r["label"])]})
    ITEMS.write_text(json.dumps({"seed": SEED, "items": items}, indent=1))
    print(f"{len(items)} items -> {ITEMS}")
    if MANIFEST.exists():
        verify(items)


def verify(items: list) -> None:
    want = {m["id"]: m for m in json.loads(MANIFEST.read_text())["items"]}
    bad = [it["id"] for it in items
           if it["id"] not in want
           or bench._sha(it["state"]) != want[it["id"]]["state_sha256"]
           or bench._sha(it["instructions"]) != want[it["id"]]["instructions_sha256"]
           or it["gold"] != want[it["id"]]["gold"]]
    if bad or len(items) != len(want):
        raise SystemExit(f"{len(bad)} items differ from the manifest (first: {bad[:5]})")
    print(f"all {len(items)} items match the published manifest")


def run(arm: str) -> None:
    items = json.loads(ITEMS.read_text())["items"]
    done = set()
    if LOG.exists():
        done = {json.loads(l)["item"] for l in LOG.read_text().splitlines()
                if l.strip() and json.loads(l)["arm"] == arm and not json.loads(l).get("error")}
    todo = [it for it in items if it["id"] not in done]
    print(f"{arm}: {len(todo)} items to run")
    for n, it in enumerate(todo, 1):
        # Jev is given option names without descriptions as `None`; send an empty
        # description instead so both systems see exactly the same criteria.
        it = dict(it, options={k: (v or k) for k, v in it["options"].items()})
        rec = {"item": it["id"], "task": it["task"], "arm": arm, "model": bench.ARMS[arm],
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            rec.update(bench.run_jev(it) if arm == "jev" else bench.run_laya(arm, it))
            p = rec["probs"]
            rec["answer"] = max(p, key=p.get)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:500]
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if n % 20 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}]", flush=True)


def report() -> None:
    items = {i["id"]: i for i in json.loads(ITEMS.read_text())["items"]}
    recs = {}
    for l in LOG.read_text().splitlines():
        r = json.loads(l)
        if not r.get("error"):
            recs[(r["item"], r["arm"])] = r
    arms = sorted({a for _, a in recs}, key=lambda a: list(bench.ARMS).index(a))
    out = {}
    for task in ("agnews", "emotion"):
        ids = [i for i in items if items[i]["task"] == task]
        print(task.upper())
        for arm in arms:
            rs = [(items[i], recs[(i, arm)]) for i in ids if (i, arm) in recs]
            correct = [float(r["answer"] == it["gold"]) for it, r in rs]
            confs = [bench_conf(r) for _, r in rs]
            lo, hi = boot(correct)
            m = {"n": len(rs), "accuracy": st.mean(correct), "ci": [lo, hi],
                 "mean_conf": st.mean(confs), "ece": ece(confs, correct),
                 "p50_ms": 1000 * st.median(r["wall_s"] for _, r in rs)}
            out.setdefault(task, {})[arm] = m
            print(f"  {arm:8} n={m['n']:3} acc {m['accuracy']:.3f} ({lo:.2f}-{hi:.2f})  "
                  f"conf {m['mean_conf']:.2f}  ECE {m['ece']:.3f}  p50 {m['p50_ms']:.0f} ms")
        # paired: items where exactly one of Jev / each Laya arm is right
        for arm in arms:
            if arm == "jev":
                continue
            both = [i for i in ids if (i, "jev") in recs and (i, arm) in recs]
            j = sum(recs[(i, "jev")]["answer"] == items[i]["gold"] and recs[(i, arm)]["answer"] != items[i]["gold"] for i in both)
            l = sum(recs[(i, arm)]["answer"] == items[i]["gold"] and recs[(i, "jev")]["answer"] != items[i]["gold"] for i in both)
            print(f"  paired vs jev: {arm} only-right {l}, jev only-right {j}")
    (OUT / "ext_summary.json").write_text(json.dumps(out, indent=1))


def bench_conf(r) -> float:
    p = r["probs"]
    s = sum(max(float(v), 0) for v in p.values()) or 1
    return max(max(float(v), 0) for v in p.values()) / s


def boot(xs, n=2000):
    rng = random.Random(0)
    m = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def ece(confs, correct, bins=10):
    total = 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(confs) if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if idx:
            total += len(idx) / len(confs) * abs(st.mean(confs[i] for i in idx) - st.mean(correct[i] for i in idx))
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "run", "report"])
    ap.add_argument("--arms", default="jev")
    a = ap.parse_args()
    if a.cmd == "sample":
        sample()
    elif a.cmd == "run":
        for arm in a.arms.split(","):
            run(arm)
    else:
        report()
