"""Metrics for the Jev benchmark, computed only from the logs.

  python analyze.py            # prints tables, writes paper/jev_bench/summary.json

Per task: accuracy with a bootstrap 95% CI, plus
  yes/no   AUROC on P(yes), Brier
  rating   mean absolute error in stars, Spearman (expected stars vs gold)
  ambig    Jensen-Shannon divergence from the 100-person label distribution
Pooled over all tasks: calibration of each model's confidence in its own pick
(ECE, 10 equal-width bins; Brier), latency p50/p95, cost per 1,000 decisions,
median reasoning tokens, failures.

A model's confidence is the probability it gave its own pick: the maximum of its
normalized distribution, or, for routing, where the LLMs return one answer plus a
confidence, that number. Jev's is the maximum of its `probabilities`, not its
`confidence` field, which TypeSafe says is a summary statistic, not a probability.
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results"
ARMS = ["jev", "fable", "astra", "kimi", "minimax", "deepseek"]
LAYA_ARMS = ["laya", "laya-ml", "laya-td"]  # local, run 2026-09-26 (--with-laya)
TASKS = ["route", "yesno", "rate", "ambig"]
rng = random.Random(0)


def load():
    # The manifest has everything scoring needs (task, options, answer, human split)
    # and none of the dataset text, so the published logs re-score without it.
    items = {i["id"]: i for i in json.loads((OUT / "items_manifest.json").read_text())["items"]}
    recs = [json.loads(l) for l in (OUT / "calls.jsonl").read_text().splitlines() if l.strip()]
    ok = {}
    fails = {a: 0 for a in ARMS + LAYA_ARMS}
    for r in recs:
        if r.get("error"):
            fails[r["arm"]] += 1
        else:
            ok[(r["item"], r["arm"])] = r
    # A failure later retried successfully is not a failure of the final run.
    for r in recs:
        if r.get("error") and (r["item"], r["arm"]) in ok:
            fails[r["arm"]] -= 1
    return items, ok, fails


def norm(p: dict) -> dict:
    p = {k: max(float(v), 0.0) for k, v in p.items()}
    s = sum(p.values())
    return {k: v / s for k, v in p.items()} if s > 0 else p


def pick_conf(rec) -> tuple[str, float]:
    p = rec["probs"]
    if len(p) == 1:  # compact answer: one option and its stated confidence
        (k, v), = p.items()
        return k, min(max(float(v), 0.0), 1.0)
    p = norm(p)
    k = max(p, key=p.get)
    return k, p[k]


def boot_ci(xs, n=2000):
    if not xs:
        return (float("nan"),) * 2
    means = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def jsd(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0) + q.get(k, 0)) for k in keys}
    def kl(a):
        return sum(a.get(k, 0) * math.log2(a.get(k, 0) / m[k]) for k in keys if a.get(k, 0) > 0)
    return 0.5 * kl(p) + 0.5 * kl(q)


def ece(confs, correct, bins=10):
    total, n = 0.0, len(confs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            total += len(idx) / n * abs(st.mean(confs[i] for i in idx) - st.mean(correct[i] for i in idx))
    return total


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def main():
    items, ok, fails = load()
    summary = {"tasks": {}, "overall": {}}
    for task in TASKS:
        ids = [i for i in items if items[i]["task"] == task]
        summary["tasks"][task] = {}
        for arm in ARMS:
            recs = [(items[i], ok[(i, arm)]) for i in ids if (i, arm) in ok]
            if not recs:
                continue
            correct = [float(pick_conf(r)[0] == it["gold"]) for it, r in recs]
            m = {"n": len(recs), "accuracy": st.mean(correct), "acc_ci": boot_ci(correct)}
            if task == "yesno":
                py = [norm(r["probs"]).get("yes", 0.0) for _, r in recs]
                gold = [it["gold"] == "yes" for it, _ in recs]
                m["auroc"] = auroc(py, gold)
                m["brier"] = st.mean((p - g) ** 2 for p, g in zip(py, gold))
            if task == "rate":
                pred = [int(pick_conf(r)[0]) for _, r in recs]
                gold = [int(it["gold"]) for it, _ in recs]
                expected = [sum(int(k) * v for k, v in norm(r["probs"]).items()) for _, r in recs]
                m["mae_stars"] = st.mean(abs(p - g) for p, g in zip(pred, gold))
                m["spearman"] = spearman(expected, gold)
            if task == "ambig":
                d = [jsd(norm(r["probs"]), it["human"]) for it, r in recs]
                m["jsd_human"] = st.mean(d)
                m["jsd_ci"] = boot_ci(d)
            summary["tasks"][task][arm] = m

    for arm in ARMS:
        recs = [(items[i], r) for (i, a), r in ok.items() if a == arm]
        if not recs:
            continue
        confs, correct = [], []
        for it, r in recs:
            k, c = pick_conf(r)
            confs.append(c)
            correct.append(float(k == it["gold"]))
        wall = [r["wall_s"] for _, r in recs]
        cost = [r.get("cost") or 0.0 for _, r in recs]
        rtok = [((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                for _, r in recs]
        summary["overall"][arm] = {
            "n": len(recs), "failures": fails[arm],
            "accuracy": st.mean(correct), "acc_ci": boot_ci(correct),
            "mean_confidence": st.mean(confs),
            "ece": ece(confs, correct),
            "brier_top": st.mean((c - y) ** 2 for c, y in zip(confs, correct)),
            "latency_p50_s": st.median(wall), "latency_p95_s": pct(wall, 0.95),
            "cost_per_1k_usd": 1000 * st.mean(cost), "total_cost_usd": sum(cost),
            "reasoning_tokens_median": st.median(rtok), "reasoning_tokens_mean": st.mean(rtok),
        }

    (OUT / SUMMARY).write_text(json.dumps(summary, indent=1))

    print("OVERALL (all tasks pooled)")
    print(f"{'arm':9} {'n':>4} {'fail':>4} {'acc':>6} {'95% CI':>13} {'conf':>5} {'ECE':>5} "
          f"{'p50 s':>6} {'p95 s':>6} {'$/1k':>7} {'rsn tok':>7}")
    for arm, m in summary["overall"].items():
        print(f"{arm:9} {m['n']:4} {m['failures']:4} {m['accuracy']:6.3f} "
              f"{m['acc_ci'][0]:6.3f}-{m['acc_ci'][1]:.3f} {m['mean_confidence']:5.2f} {m['ece']:5.3f} "
              f"{m['latency_p50_s']:6.2f} {m['latency_p95_s']:6.2f} {m['cost_per_1k_usd']:7.3f} "
              f"{m['reasoning_tokens_mean']:7.0f}")
    for task in TASKS:
        print(f"\n{task.upper()}")
        for arm, m in summary["tasks"][task].items():
            extra = ""
            if task == "yesno":
                extra = f"AUROC {m['auroc']:.3f}  Brier {m['brier']:.3f}"
            elif task == "rate":
                extra = f"MAE {m['mae_stars']:.2f} stars  Spearman {m['spearman']:.3f}"
            elif task == "ambig":
                extra = f"JSD vs humans {m['jsd_human']:.3f} ({m['jsd_ci'][0]:.3f}-{m['jsd_ci'][1]:.3f})"
            print(f"  {arm:9} n={m['n']:3} acc {m['accuracy']:.3f} "
                  f"({m['acc_ci'][0]:.2f}-{m['acc_ci'][1]:.2f})  {extra}")


SUMMARY = "summary.json"

if __name__ == "__main__":
    import sys
    if "--with-laya" in sys.argv:
        # A separate file, so summary.json (six API arms) stays byte-identical:
        # adding arms changes the bootstrap draw order and would shift the other arms' CIs.
        ARMS = ARMS + LAYA_ARMS
        SUMMARY = "summary_with_laya.json"
    main()
