"""Charts for the write-up, from results/summary.json and the logs.

  python charts.py   -> results/fig{1,2,3}_*.png

Emphasis encoding throughout: Jev in blue, Claude Fable 5.1 in orange, every other
model in gray, every mark labelled directly. Palette validated with the dataviz
skill's validator (light surface #fcfcfb).
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from analyze import OUT, jsd, load, norm  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
JEV, FABLE, OTHER = "#2a78d6", "#eb6834", "#a8a69e"
NAMES = {"jev": "Jev 1.13", "fable": "Claude Fable 5.1", "astra": "GPT-6 Astra",
         "kimi": "Kimi K3", "minimax": "MiniMax M3", "deepseek": "DeepSeek V4.1 Flash"}
COLOR = {"jev": JEV, "fable": FABLE}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def frame(ax, title, subtitle, source):
    ax.figure.text(0.035, 0.955, title, fontsize=17, fontweight="bold", color=INK, va="top")
    ax.figure.text(0.035, 0.895, subtitle, fontsize=11.5, color=INK2, va="top")
    ax.figure.text(0.035, 0.025, source, fontsize=9.5, color=MUTED, va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


SOURCE = "200 decisions (BANKING77, BoolQ, Yelp, ChaosNLI), one run via OpenRouter, 19 Sep 2026. Reasoning effort medium for the LLMs."


def fig_accuracy_cost(s):
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=160)
    fig.subplots_adjust(left=0.1, right=0.96, top=0.8, bottom=0.17)
    ov = s["overall"]
    offsets = {"jev": (10, -4), "fable": (-12, 8), "astra": (-12, 8), "kimi": (10, -14),
               "minimax": (12, -16), "deepseek": (12, 14)}
    align = {"fable": "right", "astra": "right"}
    for arm, m in ov.items():
        x, y = m["cost_per_1k_usd"], 100 * m["accuracy"]
        lo, hi = 100 * m["acc_ci"][0], 100 * m["acc_ci"][1]
        c = COLOR.get(arm, OTHER)
        ax.plot([x, x], [lo, hi], color=c, lw=2, solid_capstyle="round", alpha=0.55, zorder=2)
        ax.scatter([x], [y], s=110, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        dx, dy = offsets[arm]
        ax.annotate(f"{NAMES[arm]}\n{y:.1f}% · ${x:,.3f}/1k",
                    (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=align.get(arm, "left"), va="center", fontsize=10.5,
                    color=INK if arm in COLOR else INK2,
                    fontweight="bold" if arm in COLOR else "normal")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 40)
    ax.set_ylim(62, 92)
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(["$0.01", "$0.10", "$1", "$10"])
    ax.set_yticks(range(65, 95, 5))
    ax.set_yticklabels([f"{v}%" for v in range(65, 95, 5)])
    ax.set_xlabel("Cost per 1,000 decisions (log scale)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    frame(ax, "Only the two priciest models clearly beat Jev",
          "Accuracy vs cost. Vertical bars are 95% intervals. Kimi, MiniMax and DeepSeek are\n"
          "within noise of Jev on the same items; Claude Fable 5.1 (+11.5 pts) and GPT-6 Astra (+6.5) are not.",
          SOURCE)
    fig.savefig(OUT / "fig1_accuracy_vs_cost.png")
    plt.close(fig)


def human_baselines():
    items, _, _ = load()
    amb = [i for i in items.values() if i["task"] == "ambig"]
    onehot = [jsd({i["gold"]: 1.0}, i["human"]) for i in amb]
    uniform = [jsd({k: 1 / 3 for k in i["options"]}, i["human"]) for i in amb]
    return st.mean(onehot), st.mean(uniform)


def fig_human_agreement(s):
    rows = sorted(s["tasks"]["ambig"].items(), key=lambda kv: kv[1]["jsd_human"])
    onehot, uniform = human_baselines()
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=160)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.78, bottom=0.17)
    ys = list(range(len(rows)))[::-1]
    for y, (arm, m) in zip(ys, rows):
        c = COLOR.get(arm, OTHER)
        ax.barh(y, m["jsd_human"], height=0.56, color=c, zorder=2)
        ax.plot(m["jsd_ci"], [y, y], color=INK2, lw=1.2, zorder=3)
        ax.text(m["jsd_ci"][1] + 0.005, y, f"{m['jsd_human']:.3f}", va="center", fontsize=10.5,
                color=INK if arm in COLOR else INK2, fontweight="bold" if arm in COLOR else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([NAMES[a] for a, _ in rows], fontsize=11.5, color=INK)
    for val, label in ((uniform, "a blind guess:\n1/3 each"), (onehot, "100% sure of\nthe majority answer")):
        ax.axvline(val, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(val, len(rows) - 0.35, label, fontsize=9.5, color=MUTED, ha="center", va="bottom")
    ax.set_xlim(0, max(onehot, max(m["jsd_ci"][1] for _, m in rows)) * 1.18)
    ax.set_xlabel("Distance from the human split (Jensen-Shannon divergence; lower = closer)")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    frame(ax, "Only one model hedges the way people do",
          "50 ambiguous sentence pairs, each labelled by 100 people (ChaosNLI): how far each model's\n"
          "probabilities sit from the human split. Only two beat a blind 1/3 guess. Lines: 95% intervals.",
          SOURCE)
    fig.savefig(OUT / "fig2_human_agreement.png")
    plt.close(fig)


def fig_latency(s):
    rows = sorted(s["overall"].items(), key=lambda kv: kv[1]["latency_p50_s"])
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.76, bottom=0.19)
    ys = list(range(len(rows)))[::-1]
    for y, (arm, m) in zip(ys, rows):
        c = COLOR.get(arm, OTHER)
        ax.barh(y, m["latency_p50_s"], height=0.56, color=c, zorder=2)
        ax.plot([m["latency_p95_s"]] * 2, [y - 0.28, y + 0.28], color=INK2, lw=1.6, zorder=3)
        ax.text(m["latency_p95_s"] * 1.12, y, f"{m['latency_p50_s']:.2f} s  (slowest 5%: {m['latency_p95_s']:.1f} s)",
                va="center", fontsize=10.5, color=INK if arm in COLOR else INK2,
                fontweight="bold" if arm in COLOR else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([NAMES[a] for a, _ in rows], fontsize=11.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 200)
    ax.set_xticks([0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.3 s", "1 s", "3 s", "10 s", "30 s"])
    ax.set_xlabel("Time per decision, median (bar) and slowest 5% (tick), log scale")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    frame(ax, "Jev answers in under half a second",
          "Wall time per decision from India through OpenRouter, network included.",
          SOURCE)
    fig.savefig(OUT / "fig3_latency.png")
    plt.close(fig)


if __name__ == "__main__":
    summary = json.loads((OUT / "summary.json").read_text())
    fig_accuracy_cost(summary)
    fig_human_agreement(summary)
    fig_latency(summary)
    oh, un = human_baselines()
    print(f"baselines: one-hot majority {oh:.3f}, uniform {un:.3f}")
    print("wrote", *sorted(p.name for p in OUT.glob("fig*.png")))
