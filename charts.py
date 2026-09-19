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


def fig_accuracy_cost_modern(s):
    """The same data as fig 1, styled as a cover: dark surface, glow, one callout.
    Colors are the dark-mode steps of the same blue/orange, validated on #101218."""
    BG, INK_D, INK2_D, MUTED_D, GRID_D = "#101218", "#ffffff", "#c3c2b7", "#8f8d86", "#232733"
    JEV_D, FABLE_D, OTHER_D = "#3987e5", "#d95926", "#8f8d86"
    col = {"jev": JEV_D, "fable": FABLE_D}
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=BG)
    ax = fig.add_axes([0.075, 0.13, 0.885, 0.6], facecolor=BG)
    ov = s["overall"]
    place = {"jev": (18, -2, "left"), "fable": (-22, 40, "right"), "astra": (18, -32, "left"),
             "kimi": (-18, -30, "right"), "minimax": (18, -24, "left"), "deepseek": (18, 22, "left")}
    for arm, m in ov.items():
        x, y = m["cost_per_1k_usd"], 100 * m["accuracy"]
        lo, hi = 100 * m["acc_ci"][0], 100 * m["acc_ci"][1]
        c = col.get(arm, OTHER_D)
        ax.plot([x, x], [lo, hi], color=c, lw=3, alpha=0.35, solid_capstyle="round", zorder=2)
        if arm in col:  # soft glow: stacked translucent discs
            for size, a in ((2600, 0.05), (1500, 0.08), (800, 0.14)):
                ax.scatter([x], [y], s=size, color=c, alpha=a, linewidth=0, zorder=3)
        ax.scatter([x], [y], s=260 if arm in col else 170, color=c, edgecolor=BG, linewidth=2.5, zorder=4)
        dx, dy, ha = place[arm]
        ax.annotate(NAMES[arm], (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="bottom",
                    fontsize=17 if arm in col else 14, fontweight="bold" if arm in col else "normal",
                    color=INK_D if arm in col else INK2_D, zorder=5)
        ax.annotate(f"{y:.1f}%  ·  ${x:,.3f} per 1k", (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="top", fontsize=13 if arm in col else 12,
                    color=c if arm in col else MUTED_D, zorder=5)
    # Callout: what the extra money buys.
    j, f = ov["jev"], ov["fable"]
    ax.annotate("", xy=(f["cost_per_1k_usd"] * 0.8, 100 * f["accuracy"] - 0.4),
                xytext=(j["cost_per_1k_usd"] * 1.35, 100 * j["accuracy"] + 0.5),
                arrowprops=dict(arrowstyle="-|>", color=INK2_D, lw=1.4, alpha=0.7,
                                connectionstyle="arc3,rad=-0.18", mutation_scale=18), zorder=1)
    ratio = f["cost_per_1k_usd"] / j["cost_per_1k_usd"]
    gain = 100 * (f["accuracy"] - j["accuracy"])
    ax.text(0.33, 85.8, f"{ratio:.0f}× the price for +{gain:.1f} points", fontsize=21,
            fontweight="bold", fontfamily="Avenir Next", color=INK_D, ha="center", va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.012, 45)
    ax.set_ylim(63, 91)
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(["$0.01", "$0.10", "$1", "$10"], fontsize=13, color=MUTED_D)
    ax.set_yticks(range(65, 95, 5))
    ax.set_yticklabels([f"{v}%" for v in range(65, 95, 5)], fontsize=13, color=MUTED_D)
    ax.set_xlabel("Cost per 1,000 decisions (log scale)", fontsize=13, color=INK2_D, labelpad=10)
    ax.set_ylabel("Accuracy on 200 decisions", fontsize=13, color=INK2_D, labelpad=10)
    ax.grid(True, color=GRID_D, lw=1)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.075, 0.93, "Only the two priciest models clearly beat Jev", fontsize=34,
             fontweight="bold", fontfamily="Avenir Next", color=INK_D, va="top")
    fig.text(0.075, 0.845, "TypeSafe's Jev vs five frontier LLMs with reasoning on. Kimi K3, MiniMax M3 and "
             "DeepSeek V4.1 Flash are within noise of Jev;\nClaude Fable 5.1 and GPT-6 Astra are clearly ahead. "
             "Vertical bars: 95% intervals.", fontsize=15, color=INK2_D, va="top", linespacing=1.5)
    fig.text(0.075, 0.035, SOURCE, fontsize=11, color=MUTED_D, va="bottom")
    fig.savefig(OUT / "fig1_accuracy_vs_cost_modern.png", facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    summary = json.loads((OUT / "summary.json").read_text())
    fig_accuracy_cost(summary)
    fig_human_agreement(summary)
    fig_latency(summary)
    fig_accuracy_cost_modern(summary)
    oh, un = human_baselines()
    print(f"baselines: one-hot majority {oh:.3f}, uniform {un:.3f}")
    print("wrote", *sorted(p.name for p in OUT.glob("fig*.png")))
