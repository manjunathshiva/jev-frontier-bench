"""Figure for the Laya article: Jev vs Laya on identical questions, six tasks.

Reads summary_with_laya.json (200-item set) and ext_summary.json (AG News, Emotion).
Laya = the default English checkpoint the Router picks. Palette = the dataviz skill's
first two categorical slots (validated all-pairs, light mode).

    python laya_chart.py   # -> results/fig4_laya_vs_jev.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent / "results"
JEV, LAYA = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"

s = json.loads((DATA / "summary_with_laya.json").read_text())["tasks"]
e = json.loads((DATA / "ext_summary.json").read_text())
rows = [  # label, jev, laya, n
    ("News topic (AG News, 4 options)", e["agnews"]["jev"]["accuracy"], e["agnews"]["laya"]["accuracy"], 100),
    ("Ambiguous statements (ChaosNLI)", s["ambig"]["jev"]["accuracy"], s["ambig"]["laya"]["accuracy"], 50),
    ("Emotion (DAIR, 6 options)", e["emotion"]["jev"]["accuracy"], e["emotion"]["laya"]["accuracy"], 100),
    ("Yes/no from a passage (BoolQ)", s["yesno"]["jev"]["accuracy"], s["yesno"]["laya"]["accuracy"], 50),
    ("Star rating 1–5 (Yelp)", s["rate"]["jev"]["accuracy"], s["rate"]["laya"]["accuracy"], 50),
    ("Bank intent (BANKING77, 78 options)", s["route"]["jev"]["accuracy"], s["route"]["laya"]["accuracy"], 50),
]

plt.rcParams.update({"font.family": "Arial", "font.size": 12})
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for y, (label, j, l, n) in enumerate(reversed(rows)):
    ax.plot([min(j, l), max(j, l)], [y, y], color=GRID, lw=6, solid_capstyle="round", zorder=1)
    ax.scatter([j], [y], s=150, color=JEV, marker="o", edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Jev 1.13 (TypeSafe, API)" if y == 0 else None)
    ax.scatter([l], [y], s=150, color=LAYA, marker="s", edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Laya (open source, on the Mac)" if y == 0 else None)
    # values sit outside the pair: the lower one on the left, the higher on the right
    lo, hi = sorted([(j, "Jev"), (l, "Laya")])
    ax.text(lo[0] - 0.018, y, f"{lo[0]:.2f}", ha="right", va="center", color=INK2, fontsize=10.5)
    ax.text(hi[0] + 0.018, y, f"{hi[0]:.2f}", ha="left", va="center", color=INK2, fontsize=10.5)
    ax.text(0.0, y + 0.33, f"{label}  ·  n={n}", ha="left", va="bottom", color=INK, fontsize=11)

ax.set_xlim(0, 1.05)
ax.set_ylim(-0.6, len(rows) - 0.2)
ax.set_yticks([])
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
ax.set_xlabel("Accuracy on identical questions", color=INK2, fontsize=12, labelpad=8)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(colors=INK2, length=0, pad=6)
fig.legend(loc="upper left", bbox_to_anchor=(0.035, 0.885), ncol=2, frameon=False, fontsize=11,
           labelcolor=INK2, handletextpad=0.4, columnspacing=1.8)

fig.text(0.04, 0.965, "Laya leads on two tasks, Jev on three, one is a tie", fontsize=17,
         fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.04, 0.915, "Same questions, same instructions, same options. Laya answers in 20–50 ms on a Mac, "
         "for free; Jev's API takes about 430 ms.", fontsize=11, color=INK2, ha="left", va="top")
fig.text(0.04, 0.025, "Laya = the default English checkpoint (laya 0.3.20). Jev 1.13 via OpenRouter. "
         "Differences on the 50-item tasks under ~15 points are within sampling noise.",
         fontsize=9, color=MUTED, ha="left", va="bottom")
fig.subplots_adjust(left=0.04, right=0.97, top=0.8, bottom=0.16)
out = DATA / "fig4_laya_vs_jev.png"
fig.savefig(out, facecolor=SURFACE)
print(out)
