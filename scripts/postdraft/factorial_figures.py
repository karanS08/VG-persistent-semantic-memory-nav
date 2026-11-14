"""Figure for Test 2.1 from data/results_postdraft/factorial_influence/factorial_episodes.csv
(produced by factorial_analysis.py). Offline."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/results_postdraft/factorial_influence"
rows = list(csv.DictReader(open(OUT / "factorial_episodes.csv")))
Q = ["a large window", "a wall-mounted radiator", "a cabinet", "a wall vent", "a bench"]
COND = {"A": ("A: reset mem / reset pos", "#999999"), "B": ("B: persist mem / reset pos", "#b2182b"),
        "C": ("C: reset mem / persist pos", "#4393c3"), "D": ("D: persist both", "#762a83")}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for k, (metric, ylab) in enumerate([("spl", "SPL"), ("steps", "steps to termination")]):
    ax = axes[k]
    for ci, c in enumerate("ABCD"):
        for qi, q in enumerate(Q):
            v = [float(r[metric]) for r in rows if r["cond"] == c and r["query"] == q]
            x = qi + (ci - 1.5) * 0.18
            ax.bar(x, np.mean(v), width=0.17, color=COND[c][1], alpha=0.55, label=COND[c][0] if qi == 0 else None)
            ep = [r for r in rows if r["cond"] == c and r["query"] == q]
            for r in ep:
                ax.scatter(x, float(r[metric]), s=10, color="black" if r["success"] == "1" else "red",
                           marker="o" if r["success"] == "1" else "x", zorder=3)
    ax.set_xticks(range(len(Q))); ax.set_xticklabels([q.replace("a ", "").replace("wall-mounted ", "") for q in Q], fontsize=8)
    ax.set_ylabel(ylab); ax.grid(True, axis="y", alpha=0.2)
    ax.axvline(0.5, color="k", lw=0.6, ls=":")
axes[0].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(1.1, -0.13), ncol=4)
axes[0].set_title("Per query (query 0 identical by design; x = failure)", fontsize=9)
ax = axes[2]
for ci, c in enumerate("ABCD"):
    sess = sorted({r["session"] for r in rows if r["cond"] == c})
    tot = [sum(float(r["wall_clock_s"]) for r in rows if r["session"] == s) for s in sess]
    ax.bar(ci, np.mean(tot), color=COND[c][1], alpha=0.55)
    for s, t in zip(sess, tot):
        ax.scatter(ci, t, color="black", s=12, zorder=3)
        ax.annotate(s, (ci + 0.08, t), fontsize=6)
ax.set_xticks(range(4)); ax.set_xticklabels(list("ABCD")); ax.set_ylabel("session wall-clock (s, 5 queries)")
ax.set_title("Whole-session time (dots = sessions)", fontsize=9); ax.grid(True, axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "fig_factorial.pdf")
print("wrote", OUT / "fig_factorial.pdf")
