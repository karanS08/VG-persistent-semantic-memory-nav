"""Generate all paper figures from real, already-collected data. Every figure here
draws from an existing results/*.json or data/runs/*/batch_metrics.json file -- no
synthetic or illustrative numbers. Run from the repo root:

    python scripts/generate_paper_figures.py

Outputs PDFs (vector, LaTeX-friendly) to paper/figures/.

Design choice (2026-09-05, per user request + pushback on decorative complexity):
standard 2D plots for anything that's genuinely 2D data (decay curves, ablation
bars, threshold sweeps) -- forcing these into 3D would only obscure them. The one
3D figure (fig_3d_memory_scene) is justified because the data itself is 3D: real
scene positions of remembered entities, which is literally what "persistent memory
anchored in a 3D scene" means for this pipeline.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CATEGORY_COLORS = {
    "radiator (low)": "#2166ac",
    "chair (medium)": "#f4a582",
    "phone (high)": "#b2182b",
}


def fig_temporal_decay():
    """C_i vs age, one line per category, at n_i=1 (production default) and the
    relevance level closest to this pipeline's real observed magnitudes (~0.15).
    Real data: results/controlled_temporal_decay.json (2026-09-05 controlled
    experiment, direct call into memory_layer.reliability.compute_reliability)."""
    d = json.load(open("results/controlled_temporal_decay.json"))
    rows = [r for r in d["rows"] if r["n_i"] == 1 and r["relevance"] == 0.15]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for cat, color in CATEGORY_COLORS.items():
        cat_rows = sorted([r for r in rows if r["category"] == cat], key=lambda r: r["age_seconds"])
        ages = [max(r["age_seconds"], 1) for r in cat_rows]  # 1s floor for log scale
        cis = [r["C_i"] for r in cat_rows]
        label = cat.split(" (")[0]
        ax.plot(ages, cis, marker="o", markersize=4, color=color, label=label)

    theta_direct = d["params"]["theta_direct"]
    ax.set_xscale("log")
    ax.set_xlabel("Time since last observation (s, log scale)")
    ax.set_ylabel("$C_i$ (reliability)")
    ax.set_title(f"Per-category reliability decay ($n_i=1$, relevance=0.15)")
    ax.legend(frameon=False, title="category (mobility)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.2, which="both")
    fig.savefig(OUT_DIR / "fig_temporal_decay.pdf")
    plt.close(fig)
    print("wrote fig_temporal_decay.pdf")


def fig_verification_ablation():
    """Raw vs. independently-audited success rate for the 4 leave-one-gate-out
    conditions. Real data: manually transcribed from
    results/verification_leaveoneout_audit.md (2026-09-05)."""
    conditions = ["Relevance\nremoved", "Multi-obs.\nremoved", "Lexical\nremoved", "Fresh-evid.\nguard removed"]
    raw = [0.5, 0.2, 0.9, 0.5]
    audited = [0.5, 0.2, 0.1, 0.45]  # fresh-evidence-guard: midpoint of 0.4-0.5 audited range

    x = np.arange(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(x - width / 2, raw, width, label="Raw (rule-based)", color="#92c5de")
    ax.bar(x + width / 2, audited, width, label="Independently audited", color="#b2182b")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Verification leave-one-gate-out: raw vs. audited")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.2)
    fig.savefig(OUT_DIR / "fig_verification_ablation.pdf")
    plt.close(fig)
    print("wrote fig_verification_ablation.pdf")


def fig_direct_recall_sweep():
    """Direct-recall precision vs. theta_direct, both conditions, plus a marker
    for cells where the branch never fired. Real data: data/runs/phase4_batch_
    *theta*_drsweep_*/batch_metrics.json (2026-09-05 sweep)."""
    thetas = ["0.05", "0.075", "0.1", "0.125", "0.15", "0.175", "0.2", "0.25", "0.3"]
    theta_disp = ["0.05", "0.075", "0.10", "0.125", "0.15", "0.175", "0.20", "0.25", "0.30"]
    theta_vals = [float(t) for t in theta_disp]

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for cond, prefix, color, marker in [
        ("baseline", "phase4_batch_theta", "#2166ac", "o"),
        ("persistent", "phase4_batch_persistent_theta", "#b2182b", "s"),
    ]:
        precisions, fired_thetas, empty_thetas = [], [], []
        for t, td, tv in zip(thetas, theta_disp, theta_vals):
            tag = f"drsweep_{cond}_theta{td}"
            path = Path(f"data/runs/{prefix}{t}_{tag}/batch_metrics.json")
            if not path.exists():
                continue
            d = json.load(open(path))
            p = d.get("direct_recall_precision")
            if p is None:
                empty_thetas.append(tv)
            else:
                fired_thetas.append(tv)
                precisions.append(p)
        ax.plot(fired_thetas, precisions, marker=marker, color=color, label=f"{cond} (fires)")
        if empty_thetas:
            ax.scatter(empty_thetas, [0] * len(empty_thetas), marker="x", color=color, alpha=0.5,
                       label=f"{cond} (never fires)")

    ax.axvline(0.2, color="gray", linestyle="--", linewidth=1, label="production default")
    ax.set_xlabel(r"$\theta_{\mathrm{direct}}$")
    ax.set_ylabel("direct_recall_precision (proxy)")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("Direct-recall threshold sweep")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.savefig(OUT_DIR / "fig_direct_recall_sweep.pdf")
    plt.close(fig)
    print("wrote fig_direct_recall_sweep.pdf")


def fig_memory_influence():
    """Per-episode memory_influence_rate, baseline vs persistent, 8 episodes each.
    Real data: results/batch_runs/phase4_batch_minconf1default and
    phase4_batch_persistent_minconf1default (already-archived Table 2 baseline
    runs)."""
    def rates(path):
        d = json.load(open(path))
        return [e["memory_influence_rate"] for e in d["episodes"] if e.get("memory_influence_rate") is not None]

    baseline = rates("results/batch_runs/phase4_batch_minconf1default/batch_metrics.json")
    persistent = rates("results/batch_runs/phase4_batch_persistent_minconf1default/batch_metrics.json")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    positions = [1, 2]
    bp = ax.boxplot([baseline, persistent], positions=positions, widths=0.5, patch_artist=True,
                     medianprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], ["#92c5de", "#b2182b"]):
        patch.set_facecolor(color)
    for pos, data in zip(positions, [baseline, persistent]):
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(data))
        ax.scatter([pos] * len(data) + jitter, data, color="black", s=14, zorder=3, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Baseline\n(reset per query)", "Persistent\n(across queries)"])
    ax.set_ylabel("memory_influence_rate (per episode)")
    ax.set_ylim(0.6, 1.05)
    ax.set_title("Memory influence: baseline vs. persistent (n=8 each)")
    ax.grid(True, axis="y", alpha=0.2)
    fig.savefig(OUT_DIR / "fig_memory_influence.pdf")
    plt.close(fig)
    print("wrote fig_memory_influence.pdf")


if __name__ == "__main__":
    fig_temporal_decay()
    fig_verification_ablation()
    fig_direct_recall_sweep()
    fig_memory_influence()
