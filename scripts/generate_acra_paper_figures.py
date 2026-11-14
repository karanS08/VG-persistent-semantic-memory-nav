"""Two additional figures for acra_paper/main.tex, replacing text tables with
plots per the reviewer's editorial pass request ("convert suitable tables/results
into plots where possible", "increase visual content using existing results
rather than running new experiments"). Every number here is copied verbatim from
the already-published, already-audited tables in acra_paper/main.tex (Table
ablation-audited and Table memory-age-sweep) -- no new experiments, no
re-derivation from raw JSON, so there is no risk of drift from the audited text.

Run from the repo root:
    python scripts/generate_acra_paper_figures.py
Outputs PDFs to acra_paper/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys; sys.path.insert(0, "scripts")
from paper_figstyle import finalize, COL, FULL  # print-size normalisation

OUT_DIR = Path("acra_paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_memory_ablation():
    """Five conditions on the standard battery, STRICT audit, one grouped panel at single-column width (2026-10-03).
    Values: EXPERIMENT_RESULTS.md Test 0.2 (results/post_paper/rq1_trace.json)."""
    conditions = ["Memory\noff", "Full\nmemory", "No verif.\ndiscount", "Uniform\ndecay", "Persistent"]
    strict_success = [0.125, 0.375, 0.50, 0.625, 0.50]
    strict_spl = [0.062, 0.085, 0.131, 0.154, 0.275]
    fig, ax = plt.subplots(figsize=(3.375, 2.4))
    x = np.arange(len(conditions)); w = 0.38
    b1 = ax.bar(x - w / 2, strict_success, w, color="#b2182b", label="Success (strict audit)", zorder=3)   # red = audited, as in the verification figure
    b2 = ax.bar(x + w / 2, strict_spl, w, color="#8c8c8c", label="SPL", zorder=3)
    for bars, vals in ((b1, strict_success), (b2, strict_spl)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", rotation=90, fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=7, linespacing=1.05)
    ax.set_ylabel("Strict-audited score"); ax.set_ylim(0, 0.8)
    ax.grid(True, axis="y", alpha=0.2); ax.tick_params(axis="x", length=0, pad=2)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1, borderaxespad=0.1, columnspacing=1.2)
    finalize(fig, OUT_DIR / "fig_memory_ablation.pdf", 3.375, 2.4)
    print("wrote fig_memory_ablation.pdf (single column)")


def fig_memory_age_sweep():
    """Table memory-age-sweep: audited success + SPL vs. injected memory age.
    Categorical x-axis (evenly spaced) since the sweep values themselves are the
    conditions being compared, not a quantity to interpolate linearly."""
    labels = ["0h", "1h", "3h", "6h", "24h"]
    x = np.arange(len(labels))
    audited_success = [0.750, 0.375, 0.625, 0.625, 0.625]
    audited_spl = [0.393, 0.148, 0.342, 0.301, 0.465]
    false_positive_idx = {2, 3}  # 3h, 6h

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax2 = ax.twinx()

    l1, = ax.plot(x, audited_success, marker="o", markersize=7, color="#2166ac", label="Audited success")
    l2, = ax2.plot(x, audited_spl, marker="s", markersize=7, color="#b2182b", label="Audited SPL")

    for i in false_positive_idx:
        ax.plot(x[i], audited_success[i], marker="*", markersize=16,
                 color="#f4a582", markeredgecolor="#7f2704", zorder=5)
    ax.plot([], [], marker="*", markersize=10, color="#f4a582",
             markeredgecolor="#7f2704", linestyle="none", label="contains a false positive")

    ax.set_xlabel("Injected memory age")
    ax.set_ylabel("Audited success", color="#2166ac")
    ax2.set_ylabel("Audited SPL", color="#b2182b")
    ax.set_ylim(0, 1.0)
    ax2.set_ylim(0, 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.3, len(labels) - 0.7)
    ax.set_title("Memory-age sweep, persistent mode")
    ax.grid(True, alpha=0.2)
    fig.legend(handles=[l1, l2, ax.get_legend_handles_labels()[0][-1]],
               loc="upper center", ncol=1, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT_DIR / "fig_memory_age_sweep.pdf")
    plt.close(fig)
    print("wrote fig_memory_age_sweep.pdf")


def fig_verification_ablation():
    """Leave-one-gate-out: reported vs. audited success (n=10 each).
    2026-09-29: replaces paper/figures' version, which plotted 0.45 for the
    fresh-evidence guard ("midpoint of 0.4-0.5") -- not an observable value
    at n=10. Plotted: strict audit (4/10); lenient (5/10) is stated in the
    caption. Source: results/verification_leaveoneout_audit.md and
    data/results_postdraft/gt_audit_precision/precision_by_condition.csv."""
    conditions = ["Relevance\nremoved", "Multi-obs.\nremoved", "Lexical\nremoved", "Fresh-evid.\nguard removed"]
    raw = [0.5, 0.2, 0.9, 0.5]
    audited = [0.5, 0.2, 0.1, 0.4]
    x = np.arange(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(x - width / 2, raw, width, label="Reported by system", color="#92c5de")
    ax.bar(x + width / 2, audited, width, label="Audited (authors')", color="#b2182b")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Verification leave-one-gate-out: reported vs. audited")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.2)
    fig.savefig(OUT_DIR / "fig_verification_ablation.pdf")
    plt.close(fig)
    print("wrote fig_verification_ablation.pdf")


if __name__ == "__main__":
    fig_memory_ablation()
    fig_memory_age_sweep()
    fig_verification_ablation()
