"""2026-09-29: figures for the Post-Paper Test Plan results, into acra_paper/figures/.

fig_factorial.pdf        2x2 memory x session factorial (Test 2.1), queries 2-5, n=12/cell
fig_verification_ablation.pdf  leave-one-gate-out + all-gates-removed, reported vs audited
fig_influence_decomposition.pdf  what "memory-influenced" decisions are (Test 2.2 part 1)

Palette: #2166ac / #b2182b (validated: CVD dE 21.1, normal 28.7) + neutral grey
for "reported"; every series also has a marker/label so identity is not colour-only.
Sources: results/post_paper/factorial_summary.json,
results/verification_leaveoneout_audit.md + results/full_no_verification_audit.md,
data/results_postdraft/factorial_influence/influence_probe_decomposition.json.
Run: ~/miniconda3/envs/rpv_mem/bin/python scripts/generate_postplan_figures.py
"""
import json
import random
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, "scripts")
from paper_figstyle import finalize, COL, FULL  # print-size normalisation

OUT = Path("acra_paper/figures")
BLUE, RED, GREY = "#2166ac", "#b2182b", "#8c8c8c"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.2})


def boot_ci(xs, n=10000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choice(xs) for _ in xs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def fig_factorial():
    S = json.load(open("results/post_paper/factorial_summary.json"))["conditions"]
    eps = {c: [e for e in S[c]["episodes"] if not e["key"].endswith(".0")] for c in "ABCD"}
    metrics = [("(a) Strict-audited SPL \u2191", lambda e: e["spl"] * e["strict"], (0, 1.05), "{:.2f}"),
               ("(b) Steps per query \u2193", lambda e: e["steps"], (0, 520), "{:.0f}")]
    # (memory reset -> memory kept) for each session level; x offset keeps CIs apart
    lines = [("Session reset", "A", "B", BLUE, "o", -0.07),
             ("Session kept", "C", "D", RED, "s", 0.07)]
    fig, axes = plt.subplots(1, 2)
    for ax, (name, f, ylim, fmt) in zip(axes, metrics):
        for label, c0, c1, col, mk, dx in lines:
            ys, lo, hi = [], [], []
            # 2026-11-03: bars = range of the three session means (sessions, not
            # episodes, are the independent unit); each session also drawn as a dot.
            for j, c in enumerate((c0, c1)):
                by = {}
                for e in eps[c]:
                    by.setdefault(e["run"], []).append(f(e))
                sm = [st.mean(v) for v in by.values()]
                assert len(sm) == 3, (c, len(sm))
                m = st.mean(sm)
                ys.append(m); lo.append(m - min(sm)); hi.append(max(sm) - m)
                ax.scatter([j + dx + 0.09 * (1 if dx > 0 else -1)] * len(sm), sm, s=9, facecolors="none",
                           edgecolors=col, linewidths=0.8, zorder=3)
            ax.errorbar([dx, 1 + dx], ys, yerr=[lo, hi], color=col, marker=mk, markersize=4.5,
                        lw=1.4, elinewidth=1, capsize=2, label=label)
            if c1 == "B":  # the headline cell: memory kept, session reset
                ax.annotate(fmt.format(ys[1]), (1 + dx, ys[1]), xytext=(-9, 0), textcoords="offset points",
                            ha="right", va="center", color=col, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Memory\nreset", "Memory\nkept"])
        ax.set_xlim(-0.4, 1.4); ax.set_ylim(*ylim); ax.set_title(name)
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.5)
    finalize(fig, OUT / "fig_factorial.pdf", COL, 2.2, rect=(0, 0, 1, 0.9))
    print("wrote fig_factorial.pdf")


def fig_verification_ablation():
    # n=10 each; fresh-evidence guard: 4/10 strict (5/10 lenient, stated in caption)
    conds = ["Relevance", "Multiple\nsightings", "Lexical\nmatch", "Fresh\nevidence", "All four"]   # names as in Section 3.3
    reported = [0.5, 0.2, 0.9, 0.5, 1.0]
    audited = [0.5, 0.2, 0.1, 0.4, 0.0]
    x = range(len(conds)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    ax.bar([i - w / 2 for i in x], reported, w, color=GREY, label="Reported")
    ax.bar([i + w / 2 for i in x], audited, w, color=RED, label="Audited")
    # 2026-09-29: second run of every condition as dots (reported: grey ring,
    # strict audited: red ring), from results/post_paper/new_runs_summary.json
    G = json.load(open("results/post_paper/new_runs_summary.json"))["groups"]
    rep2 = [G[k] for k in ("verif. no relevance rep2", "verif. no reconfirmation rep2", "verif. no lexical rep2",
                           "verif. no fresh-evidence guard rep2", "verif. ALL gates removed rep2")]
    ax.scatter([i - w / 2 for i in x], [g["raw_rate"] for g in rep2], s=40, facecolor="white",
               edgecolor="0.15", lw=1.5, zorder=3, label="Second run")
    ax.scatter([i + w / 2 for i in x], [g["strict_rate"] for g in rep2], s=40, facecolor="white",
               edgecolor="0.15", lw=1.5, zorder=3)
    ax.axvline(3.5, color="0.6", lw=1, ls="--")
    for i in (2, 4):  # label only the gaps that carry the claim
        ax.annotate(f"{reported[i]:.1f} vs {audited[i]:.1f}", (i, reported[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color="0.2")
    ax.set_xticks(list(x)); ax.set_xticklabels(conds, fontsize=8)
    ax.set_ylim(-0.04, 1.12); ax.set_ylabel("Success rate (n=10)"); ax.set_xlabel("Verification requirement removed")
    ax.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0),
              handletextpad=0.3, columnspacing=0.8)
    finalize(fig, OUT / "fig_verification_ablation.pdf", COL, 2.4)
    print("wrote fig_verification_ablation.pdf")


def fig_influence_decomposition():
    runs = json.load(open("data/results_postdraft/factorial_influence/influence_probe_decomposition.json"))["per_run"][:4]
    names = ["Full memory", "Full memory (rep 2)", "Persistent", "Persistent (rep 2)"]
    order = [0, 2, 1, 3]
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    for row, (i, name) in enumerate(zip(order, names)):
        r = runs[i]; n = r["n_evals"]
        parts = [(r["not_influenced"] / n, GREY, "not influenced"),
                 (r["influenced_target_changed"] / n, RED, "memory overrode a live option"),
                 (r["influenced_live_empty_live0"] / n, BLUE, "memory used, nothing live")]
        left = 0
        for v, col, lab in parts:
            ax.barh(row, v, left=left, color=col, edgecolor="white", linewidth=1.5,
                    label=lab if row == 0 else None)
            left += v
        ax.annotate(f"n={n}", (1.01, row), va="center", fontsize=7, color="0.3", annotation_clip=False)
    ax.set_yticks(range(4)); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("Share of contextual-anchor decisions")
    ax.legend(frameon=False, ncol=1, loc="lower left", bbox_to_anchor=(-0.02, 1.0))
    ax.grid(False)
    finalize(fig, OUT / "fig_influence_decomposition.pdf", COL, 2.1)
    print("wrote fig_influence_decomposition.pdf")


def fig_decay_stale():
    """Decay vs no decay at 24 h memory age (persistent, 8-query battery):
    share of target choices whose memory target was last seen before the
    current query began, per run, from decisions_ep*.jsonl. Picks up
    no-decay rep2 automatically once it exists."""
    import glob
    conds = [("Decay on", "data/runs/phase4_batch_persistent_pp_decay24h_rep*", BLUE),
             ("Decay off", "data/runs/phase4_batch_persistent_lambda0.0_pp_nodecay24h_rep*", GREY)]
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for i, (name, pat, col) in enumerate(conds):
        rates = []
        for run in sorted(glob.glob(pat)):
            d = [json.loads(l) for f in glob.glob(f"{run}/decisions_ep*.jsonl") for l in open(f)]
            d = [x for x in d if x["memory_influenced"] is not None]
            rates.append(sum(bool(x["memory_target_pre_query"]) for x in d) / len(d))
        m = st.mean(rates)
        ax.bar(i, m, width=0.55, color=col)
        ax.scatter([i] * len(rates), rates, color="0.15", s=16, zorder=3)
        ax.annotate(f"{m:.2f} (runs={len(rates)})", (i, max(rates + [m])), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8)
        print(name, [round(r, 3) for r in rates])
    ax.set_xticks([0, 1]); ax.set_xticklabels([c[0] for c in conds])
    ax.set_ylim(0, 1); ax.set_ylabel("Stale-target rate")
    finalize(fig, OUT / "fig_decay_stale.pdf", COL, 1.8)
    print("wrote fig_decay_stale.pdf")


if __name__ == "__main__":
    fig_factorial()
    fig_verification_ablation()
    fig_influence_decomposition()
    fig_decay_stale()
