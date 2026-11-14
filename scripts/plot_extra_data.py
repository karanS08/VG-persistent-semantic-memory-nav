"""2026-09-29: plots of existing-but-unused data (RAW, unaudited) for review.

fig_long_session.pdf : 28 persistent 10-query sessions (*megabatch*, *longsession10*)
                       vs fresh-start searches of the same queries; + memory size.
fig_latency.pdf      : latency-injection sweep (0-30 s extra DAAAM delay).
Out: results/post_paper/figs/. Run with the rpv_mem python.
"""
import glob
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, "scripts")
from paper_figstyle import finalize, COL, FULL  # print-size normalisation

OUT = Path("results/post_paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
BLUE, RED, GREY = "#2166ac", "#b2182b", "#8c8c8c"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.2})


def runs(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(f"data/runs/{pat}/batch_metrics.json"))]


def ci(xs, n=5000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choice(xs) for _ in xs) for _ in range(n))
    return st.mean(xs), ms[int(0.025 * n)], ms[int(0.975 * n)]


def fig_long_session():
    sessions = runs("*megabatch*") + runs("*longsession10*")
    queries = sessions[0]["queries_used"]
    assert all(s["queries_used"] == queries for s in sessions)
    per_q = defaultdict(list)
    for s in sessions:
        for e in s["episodes"]:
            per_q[e["episode_id"]].append(e)
    # fresh-start reference: same query searched with memory + position reset
    fresh = defaultdict(list)
    for d in runs("phase4_batch_minconf1default") + runs("phase4_batch_minconf1default_rep2") + \
             runs("phase4_batch_pp_full_rep*") + runs("phase4_batch_reset5*"):
        for e in d["episodes"]:
            fresh[e["query"]].append(e)
    short = [q.replace("a ", "", 1).replace("wall-mounted ", "") for q in queries]
    x = list(range(len(queries)))

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
    for ax, key, name, ylim in [(axes[0], "steps_taken", "Steps per query", (0, 520)),
                                (axes[1], "spl", "SPL (raw)", (0, 1.05))]:
        for lab, src, col, mk, off in [("fresh start", lambda i: fresh[queries[i]], GREY, "o", -0.12),
                                       ("persistent session", lambda i: per_q[i], RED, "s", 0.12)]:
            pts = [(i, *ci([e[key] for e in src(i)])) for i in x if src(i)]
            ax.errorbar([p[0] + off for p in pts], [p[1] for p in pts],
                        yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                        fmt=mk, color=col, markersize=6, capsize=2, lw=1.5, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(short, rotation=55, ha="right", fontsize=7.5)
        ax.set_ylim(*ylim); ax.set_title(name, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.36, 0.93))
    mem = [ci([e["memory_size_at_end"] for e in per_q[i]]) for i in x]
    axes[2].plot([i + 1 for i in x], [m[0] for m in mem], color=BLUE, marker="o", lw=2)
    axes[2].fill_between([i + 1 for i in x], [m[1] for m in mem], [m[2] for m in mem], color=BLUE, alpha=0.15, lw=0)
    axes[2].set_xlabel("queries answered in session"); axes[2].set_title("Memory entries stored", fontsize=9)
    axes[2].set_ylim(0, None)
    fig.suptitle(f"{len(sessions)} persistent 10-query sessions (scene 00770) vs fresh-start searches of the same query "
                 "- raw, unaudited; bars = bootstrap 95% CI", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / "fig_long_session.pdf"); fig.savefig(OUT / "fig_long_session.png", dpi=110)
    plt.close(fig)
    ns = {q: len(fresh[q]) for q in queries}
    print("sessions", len(sessions), "fresh n per query", ns)
    for i in x:
        f = fresh[queries[i]]; p = per_q[i]
        print(f"{short[i]:14s} fresh steps {st.mean(e['steps_taken'] for e in f):5.0f} succ {st.mean(e['outcome']=='success' for e in f):.2f} "
              f"| session steps {st.mean(e['steps_taken'] for e in p):5.0f} succ {st.mean(e['outcome']=='success' for e in p):.2f} "
              f"spl {st.mean(e['spl'] for e in p):.2f} | mem {st.mean(e['memory_size_at_end'] for e in p):.0f}")


def fig_latency():
    rows = sorted(((d["inject_latency_s"], d["episodes"]) for d in runs("*latency*")), key=lambda r: r[0])
    lat = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for key, lab, col, mk in [(lambda e: e["outcome"] == "success", "success (raw)", RED, "s"),
                              (lambda e: e["spl"], "SPL (raw)", BLUE, "o")]:
        pts = [ci([float(key(e)) for e in eps]) for _, eps in rows]
        ax.errorbar(lat, [p[0] for p in pts], yerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]],
                    color=col, marker=mk, capsize=2, lw=1.5, label=lab)
    ax.set_xlabel("extra perception latency injected (s)"); ax.set_ylim(0, 1)
    ax.set_title("Latency injection, standard battery (n=8 per point)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency.pdf"); fig.savefig(OUT / "fig_latency.png", dpi=110)
    plt.close(fig)


def fig_long_session_audited():
    """Strict-audited long-horizon figure (paper Fig.: fig_long_session.pdf).
    Labels: results/post_paper/long_session_audit_labels.json.
    Exclusion rule (2026-09-29): a session whose memory store never holds an
    entry is a pipeline failure, not a memory-persistence trial -- excludes
    phase4_batch_persistent_longsession10_covmem (0 entries at every query,
    0/10; re-run as _covmem_v2)."""
    L = json.load(open("results/post_paper/long_session_audit_labels.json"))["labels"]
    def eps(pat, sessions=False):
        out = []
        for f in sorted(glob.glob(f"data/runs/{pat}/batch_metrics.json")):
            d = json.load(open(f)); r = f.split("/")[2]
            if sessions and max(e["memory_size_at_end"] for e in d["episodes"]) == 0:
                print("  excluded (empty memory store):", r); continue
            for e in d["episodes"]:
                lab = L.get(f"{r}/{e['episode_id']}") if e["outcome"] == "success" else None
                assert e["outcome"] != "success" or lab, f"unaudited success {r}/{e['episode_id']}"
                out.append(dict(e, strict=lab == "strict", lenient=lab in ("strict", "lenient"), qs=d["queries_used"]))
        return out
    sess = eps("*megabatch*", True) + eps("*longsession10*", True)
    n_sess = len(sess) // 10
    queries = sess[0]["qs"]
    fresh = [e for pat in ("phase4_batch_minconf1default", "phase4_batch_minconf1default_rep2",
                           "phase4_batch_pp_full_rep1", "phase4_batch_pp_full_rep2", "phase4_batch_reset5*")
             for e in eps(pat)]
    per_q = defaultdict(list); fr = defaultdict(list)
    for e in sess: per_q[e["episode_id"]].append(e)
    for e in fresh: fr[e["query"]].append(e)
    short = [q.replace("a ", "", 1).replace("wall-mounted ", "").replace("framed ", "") for q in queries]
    x = list(range(len(queries)))
    metrics = [("Navigation steps", lambda e: e["steps_taken"], (0, 520)),
               ("Strict-audited success", lambda e: float(e["strict"]), (0, 1.05)),
               ("Strict-audited SPL", lambda e: e["spl"] * e["strict"], (0, 1.05))]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    for ax, (name, f, ylim) in zip(axes, metrics):
        for lab, src, col, mk, off, join in [
                ("Fresh start (n=3\u20137 per query)", lambda i: fr[queries[i]], GREY, "o", -0.14, False),
                (f"Persistent session (n={n_sess} sessions \u00d7 10 queries)", lambda i: per_q[i], RED, "s", 0.14, True)]:
            pts = [(i, *ci([f(e) for e in src(i)])) for i in x]
            xs = [p[0] + off for p in pts]
            if join:
                ax.plot(xs, [p[1] for p in pts], color=col, lw=0.8, alpha=0.6)
            ax.errorbar(xs, [p[1] for p in pts],
                        yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                        fmt=mk, color=col, markersize=4.5, capsize=1.5, lw=1.1, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([f"{i+1} {q}" for i, q in enumerate(short)],
                                             rotation=60, ha="right", fontsize=6.5, rotation_mode="anchor")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(*ylim); ax.set_title(name, fontsize=8)
        ax.set_xlabel("sequential query order \u2192", fontsize=6.5, color="0.4")
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle(f"Long-horizon persistent retrieval \u2014 {n_sess} sessions \u00d7 10 sequential queries "
                 "(memory reaches ~2,000 entries by query 10)", fontsize=8.5, y=1.02)
    fig.legend(h, l, frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.97))
    finalize(fig, Path("acra_paper/figures/fig_long_session.pdf"), FULL, 2.7, rect=(0, 0, 1, 0.9))

    def perm(a, b, n=50000):
        rng = random.Random(0); o = st.mean(a) - st.mean(b); pool = a + b; k = len(a); h = 0
        for _ in range(n):
            rng.shuffle(pool); h += abs(st.mean(pool[:k]) - st.mean(pool[k:])) >= abs(o) - 1e-12
        return o, (h + 1) / (n + 1)
    q1 = [e for e in sess if e["episode_id"] > 0]
    fq1 = [e for q in queries[1:] for e in fr[q]]
    print(f"sessions={n_sess}; queries 2-10: session n={len(q1)}, fresh n={len(fq1)}")
    for name, f in (("strict success", lambda e: float(e["strict"])), ("strict SPL", lambda e: e["spl"] * e["strict"]),
                    ("steps", lambda e: float(e["steps_taken"]))):
        a, b = [f(e) for e in q1], [f(e) for e in fq1]
        d, pv = perm(a, b)
        print(f"  {name:15s} session {st.mean(a):.3f} fresh {st.mean(b):.3f} diff {d:+.3f} p={pv:.4f}")
    for i in x:
        a, b = fr[queries[i]], per_q[i]
        print(f"  {i+1:2d} {short[i]:13s} fresh steps {st.mean(e['steps_taken'] for e in a):4.0f} strict {st.mean(e['strict'] for e in a):.2f} "
              f"aSPL {st.mean(e['spl']*e['strict'] for e in a):.2f} | session steps {ci([e['steps_taken'] for e in b])} "
              f"strict {ci([float(e['strict']) for e in b])} aSPL {ci([e['spl']*e['strict'] for e in b])} "
              f"mem {st.mean(e['memory_size_at_end'] for e in b):.0f}")


def fig_lag_baseline():
    """Lag baseline (reviewer check on RQ1): memory-off vs memory-off with a
    30 s live window vs full memory, standard 8-query battery, strict audit.
    Labels: postplan_audit_labels.json (lag + pp_full runs) and
    long_session_audit_labels.json (minconf1default fresh runs). Unaudited
    successes raise, so only fully-audited runs are plotted."""
    L = {**json.load(open("results/post_paper/long_session_audit_labels.json"))["labels"],
         **json.load(open("results/post_paper/postplan_audit_labels.json"))["labels"]}
    conds = [("Memory-off\n(control)", ["phase4_batch_memory_off_lagbase_rep*"]),
             ("Memory-off +\n30 s live window", ["phase4_batch_memory_off_livewin30_lagbase_rep*"]),
             ("Full memory", ["phase4_batch_minconf1default", "phase4_batch_minconf1default_rep2",
                              "phase4_batch_pp_full_rep1", "phase4_batch_pp_full_rep2"])]
    rows = []
    for name, pats in conds:
        files = sorted(f for p in pats for f in glob.glob(f"data/runs/{p}/batch_metrics.json"))
        per_run = []
        for f in files:
            r = f.split("/")[2]; E = json.load(open(f))["episodes"]
            labs = [L.get(f"{r}/{e['episode_id']}") if e["outcome"] == "success" else "fail" for e in E]
            assert None not in labs, f"unaudited success in {r}"
            per_run.append((r, sum(l == "strict" for l in labs), sum(l in ("strict", "lenient") for l in labs), len(E)))
        rows.append((name, per_run))
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for i, (name, runs_) in enumerate(rows):
        k = sum(r[1] for r in runs_); n = sum(r[3] for r in runs_)
        ax.bar(i, k / n, color=[GREY, BLUE, RED][i], width=0.6)
        ax.scatter([i] * len(runs_), [r[1] / r[3] for r in runs_], color="0.15", s=14, zorder=3)
        ax.annotate(f"{k}/{n}", (i, k / n), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        print(f"{name.replace(chr(10), ' '):32s} strict {k}/{n}={k/n:.3f}  runs: " +
              ", ".join(f"{r[0].replace('phase4_batch_', '')} {r[1]}/{r[3]} (lenient {r[2]})" for r in runs_))
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Strict-audited success")
    finalize(fig, Path("acra_paper/figures/fig_lag_baseline.pdf"), COL, 2.1)


if __name__ == "__main__":
    fig_long_session_audited()
    fig_lag_baseline()

