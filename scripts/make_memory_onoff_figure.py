"""Memory off vs on, same queries and start pose, drawn on the real top-down
render of HM3D 00770 (2026-09-29).
off: phase4_batch_reset5_rep2  (memory and session reset before every query)
on : phase4_batch_persist_memory_only_persistmemoryonly5_rep2 (memory kept,
     session reset -> same start pose as 'off')
All memory-on searches here are 12-29 steps, i.e. finished before the new DAAAM
session produced groundings, so they are unaffected by the object-number
collision found 2026-09-29 (PROGRESS.md). Paths from the .rrd recordings;
map + world->pixel mapping from render_topdown_00770.py."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rerun.dataframe as rd
from PIL import Image

sys.path.insert(0, "scripts")
from make_teaser import to_px, col  # noqa: E402
from paper_figstyle import FULL, finalize  # noqa: E402

RUNS = {"off": Path("data/runs/phase4_batch_reset5_rep2"),
        "on": Path("data/runs/phase4_batch_persist_memory_only_persistmemoryonly5_rep2")}
T = Path("acra_paper/figures/teaser")
QCOL = ["#8c8c8c", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]


def load(run):
    eps = json.load(open(run / "batch_metrics.json"))["episodes"]
    paths = []
    for e in eps:
        rec = rd.load_recording(str(run / f"episode_{e['episode_id']}.rrd"))
        paths.append(np.array([p[0] for p in col(rec, "/world/agent/position", "Position3D") if p]))
    return eps, paths


def main():
    m = json.load(open(T / "topdown_00770.json"))
    top = np.array(Image.open(T / "topdown_00770.png"))
    data = {k: load(v) for k, v in RUNS.items()}
    allp = np.concatenate([p for eps, ps in data.values() for p in ps])
    (x0, z0), (x1, z1) = [to_px(q, m) for q in (allp.min(0) - 1.5, allp.max(0) + 1.5)]
    x0, x1 = sorted((x0, x1)); z0, z1 = sorted((z0, z1))
    ph = 2.25
    pw = ph * (x1 - x0) / (z1 - z0)
    fig = plt.figure()
    titles = {"off": "Memory off: reset before every query", "on": "Memory on: kept between queries"}
    for k, key in enumerate(("off", "on")):
        eps, paths = data[key]
        ax = fig.add_axes([(0.05 + k * (pw + 0.25)) / FULL, 0.28 / 2.9, pw / FULL, ph / 2.9])
        ax.imshow(top); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i, (p, e) in enumerate(zip(paths, eps)):
            px, pz = to_px(p, m)
            q = e["query"].replace("a ", "", 1).replace("wall-mounted ", "").replace("large ", "")
            ok = e["outcome"] == "success"
            lab = f"Q{i + 1} {q}: {e['steps_taken']} steps" + ("" if ok else " (not found)")
            ax.plot(px, pz, color=QCOL[i], lw=0.8 if i == 0 else 1.6, alpha=0.6 if i == 0 else 1,
                    ls="-" if i == 0 else (0, (3, 1.2)), label=lab, zorder=4 + (i > 0))
            ax.scatter(px[-1], pz[-1], marker="*" if ok else "X", s=60 if ok else 30, color=QCOL[i],
                       ec="black", lw=0.4, zorder=7)
        sx, sz = to_px(paths[1][0], m)
        ax.scatter(sx, sz, s=30, color="#1f3b73", ec="white", lw=1.0, zorder=8, label="start of each query")
        ax.set_xlim(x0, x1); ax.set_ylim(z1, z0)
        ax.set_title(titles[key], fontweight="bold")
        tot = sum(e["steps_taken"] for e in eps[1:])
        ax.text(0.01, 0.02, f"queries 2–5: {tot} steps", transform=ax.transAxes, fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.legend(frameon=True, framealpha=0.9, loc="upper right", handlelength=1.4, borderpad=0.4)
    finalize(fig, Path("acra_paper/figures/fig_memory_onoff.pdf"), FULL, 2.9, tight=False)


if __name__ == "__main__":
    main()
