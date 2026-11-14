"""Memory off vs on, drawn on the isometric (long-edge) render of HM3D 00770
(2026-09-29). Paths: .rrd recordings of the same runs as the flat version
(off = phase4_batch_reset5_rep2, on = phase4_batch_persist_memory_only_persistmemoryonly5_rep2).
Projection: acra_paper/teaser_assets/isometric_00770_yaw0.json (camera basis +
scale fitted against the navmesh). Paths are drawn at floor height; they are
not occluded by walls. Out: acra_paper/teaser_assets/video_isometric_onoff_1920x1080.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, "scripts")
from make_memory_onoff_figure import load  # noqa: E402

A = Path("acra_paper/teaser_assets")
RUNS = {"off": Path("data/runs/phase4_batch_reset5_rep2"),
        "on": Path("data/runs/phase4_batch_persist_memory_only_persistmemoryonly5_rep2")}
QCOL = {2: "#1b9e77", 3: "#d95f02", 4: "#7570b3", 5: "#e7298a"}
NAME = {2: "radiator", 3: "cabinet", 4: "wall vent", 5: "bench"}
OFF_C, ON_C, INK = "#c0392b", "#1e8449", "#1f2a44"


def proj(P, m):
    d = np.asarray(P, float) - np.array(m["cam"])
    return (m["W"] / 2 + d @ np.array(m["right"]) * m["px_per_m"] + m["dcol"],
            m["H"] / 2 - d @ np.array(m["up"]) * m["px_per_m"] + m["drow"])


def main():
    m = json.load(open(A / "isometric_00770_yaw0.json"))
    img = np.array(Image.open(A / "isometric_00770_yaw0.png"))
    rows = np.where(img.sum((1, 2)) > 0)[0]; cols = np.where(img.sum((0, 2)) > 0)[0]
    r0, r1 = rows.min() - 20, rows.max() + 20
    c0, c1 = cols.min() - 30, cols.max() + 30
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.955, "Same robot, same searches: with and without persistent memory", ha="center",
             va="center", fontsize=30, fontweight="bold", color=INK)
    strip_h = (1080 - 240) / 2
    for k, key in enumerate(("off", "on")):
        eps, paths = load(RUNS[key])
        y0 = 1080 - 95 - (k + 1) * strip_h
        ax = fig.add_axes([0.2, y0 / 1080, 0.79, (strip_h - 10) / 1080])
        ax.imshow(img); ax.set_xlim(c0, c1); ax.set_ylim(r1, r0); ax.set_axis_off()
        for i, (p, e) in enumerate(zip(paths, eps)):
            if i == 0:
                continue
            P = p.copy(); P[:, 1] = 0.13
            x, y = proj(P, m)
            q = i + 1                                     # query number (paths are 0-indexed)
            ax.plot(x, y, color=QCOL[q], lw=3.2, solid_capstyle="round",
                    path_effects=[pe.Stroke(linewidth=5.5, foreground="white"), pe.Normal()], zorder=4)
            ok = e["outcome"] == "success"
            ax.scatter(x[-1], y[-1], marker="*" if ok else "X", s=520 if ok else 260, color=QCOL[q], ec="white", lw=1.8, zorder=6)
        sx, sy = proj(np.r_[paths[1][0][0], 0.13, paths[1][0][2]], m)
        ax.scatter(sx, sy, s=260, color=INK, ec="white", lw=2.5, zorder=7)
        col = OFF_C if key == "off" else ON_C
        head = "Memory OFF" if key == "off" else "Memory ON"
        sub = "forgets after\nevery search" if key == "off" else "remembers\nbetween searches"
        tot = sum(e["steps_taken"] for e in eps[1:])
        yc = (y0 + strip_h / 2) / 1080
        fig.text(0.1, yc + 0.075, head, ha="center", va="center", fontsize=28, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.4", fc=col, ec="none"))
        fig.text(0.1, yc + 0.0, sub, ha="center", va="center", fontsize=17, color="0.3")
        fig.text(0.1, yc - 0.085, f"{tot:,} steps", ha="center", va="center", fontsize=34, fontweight="bold", color=col)
    qs = {k: {i + 1: e for i, e in enumerate(load(RUNS[k])[0])} for k in RUNS}
    for j, i in enumerate((2, 3, 4, 5)):
        off = qs["off"][i]; on = qs["on"][i]
        s_off = f"{off['steps_taken']}" + ("" if off["outcome"] == "success" else " (not found)")
        fig.text((380 + (j % 2) * 640) / 1920, (92 - (j // 2) * 34) / 1080, f"● {NAME[i]}: {s_off} → {on['steps_taken']} steps",
                 ha="left", va="center", fontsize=17, fontweight="bold", color=QCOL[i])
    fig.text(0.5, 0.018, "Real Habitat-Sim runs, HM3D scene 00770. ★ object found, ✖ step budget ran out, "
             "● start. Steps for four searches after a shared first search.", ha="center", va="center",
             fontsize=13, color="0.45")
    out = A / "video_isometric_onoff_1920x1080.png"
    fig.savefig(out, dpi=100); print("wrote", out)


if __name__ == "__main__":
    main()
