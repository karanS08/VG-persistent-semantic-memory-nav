"""Video frame (1920x1080): the same five searches with persistent memory off vs
on, real paths on the real top-down render of HM3D 00770 (2026-09-29).
Data: acra_paper/teaser_assets/paths_memory_onoff.json (from
export_teaser_assets.py: off = phase4_batch_reset5_rep2, on =
phase4_batch_persist_memory_only_persistmemoryonly5_rep2). Markers mark where
each search ended (star = verified find, X = step budget ran out).
Out: acra_paper/teaser_assets/video_memory_onoff_1920x1080.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

A = Path("acra_paper/teaser_assets")
QCOL = {2: "#1b9e77", 3: "#d95f02", 4: "#7570b3", 5: "#e7298a"}
NAME = {2: "radiator", 3: "cabinet", 4: "wall vent", 5: "bench"}
OFF_C, ON_C, INK = "#c0392b", "#1e8449", "#1f2a44"


def main():
    d = json.load(open(A / "paths_memory_onoff.json"))
    base = Image.open(A / "map_topdown_house.png").convert("RGB")
    base = ImageEnhance.Brightness(ImageEnhance.Color(base).enhance(0.45)).enhance(0.8)   # dim so paths pop
    base = np.array(base)
    pts = np.concatenate([np.array(q["path_px"]) for k in d for q in d[k]["queries"][1:]])
    x0, y0 = pts.min(0) - 60; x1, y1 = pts.max(0) + 60
    pw, ph = 860, 600                                    # panel size in px
    cx, cy, half_w = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, (y1 - y0) / 2 * pw / ph)
    half_h = half_w * ph / pw
    H, W = base.shape[:2]                                  # keep the view inside the map image
    cy = min(max(cy, half_h), H - half_h); cx = min(max(cx, half_w), W - half_w)
    # label offsets (points): the memory-on targets sit close together
    OFFS = {"off": {2: (0, -34), 3: (-70, 20), 4: (0, 30), 5: (70, 26)},
            "on": {2: (95, -10), 3: (-45, -40), 4: (40, 55), 5: (-80, -35)}}

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.94, "Same robot, same searches: with and without persistent memory", ha="center", va="center",
             fontsize=34, fontweight="bold", color=INK)
    fig.text(0.5, 0.885, "Real Habitat-Sim runs in HM3D scene 00770. After one shared first search (for a window), "
             "the robot is asked for four more objects.", ha="center", va="center", fontsize=18, color="0.35")
    for k, key in enumerate(("off", "on")):
        left = 80 + k * (pw + 100)
        ax = fig.add_axes([left / 1920, 200 / 1080, pw / 1920, ph / 1080])
        ax.imshow(base); ax.set_xlim(cx - half_w, cx + half_w); ax.set_ylim(cy + half_h, cy - half_h)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(OFF_C if key == "off" else ON_C); sp.set_linewidth(4)
        for q in d[key]["queries"][1:]:
            i = q["query_index"]; p = np.array(q["path_px"])
            ax.plot(p[:, 0], p[:, 1], color=QCOL[i], lw=4.5, solid_capstyle="round",
                    path_effects=[pe.Stroke(linewidth=7.5, foreground="white"), pe.Normal()], zorder=4)
            ok = q["found"]
            ax.scatter(p[-1, 0], p[-1, 1], marker="*" if ok else "X", s=900 if ok else 420, color=QCOL[i],
                       ec="white", lw=2, zorder=6)
            ax.annotate(NAME[i] + ("" if ok else " (not found)"), (p[-1, 0], p[-1, 1]), xytext=OFFS[key][i],
                        textcoords="offset points", ha="center", va="center", fontsize=15, fontweight="bold",
                        color="white", bbox=dict(boxstyle="round,pad=0.3", fc=QCOL[i], ec="none"), zorder=7,
                        arrowprops=dict(arrowstyle="-", color=QCOL[i], lw=2))
        sx, sy = d[key]["start_px"]
        ax.scatter(sx, sy, s=420, color=INK, ec="white", lw=3, zorder=8)
        ax.annotate("start", (sx, sy), xytext=(-55, -8), textcoords="offset points", ha="center", va="center", fontsize=15,
                    fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.3", fc=INK, ec="none"), zorder=8)
        head = "Memory OFF: forgets after every search" if key == "off" else "Memory ON: remembers between searches"
        fig.text((left + pw / 2) / 1920, (200 + ph + 40) / 1080, head, ha="center", va="center", fontsize=24,
                 fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.45", fc=OFF_C if key == "off" else ON_C, ec="none"))
        tot = d[key]["steps_queries_2_to_5"]
        ax.text(0.03, 0.95, f"{tot:,} steps", transform=ax.transAxes, va="top", fontsize=34, fontweight="bold",
                color=OFF_C if key == "off" else ON_C, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.92))
    # bottom row: per-object step counts, off -> on
    qs_off = {q["query_index"]: q for q in d["off"]["queries"]}; qs_on = {q["query_index"]: q for q in d["on"]["queries"]}
    for j, i in enumerate((2, 3, 4, 5)):
        off_s = f"{qs_off[i]['steps']}" + ("" if qs_off[i]["found"] else " (not found)")
        fig.text((330 + (j % 2) * 700) / 1920, (140 - (j // 2) * 42) / 1080,
                 f"\u25CF {NAME[i]}:  {off_s}  \u2192  {qs_on[i]['steps']} steps",
                 ha="left", va="center", fontsize=20, color=QCOL[i], fontweight="bold")
    fig.text(0.5, 0.045, "Steps per search without memory → with memory.  Lines: the robot's actual path; "
             "★ object found, ✖ step budget (500) ran out.", ha="center", va="center", fontsize=15, color="0.4")
    fig.savefig(A / "video_memory_onoff_1920x1080.png", dpi=100)
    print("wrote", A / "video_memory_onoff_1920x1080.png")


if __name__ == "__main__":
    main()
