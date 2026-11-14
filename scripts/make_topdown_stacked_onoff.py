"""Memory off vs on on the full top-down map of HM3D 00770, stacked like the
isometric frame (off above, on below), 1920x1080 (2026-09-29).
Data: acra_paper/teaser_assets/paths_memory_onoff.json (pixel coordinates in
map_topdown_house.png; runs phase4_batch_reset5_rep2 / ..._persistmemoryonly5_rep2).
Out: acra_paper/teaser_assets/video_topdown_onoff_1920x1080.png"""
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
    img = Image.open(A / "map_topdown_house.png").convert("RGB")
    img = np.array(ImageEnhance.Brightness(ImageEnhance.Color(img).enhance(0.6)).enhance(0.9))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.955, "Same robot, same searches: with and without persistent memory", ha="center",
             va="center", fontsize=30, fontweight="bold", color=INK)
    strip_h = (1080 - 240) / 2
    for k, key in enumerate(("off", "on")):
        y0 = 1080 - 95 - (k + 1) * strip_h
        ax = fig.add_axes([0.2, y0 / 1080, 0.79, (strip_h - 10) / 1080])
        ax.imshow(img, aspect="equal"); ax.set_axis_off()
        for q in d[key]["queries"][1:]:
            i = q["query_index"]; p = np.array(q["path_px"])
            ax.plot(p[:, 0], p[:, 1], color=QCOL[i], lw=3.2, solid_capstyle="round",
                    path_effects=[pe.Stroke(linewidth=5.5, foreground="white"), pe.Normal()], zorder=4)
            ax.scatter(p[-1, 0], p[-1, 1], marker="*" if q["found"] else "X", s=520 if q["found"] else 260,
                       color=QCOL[i], ec="white", lw=1.8, zorder=6)
        sx, sy = d[key]["start_px"]
        ax.scatter(sx, sy, s=260, color=INK, ec="white", lw=2.5, zorder=7)
        col = OFF_C if key == "off" else ON_C
        yc = (y0 + strip_h / 2) / 1080
        fig.text(0.1, yc + 0.075, "Memory OFF" if key == "off" else "Memory ON", ha="center", va="center",
                 fontsize=28, fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.4", fc=col, ec="none"))
        fig.text(0.1, yc, "forgets after\nevery search" if key == "off" else "remembers\nbetween searches",
                 ha="center", va="center", fontsize=17, color="0.3")
        fig.text(0.1, yc - 0.085, f"{d[key]['steps_queries_2_to_5']:,} steps", ha="center", va="center",
                 fontsize=34, fontweight="bold", color=col)
    qo = {q["query_index"]: q for q in d["off"]["queries"]}; qn = {q["query_index"]: q for q in d["on"]["queries"]}
    for j, i in enumerate((2, 3, 4, 5)):
        s_off = f"{qo[i]['steps']}" + ("" if qo[i]["found"] else " (not found)")
        fig.text((380 + (j % 2) * 640) / 1920, (92 - (j // 2) * 34) / 1080,
                 f"● {NAME[i]}: {s_off} → {qn[i]['steps']} steps",
                 ha="left", va="center", fontsize=17, fontweight="bold", color=QCOL[i])
    fig.text(0.5, 0.018, "Real Habitat-Sim runs, HM3D scene 00770 (top-down view). ★ object found, "
             "✖ step budget ran out, ● start. Steps for four searches after a shared first search.",
             ha="center", va="center", fontsize=13, color="0.45")
    out = A / "video_topdown_onoff_1920x1080.png"
    fig.savefig(out, dpi=100); print("wrote", out)


if __name__ == "__main__":
    main()
