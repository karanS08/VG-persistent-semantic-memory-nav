"""Five objects spread over HM3D 00770 (NE black vase, north-annex faucet, NW sink,
SW whiteboard, SE painting of the sea): memory off vs on, stacked, on the high-res map
(2026-09-29). Real runs: on = phase4_batch_persist_memory_only_spread2_on (starts
from the scripted house-tour memory), off = phase4_batch_spread2_off (reset).
Queries were SCREENED: chosen so the tour memory's top CLIP match lies at each spot
(an unscreened set failed; see EXPERIMENT_RESULTS.md) -- the figure says so.
Paths from the .rrd recordings; object markers at the position the robot verified,
or the tour-memory position if it was not found.
Out: acra_paper/teaser_assets/video_spread_onoff_1920x1080.png"""
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
from make_teaser import to_px  # noqa: E402

A = Path("acra_paper/teaser_assets")
RUNS = {"off": Path("data/runs/phase4_batch_spread2_off"),
        "on": Path("data/runs/phase4_batch_persist_memory_only_spread2_on")}
NAME = ["black vase", "faucet", "sink", "whiteboard", "sea painting"]
QCOL = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
OFF_C, ON_C, INK = "#c0392b", "#1e8449", "#1f2a44"


def main():
    m = json.load(open("acra_paper/figures/teaser/topdown_00770.json"))
    hr = json.load(open(A / "highres/map_topdown_house_highres.json"))
    img = Image.open(A / "highres/map_topdown_house_white.png")
    W0, H0 = img.size
    img = np.array(img.resize((W0 // 4, H0 // 4), Image.LANCZOS))
    s, (cx, cy) = hr["scale_vs_1600_render"], hr["crop_origin_px"]

    def px(P):
        u, v = to_px(P, m)
        return np.asarray(u) * s - cx, np.asarray(v) * s - cy

    data = {k: load(v) for k, v in RUNS.items()}
    # object label at the verified position (either run); unlabelled if never verified
    obj = {}
    for key in ("on", "off"):
        for i, e in enumerate(data[key][0]):
            v = e.get("verification_detail") or {}
            if e["outcome"] == "success" and v.get("entity_position_world") and i not in obj:
                obj[i] = v["entity_position_world"]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.955, "Five objects around the house: without and with persistent memory", ha="center",
             va="center", fontsize=30, fontweight="bold", color=INK)
    strip_h = (1080 - 240) / 2
    for k, key in enumerate(("off", "on")):
        eps, paths = data[key]
        y0 = 1080 - 95 - (k + 1) * strip_h
        ax = fig.add_axes([0.2, y0 / 1080, 0.79, (strip_h - 10) / 1080])
        ax.imshow(img, extent=(0, W0, H0, 0)); ax.set_axis_off()
        for i, (p, e) in enumerate(zip(paths, eps)):
            x, y = px(p)
            ax.plot(x, y, color=QCOL[i], lw=2.6, alpha=0.9, solid_capstyle="round",
                    path_effects=[pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()], zorder=4)
            ok = e["outcome"] == "success"
            ax.scatter(x[-1], y[-1], marker="*" if ok else "X", s=520 if ok else 260, color=QCOL[i],
                       ec="white", lw=1.8, zorder=6)
        for i, P in obj.items():
            ox, oy = px(np.array(P))
            ax.annotate(NAME[i], (ox, oy), xytext=(0, 26), textcoords="offset points", ha="center",
                        fontsize=13, fontweight="bold", color=QCOL[i], zorder=8,
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        sx, sy = px(paths[0][0])
        ax.scatter(sx, sy, s=260, color=INK, ec="white", lw=2.5, zorder=7)
        col = OFF_C if key == "off" else ON_C
        tot = sum(e["steps_taken"] for e in eps)
        nf = sum(e["outcome"] == "success" for e in eps)
        yc = (y0 + strip_h / 2) / 1080
        fig.text(0.1, yc + 0.085, "Memory OFF" if key == "off" else "Memory ON", ha="center", va="center",
                 fontsize=28, fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.4", fc=col, ec="none"))
        fig.text(0.1, yc + 0.01, "starts every search\nfrom scratch" if key == "off" else "remembers the\nhouse tour",
                 ha="center", va="center", fontsize=17, color="0.3")
        fig.text(0.1, yc - 0.07, f"{tot:,} steps", ha="center", va="center", fontsize=34, fontweight="bold", color=col)
        fig.text(0.1, yc - 0.12, f"{nf} of 5 found", ha="center", va="center", fontsize=17, fontweight="bold", color=col)
    qo, qn = data["off"][0], data["on"][0]
    for i in range(5):
        so = f"{qo[i]['steps_taken']}" + ("" if qo[i]["outcome"] == "success" else " (not found)")
        sn = f"{qn[i]['steps_taken']}" + ("" if qn[i]["outcome"] == "success" else " (not found)")
        fig.text((60 + (i % 3) * 620) / 1920, (92 - (i // 3) * 34) / 1080, f"● {NAME[i]}: {so} → {sn} steps",
                 ha="left", va="center", fontsize=17, fontweight="bold", color=QCOL[i])
    fig.text(0.5, 0.018, "Real Habitat-Sim runs, HM3D scene 00770 (top-down view). Memory ON starts from a scripted "
             "tour of the house; queries chosen where that memory ranks the target first. ★ object verified, ✖ 500-step budget ran out, ● start.",
             ha="center", va="center", fontsize=13, color="0.45")
    out = A / "video_spread_onoff_1920x1080.png"
    fig.savefig(out, dpi=100); print("wrote", out)
    for key in ("off", "on"):
        for e in data[key][0]:
            print(key, e["query"], e["outcome"], e["steps_taken"], round(e["spl"] or 0, 2))


if __name__ == "__main__":
    main()
