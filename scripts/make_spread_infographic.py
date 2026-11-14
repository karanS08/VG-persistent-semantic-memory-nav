"""Spread-object infographic for the video/teaser (2026-09-29): five objects around
HM3D 00770 (NE carved fireplace, north-annex tripod, NW mirror, SW radiator, SE
tapestry) as picture cards on the sides, joined to their spots on the top-down map.
ILLUSTRATION, labelled as such on the image:
  memory ON  = planned navmesh shortest paths from the spawn to each spot
               (spread/planned_routes.json, plan_spread_routes.py), not agent runs;
  memory OFF = real Habitat-Sim searches with no memory (phase4_batch_spread3_off),
               drawn only once that run has finished.
Cards: real eye-level renders from each spot (spread/card_<k>_<v>.png).
Out: acra_paper/teaser_assets/video_spread_infographic_1920x1080.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyBboxPatch
from PIL import Image

sys.path.insert(0, "scripts")
from make_teaser import to_px  # noqa: E402

A = Path("acra_paper/teaser_assets")
S = A / "spread"
OFF_RUN = Path("data/runs/phase4_batch_spread3_off")
# k: (label, card view, side, slot)  -- k follows spread_objects.json / planned_routes.json
OBJ = {0: ("carved fireplace", 2, "R", 1), 1: ("tripod", 7, "R", 0), 2: ("mirror", 4, "L", 0),
       3: ("radiator", 4, "L", 1), 4: ("tapestry", 4, "R", 2)}
QCOL = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#e6ab02"]
ON_C, OFF_C, INK = "#1e8449", "#c0392b", "#1f2a44"


def load_off():
    if not (OFF_RUN / "batch_metrics.json").exists():
        return None
    from make_memory_onoff_figure import load
    return load(OFF_RUN)


def main():
    m = json.load(open("acra_paper/figures/teaser/topdown_00770.json"))
    hr = json.load(open(A / "highres/map_topdown_house_highres.json"))
    im = Image.open(A / "highres/map_topdown_house_white.png")
    W0, H0 = im.size
    img = np.array(im.resize((W0 // 4, H0 // 4), Image.LANCZOS))
    s, (cx, cy) = hr["scale_vs_1600_render"], hr["crop_origin_px"]

    def px(P):
        u, v = to_px(np.asarray(P, float), m)
        return np.asarray(u) * s - cx, np.asarray(v) * s - cy

    plan = json.load(open(S / "planned_routes.json"))
    off = load_off()

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.955, "Where is it? Remembering the house vs searching from scratch", ha="center", va="center",
             fontsize=30, fontweight="bold", color=INK)
    mw = 1240; mh = mw * H0 / W0
    ax = fig.add_axes([(960 - mw / 2) / 1920, (1080 - 110 - mh) / 1080, mw / 1920, mh / 1080])
    ax.imshow(img, extent=(0, W0, H0, 0)); ax.set_xlim(0, W0); ax.set_ylim(H0, 0); ax.set_axis_off()

    if off is not None:
        for p in off[1]:
            x, y = px(p)
            ax.plot(x, y, color=OFF_C, lw=1.8, alpha=0.55, ls=(0, (4, 2)), zorder=3)
        for p, e in zip(off[1], off[0]):
            x, y = px(p)
            ok = e["outcome"] == "success"
            ax.scatter(x[-1], y[-1], marker="*" if ok else "X", s=200 if ok else 110, color=OFF_C, ec="white",
                       lw=1.2, zorder=5)
    for r in plan["routes"]:
        x, y = px(np.array(r["path_world"]))
        ax.plot(x, y, color=ON_C, lw=4.2, solid_capstyle="round", solid_joinstyle="round",
                path_effects=[pe.Stroke(linewidth=7, foreground="white"), pe.Normal()], zorder=4)
    sx, sy = px(plan["start"])
    ax.scatter(sx, sy, s=380, color=INK, ec="white", lw=3, zorder=8)
    ax.annotate("START", (sx, sy), xytext=(0, -24), textcoords="offset points", ha="center", fontsize=14,
                fontweight="bold", color=INK, zorder=9, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # cards
    cw, ch = 300, 225
    for r in plan["routes"]:
        k = r["k"]; name, view, side, slot = OBJ[k]
        gx, gy = px(r["goal"])
        ax.scatter(gx, gy, s=700, marker="o", color=QCOL[k], ec="white", lw=3, zorder=9)
        ax.text(gx, gy, str(k + 1), ha="center", va="center", fontsize=16, fontweight="bold", color="white", zorder=10)
        nslots = 3 if side == "R" else 2
        top = 110 + (820 - nslots * (ch + 45)) / 2 + slot * (ch + 45)
        left = 25 if side == "L" else 1920 - 25 - cw
        cax = fig.add_axes([left / 1920, (1080 - top - ch) / 1080, cw / 1920, ch / 1080])
        cax.imshow(Image.open(S / f"card_{k}_{view}.png")); cax.set_xticks([]); cax.set_yticks([])
        for sp in cax.spines.values():
            sp.set_edgecolor(QCOL[k]); sp.set_linewidth(5)
        fig.text((left + cw / 2) / 1920, (1080 - top - ch - 20) / 1080, f"{k + 1}  {name}", ha="center", va="center",
                 fontsize=18, fontweight="bold", color=QCOL[k])
        anchor = (cw, ch / 2) if side == "L" else (0, ch / 2)
        con = ConnectionPatch(xyA=(anchor[0] / cw * 640, 240), coordsA="data", axesA=cax,
                              xyB=(gx, gy), coordsB="data", axesB=ax, color=QCOL[k], lw=2.2, zorder=2)
        fig.add_artist(con)

    # legend / numbers
    L = sum(r["geodesic_m"] for r in plan["routes"])
    y = 0.135
    fig.add_artist(FancyBboxPatch((0.2, 0.08), 0.6, 0.13, boxstyle="round,pad=0.005,rounding_size=0.01",
                                  transform=fig.transFigure, fc="#f4f6f8", ec="none", zorder=0))
    fig.lines.append(plt.Line2D([0.225, 0.265], [y + 0.035] * 2, color=ON_C, lw=5, transform=fig.transFigure))
    fig.text(0.275, y + 0.035, f"Memory ON: straight to each remembered object  ({L:.0f} m for all five)",
             va="center", fontsize=17, fontweight="bold", color=ON_C)
    if off is not None:
        eps = off[0]; nf = sum(e["outcome"] == "success" for e in eps); st = sum(e["steps_taken"] for e in eps)
        otxt = f"Memory OFF: explores until it sees it  ({st:,} steps, {nf} of 5 found)"
    else:
        otxt = "Memory OFF: explores until it sees it  (runs in progress)"
    fig.lines.append(plt.Line2D([0.225, 0.265], [y - 0.01] * 2, color=OFF_C, lw=3, ls=(0, (4, 2)), transform=fig.transFigure))
    fig.text(0.275, y - 0.01, otxt, va="center", fontsize=17, fontweight="bold", color=OFF_C)
    fig.text(0.5, 0.012, "Illustration on HM3D scene 00770 (Habitat-Sim). Memory-ON routes are planned shortest paths to "
             "each object; memory-OFF paths are real searches with no memory. ★ found, ✖ 500-step budget ran out.",
             ha="center", va="center", fontsize=12, color="0.45")
    out = A / "video_spread_infographic_1920x1080.png"
    fig.savefig(out, dpi=100); print("wrote", out, "| off run:", "yes" if off else "not yet")


if __name__ == "__main__":
    main()
