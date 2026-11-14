"""Scene figure (2026-09-29): what the robot sees and where it went, scene 00770.

(a) third-person view of the robot (Spot model, illustration only) placed in
    the window room, rendered by scripts/render_spot_third_person.py;
(b) the robot camera at the capture pose of scripts/capture_live_grounding_
    overlay.py with DAAAM's tracking boxes, labelled with the captions DAAAM
    gave these two objects in the evaluation runs (verification_detail);
(c) top-down render of 00770 with the robot's real path for ONE search,
    "a wall-mounted radiator" (query 2), memory on (top, persistmemoryonly5_rep2)
    vs memory off (bottom, reset5_rep2), same start; star = verified (path end).
Everything is read from data on disk; nothing is drawn by hand."""
import json
import math
import sys
from pathlib import Path

import habitat_sim
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rerun.dataframe as rd
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, "scripts")
from paper_figstyle import finalize, FULL  # noqa: E402

RUN = Path("data/runs/phase4_batch_persist_memory_only_persistmemoryonly5_rep2")
NAVMESH = "data/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.basis.navmesh"
LG = Path("acra_paper/figures/live_grounding")
MPP = 0.05  # metres per map pixel
QCOL = ["0.45", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]  # query 1 (exploration) grey, then distinct


def column(rec, entity, comp):
    t = rec.view(index="step", contents=entity).select().read_all()
    return t.column("step").to_pylist(), t.column(f"{entity}:{comp}").to_pylist()


def main():
    metrics = json.load(open(RUN / "batch_metrics.json"))
    eps = metrics["episodes"]
    paths, last_objs = [], None
    for e in eps:
        rec = rd.load_recording(str(RUN / f"episode_{e['episode_id']}.rrd"))
        _, pos = column(rec, "/world/agent/position", "Position3D")
        paths.append(np.array([p[0] for p in pos if p]))
        _, P = column(rec, "/world/daaam/objects", "Position3D")
        _, T = column(rec, "/world/daaam/objects", "Text")
        last_objs = (np.array(P[-1]), T[-1])
        if e["episode_id"] == 1:  # radiator: final frame = the verified view
            _, fmt = column(rec, "/world/agent/rgb", "ImageFormat")
            _, buf = column(rec, "/world/agent/rgb", "ImageBuffer")
            f = fmt[-1][0]
            view = np.array(buf[-1][0], dtype=np.uint8).reshape(f["height"], f["width"], 3)
    obj_xyz, obj_txt = last_objs
    targets = []
    for e in eps:
        d = (e["verification_detail"] or {}).get("description") or ""
        hit = [i for i, t in enumerate(obj_txt) if t and d.startswith(t[:50])]
        targets.append(obj_xyz[hit[0]] if hit else None)

    m = json.load(open("acra_paper/figures/teaser/topdown_00770.json"))
    hr = json.load(open("acra_paper/teaser_assets/highres/map_topdown_house_highres.json"))
    top = np.array(Image.open("acra_paper/teaser_assets/highres/map_topdown_house_white.png"))
    sys.path.insert(0, "scripts")
    from make_teaser import to_px

    def world_px(P, m):  # IoU-fitted mapping of the 1600 px render, scaled to the high-res crop
        u, v = to_px(P, m)
        return u * hr["scale_vs_1600_render"] - hr["crop_origin_px"][0], v * hr["scale_vs_1600_render"] - hr["crop_origin_px"][1]

    fig = plt.figure()
    W, H = FULL, 2.2
    y0, h = 0.03, 1.86 / H
    a = fig.add_axes([0.0, y0, 1.86 * 1.2 / W, h])
    b = fig.add_axes([2.3 / W, y0, 1.86 * 4 / 3 / W, h])
    c_on = fig.add_axes([4.85 / W, y0 + h / 2 + 0.01, 2.15 / W, h / 2 - 0.06])
    c_off = fig.add_axes([4.85 / W, y0, 2.15 / W, h / 2 - 0.06])
    from PIL import ImageEnhance
    spot = np.array(ImageEnhance.Brightness(Image.open("acra_paper/figures/spot_third_person.png").convert("RGB")).enhance(1.3))
    a.imshow(spot[:, int(spot.shape[1] * 200 / 1280):int(spot.shape[1] * 1120 / 1280)])
    a.set_title("(a) Robot in the scene")
    b.imshow(Image.open(LG / "raw_frame_matched.png"))
    tracks = {t["id"]: t["bbox"] for t in json.load(open(LG / "tracks_matched.json"))}
    for tid, bb in tracks.items():
        key = tid in (3, 4)
        b.add_patch(Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False,
                              ec="#ffd700" if key else "#7CFC00", lw=1.3 if key else 0.4, alpha=1 if key else 0.6))
    for tid, name in ((4, "window"), (3, "radiator")):
        bb = tracks[tid]
        b.text(bb[0] + 3, bb[1] + 3, name, color="black", va="top", bbox=dict(fc="#ffd700", ec="none", pad=1.0))
    b.set_title("(b) Robot camera, DAAAM tracks")
    for ax in (a, b):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # (c) the same search (query 2, "a wall-mounted radiator"), same start, memory on vs off
    from make_memory_onoff_figure import RUNS, load
    runs = {k: load(RUNS[k]) for k in ("on", "off")}
    Q = 1   # 0-indexed episode = query 2
    both = np.concatenate([runs[k][1][Q] for k in runs])
    (u0, v0), (u1, v1) = world_px(both.min(0) - 1.0, m), world_px(both.max(0) + 1.0, m)
    for ax, key, col, lab in ((c_on, "on", "#1e8449", "Memory on"), (c_off, "off", "#c0392b", "Memory off")):
        p, e = runs[key][1][Q], runs[key][0][Q]
        u, v = world_px(p, m)
        ax.imshow(top)
        ax.plot(u, v, color=col, lw=1.6, solid_capstyle="round", path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ok = e["outcome"] == "success"
        ax.scatter(u[-1], v[-1], marker="*" if ok else "X", s=80, color=col, ec="black", lw=0.5, zorder=5)
        ax.scatter(u[0], v[0], marker="o", s=18, color="black", ec="white", lw=0.7, zorder=6)
        ax.set_xlim(min(u0, u1), max(u0, u1)); ax.set_ylim(max(v0, v1), min(v0, v1))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.98, 0.96, f"{lab}: {e['steps_taken']} steps", transform=ax.transAxes, va="top", ha="right", fontsize=7,
                fontweight="bold", color=col, bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.85))
    c_on.set_title("(c) Radiator search, same start")
    finalize(fig, Path("acra_paper/figures/fig_scene.pdf"), W, H, tight=False)
    print({k: (runs[k][0][Q]["query"], runs[k][0][Q]["steps_taken"], runs[k][0][Q]["outcome"]) for k in runs})


if __name__ == "__main__":
    main()
