"""Front-page teaser (2026-09-29). Every element is real simulation output:
- floor plan: orthographic render of HM3D 00770 (render_topdown_00770.py) with
  its fitted world->pixel mapping (IoU fit against the navmesh);
- paths / remembered objects / verified targets: .rrd recordings + metrics of
  memory-only session B rep2 (phase4_batch_persist_memory_only_persistmemoryonly5_rep2);
- camera frame + boxes: live_grounding capture in 00770 (DAAAM tracking boxes);
- memory card values: that run's verification_detail for entity #49 + the
  mobility prior table; robot: Spot model rendered in 00770 (illustration);
- decay curves: Eq. 1 with the configured alpha, lambda_base and mobility priors;
- step counts: factorial runs (reset5*, this session).
Objects are coloured by the query in which they first entered memory (the
recordings do not store per-object last-seen times)."""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rerun.dataframe as rd
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

sys.path.insert(0, "scripts")
from paper_figstyle import FULL, finalize  # noqa: E402

RUN = Path("data/runs/phase4_batch_persist_memory_only_persistmemoryonly5_rep2")
T = Path("acra_paper/figures/teaser")
LG = Path("acra_paper/figures/live_grounding")
QCOL = ["#6b6b6b", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
HEAD = ["#2f6db3", "#2e8b57", "#d9822b", "#c0392b", "#6a3d9a"]


def col(rec, ent, comp):
    t = rec.view(index="step", contents=ent).select().read_all()
    return t.column(f"{ent}:{comp}").to_pylist()


def load_session():
    eps = json.load(open(RUN / "batch_metrics.json"))["episodes"]
    paths, first_seen, objs = [], {}, None
    for e in eps:
        rec = rd.load_recording(str(RUN / f"episode_{e['episode_id']}.rrd"))
        paths.append(np.array([p[0] for p in col(rec, "/world/agent/position", "Position3D") if p]))
        P, L = col(rec, "/world/daaam/objects", "Position3D")[-1], col(rec, "/world/daaam/objects", "Text")[-1]
        for p, lab in zip(P, L):
            first_seen.setdefault(lab, (e["episode_id"], p))
        objs = (np.array(P), L)
    targets = []
    for e in eps:
        d = e["verification_detail"]["description"]
        hit = [i for i, t in enumerate(objs[1]) if t and d.startswith(t[:50])]
        targets.append(objs[0][hit[0]] if hit else None)
    return eps, paths, first_seen, targets


def to_px(xyz, m):
    u, v = (xyz[..., 2], xyz[..., 0]) if m["swap_xz"] else (xyz[..., 0], xyz[..., 2])
    uc, vc = (m["zc"], m["xc"]) if m["swap_xz"] else (m["xc"], m["zc"])
    return (m["sign_u"] * (u - uc) * m["px_per_m"] + m["res"] / 2 + m["dcol"],
            m["sign_v"] * (v - vc) * m["px_per_m"] + m["res"] / 2 + m["drow"])


def img_ax(fig, rect, img, title=None, color=None):
    ax = fig.add_axes(rect); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def main():
    eps, paths, first_seen, targets = load_session()
    mapping = json.load(open(T / "topdown_00770.json"))
    top = np.array(Image.open(T / "topdown_00770.png"))
    frame = np.array(Image.open(LG / "raw_frame_matched.png"))
    tracks = {t["id"]: t["bbox"] for t in json.load(open(LG / "tracks_matched.json"))}
    spot = np.array(Image.open("acra_paper/figures/spot_third_person.png"))[:, 250:1030]
    mob = json.load(open("configs/mobility_priors.json"))["categories"]
    rel = yaml.safe_load(open("configs/memory.yaml"))["reliability"]
    rad = eps[1]["verification_detail"]
    reset_steps = [sum(e["steps_taken"] for e in json.load(open(f))["episodes"])
                   for f in sorted(Path("data/runs").glob("phase4_batch_reset5*/batch_metrics.json"))]
    session_steps = sum(e["steps_taken"] for e in eps)

    W, H = FULL, 3.95
    fig = plt.figure(figsize=(W, H))
    fx = lambda inch: inch / W
    fy = lambda inch: inch / H
    # ---------------- top row: the loop ----------------
    pw, gap, y0, ph = 1.28, 0.15, 2.33, 1.5
    titles = ["1. Observe", "2. Remember", "3. Navigate", "4. Verify", "5. Continue"]
    for i, t in enumerate(titles):
        x = 0.05 + i * (pw + gap)
        fig.patches.append(FancyBboxPatch((fx(x), fy(y0)), fx(pw), fy(ph), boxstyle="round,pad=0,rounding_size=0.01",
                                          transform=fig.transFigure, fc="white", ec=HEAD[i], lw=1.0, zorder=-10))
        fig.patches.append(Rectangle((fx(x), fy(y0 + ph - 0.19)), fx(pw), fy(0.19), transform=fig.transFigure,
                                     fc=HEAD[i], ec=HEAD[i], zorder=-9))
        fig.text(fx(x + pw / 2), fy(y0 + ph - 0.095), t, ha="center", va="center", color="white", fontweight="bold")
        if i < 4:
            fig.patches.append(FancyArrowPatch((fx(x + pw + 0.01), fy(y0 + ph / 2)), (fx(x + pw + gap - 0.01), fy(y0 + ph / 2)),
                                               transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, color="0.3"))
    iw, ih = pw - 0.08, 0.86
    iy = y0 + ph - 0.19 - 0.04 - ih
    ty = y0 + 0.2
    # 1 Observe
    a = img_ax(fig, [fx(0.09), fy(iy), fx(iw), fy(ih)], frame)
    for tid, bb in tracks.items():
        key = tid in (3, 4)
        a.add_patch(Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False,
                              ec="#ffd700" if key else "#7CFC00", lw=1.0 if key else 0.5))
    a.text(tracks[4][0] + 3, tracks[4][1] + 3, "window", fontsize=6, va="top", bbox=dict(fc="#ffd700", ec="none", pad=0.6))
    a.text(tracks[3][0] + 3, tracks[3][1] - 3, "radiator", fontsize=6, va="bottom", bbox=dict(fc="#ffd700", ec="none", pad=0.6))
    fig.text(fx(0.05 + pw / 2), fy(ty), "RGB-D view; DAAAM\ndetects and captions", ha="center", va="center")
    # 2 Remember: crop + real entry values
    x2 = 0.05 + (pw + gap)
    bb = tracks[3]; crop = frame[int(bb[1]) - 12:int(bb[3]) + 6, int(bb[0]) - 10:int(bb[2]) + 10]
    img_ax(fig, [fx(x2 + 0.04), fy(iy + 0.38), fx(0.46), fy(0.46)], crop)
    fig.text(fx(x2 + 0.55), fy(iy + ih - 0.02),
             f"radiator (#{rad['entity_id']})\n3D position\nseen {rad['n_i']}×\nmobility {mob['radiator']:.2f}\nreliability {rad['reliability']:.2f}",
             va="top", fontsize=6.3, linespacing=1.25)
    fig.text(fx(x2 + 0.06), fy(iy + 0.3), "“A black, vertical, column-\nstyle radiator …”", va="top", fontsize=6, style="italic", color="0.25")
    fig.text(fx(x2 + pw / 2), fy(ty), "kept after it leaves view", ha="center", va="center")
    # 3 Navigate
    x3 = 0.05 + 2 * (pw + gap)
    n = img_ax(fig, [fx(x3 + 0.04), fy(iy), fx(iw), fy(ih)], spot)
    n.text(0.5, 0.95, "“a wall-mounted radiator”", transform=n.transAxes, ha="center", va="top", fontsize=6,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=HEAD[2], lw=0.8))
    a_steps = [e["steps_taken"] for f in sorted(Path("data/runs").glob("phase4_batch_reset5*/batch_metrics.json"))
               for e in json.load(open(f))["episodes"] if e["query"] == "a wall-mounted radiator"]
    fig.text(fx(x3 + pw / 2), fy(ty), f"memory target: {eps[1]['steps_taken']} steps\n(fresh start {min(a_steps)}–{max(a_steps)})",
             ha="center", va="center")
    # 4 Verify: real accept vs real lexical reject
    x4 = 0.05 + 3 * (pw + gap)
    bw = bb4 = tracks[4]
    win = frame[int(bw[1]):int(bw[3]), int(bw[0]):int(bw[2])]
    for k, (im, ok) in enumerate(((crop, True), (win, False))):
        ax = img_ax(fig, [fx(x4 + 0.06 + k * 0.6), fy(iy + 0.36), fx(0.52), fy(0.5)], im)
        ax.text(0.95, 0.05, "✓" if ok else "✗", transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.1", fc="#2e8b57" if ok else "#c0392b", ec="none"))
    fig.text(fx(x4 + 0.32), fy(iy + 0.3), "radiator", ha="center", va="top")
    fig.text(fx(x4 + 0.92), fy(iy + 0.3), "mirror?", ha="center", va="top")
    fig.text(fx(x4 + pw / 2), fy(ty), "caption must name the\nquery; failure lowers $v_i$", ha="center", va="center")
    # 5 Continue
    x5 = 0.05 + 4 * (pw + gap)
    fig.text(fx(x5 + pw / 2), fy(iy + ih - 0.05), "memory kept\nacross queries", ha="center", va="top", fontweight="bold")
    fig.text(fx(x5 + pw / 2), fy(iy + 0.42), f"5-query session:\n{session_steps} steps with memory\n"
             f"vs {min(reset_steps)}–{max(reset_steps)} reset",
             ha="center", va="center", fontsize=6.5, linespacing=1.3)
    fig.text(fx(x5 + pw / 2), fy(ty), "later queries start from\nwhat was seen before", ha="center", va="center")

    # ---------------- bottom: real floor plan + decay ----------------
    m = fig.add_axes([fx(0.05), fy(0.05), fx(4.9), fy(2.12)])
    m.imshow(top); m.set_xticks([]); m.set_yticks([]); m.grid(False)
    for sp in m.spines.values():
        sp.set_visible(False)
    for lab, (q, p) in first_seen.items():
        px, pz = to_px(np.array(p), mapping)
        m.scatter(px, pz, s=5, color=QCOL[q], ec="white", lw=0.2, zorder=3)
    for i, (p, e) in enumerate(zip(paths, eps)):
        px, pz = to_px(p, mapping)
        m.plot(px, pz, color=QCOL[i], lw=1.3, zorder=4,
               label=f"Q{i + 1} {e['query'].replace('a ', '', 1).replace('wall-mounted ', '').replace('large ', '')} ({e['steps_taken']} steps)")
        if targets[i] is not None:
            tx, tz = to_px(targets[i], mapping)
            m.scatter(tx, tz, marker="*", s=60, color=QCOL[i], ec="black", lw=0.4, zorder=6)
    sx, sz = to_px(paths[1][:1], mapping)
    m.scatter(sx, sz, s=16, color="black", zorder=7, label="start of each query")
    allp = np.concatenate(paths)
    (x0, z0), (x1, z1) = [to_px(q, mapping) for q in (allp.min(0) - 3.0, allp.max(0) + 3.0)]
    m.set_xlim(min(x0, x1), max(x0, x1)); m.set_ylim(max(z0, z1), min(z0, z1))
    m.legend(frameon=True, framealpha=0.9, loc="lower right", fontsize=6, handlelength=1.2, borderpad=0.4)
    m.text(0.01, 0.98, "Remembered objects, coloured by the query in which they were first seen;\n"
           "paths and verified targets (★) of one memory-only session, scene 00770",
           transform=m.transAxes, va="top", fontsize=6.3, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

    d = fig.add_axes([fx(5.45), fy(0.42), fx(1.45), fy(1.55)])
    t = np.logspace(np.log10(60), np.log10(86400 * 2), 200)
    for name, c in (("phone", "#c0392b"), ("chair", "#2f6db3"), ("radiator", "#2e8b57")):
        mu = mob[name]
        C = min(1.0, 1 - math.exp(-rel["alpha"] * 1)) * np.exp(-rel["lambda_base"] * mu * t)
        d.plot(t / 3600, C, color=c, lw=1.3, label=f"{name} ({mu:.2f})")
    d.set_xscale("log"); d.set_yscale("log"); d.set_ylim(1e-3, 0.6)
    d.set_xticks([1 / 60, 1 / 6, 1, 24]); d.set_xticklabels(["1 min", "10 min", "1 h", "1 day"])
    d.set_xlabel("Time since last seen"); d.set_ylabel("Reliability $C_i$")
    d.set_title("Mobility-aware decay (Eq. 1)")
    d.legend(frameon=False, fontsize=6, loc="lower left", title="mobility", title_fontsize=6)
    finalize(fig, Path("acra_paper/figures/fig_teaser.pdf"), W, H, tight=False)
    fig2 = None


if __name__ == "__main__":
    main()
