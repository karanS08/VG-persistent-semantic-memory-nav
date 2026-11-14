"""DRAFT 2 (2026-10-03): band 2 redrawn to show how memory builds up (window search -> 66 stored objects;
then radiator with memory kept vs wiped). Current teaser: make_teaser.py -> fig_teaser.pdf.
Front-page teaser, two bands (2026-09-29; the 3-band version is make_teaser_v2_3band.py). Every element is real
(band-2 paths: recorded runs, 2026-10-03; earlier drafts used schematic paths):
- band 2 maps: acra_paper/figures/mem_on.png / mem_off.png (author-supplied, 2026-09-29):
  our high-res 00770 render with SCHEMATIC paths drawn on it -- labelled "illustration";
- paths, remembered objects (coloured by true time since last seen), verified
  targets, memory-entry values: the teaser session
  data/runs/phase4_batch_persist_memory_only_teaser_memonly5 (rrd recordings,
  memory_snapshot_ep*.json, batch_metrics.json);
- camera frame + boxes: live_grounding capture in 00770 (DAAAM tracking);
- robot: Spot model rendered in 00770 (illustration; the evaluated agent is a
  bodiless Habitat agent);
- decay curves: Eq. 1 with configs/memory.yaml alpha, lambda_base and the
  mobility priors in configs/mobility_priors.json;
- "fresh start" step counts: factorial reset runs (phase4_batch_reset5*).
Draft 1 (two bands) is kept as scripts/make_teaser_v1.py.
Run: ~/miniconda3/envs/rpv_mem/bin/python scripts/make_teaser.py"""
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

RUN = Path("data/runs/phase4_batch_persist_memory_only_teaser_memonly5")
T = Path("acra_paper/figures/teaser")
LG = Path("acra_paper/figures/live_grounding")
C_OBS, C_MEM, C_NAV, C_VER, C_CON = "#2f6db3", "#2e8b57", "#d9822b", "#c0392b", "#6a3d9a"
QCOL = ["#8c8c8c", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
INK, SOFT = "0.15", "0.4"


def col(rec, ent, comp):
    t = rec.view(index="step", contents=ent).select().read_all()
    return t.column(f"{ent}:{comp}").to_pylist()


def to_px(xyz, m):
    xyz = np.asarray(xyz, dtype=float)
    u, v = (xyz[..., 2], xyz[..., 0]) if m["swap_xz"] else (xyz[..., 0], xyz[..., 2])
    uc, vc = (m["zc"], m["xc"]) if m["swap_xz"] else (m["xc"], m["zc"])
    return (m["sign_u"] * (u - uc) * m["px_per_m"] + m["res"] / 2 + m["dcol"],
            m["sign_v"] * (v - vc) * m["px_per_m"] + m["res"] / 2 + m["drow"])


class Canvas:
    """Inch-based placement on a fixed-size figure."""
    def __init__(self, w, h, oy=0.0):
        self.W, self.H, self.oy = w, h, oy   # oy: inches trimmed off the bottom
        self.fig = plt.figure(figsize=(w, h))

    def r(self, x, y, w, h):
        return [x / self.W, (y - self.oy) / self.H, w / self.W, h / self.H]

    def text(self, x, y, s, **kw):
        kw.setdefault("color", INK)
        return self.fig.text(x / self.W, (y - self.oy) / self.H, s, **kw)

    def box(self, x, y, w, h, fc="white", ec="0.7", lw=0.8, z=-10, rounding=0.03):
        self.fig.patches.append(FancyBboxPatch((x / self.W, (y - self.oy) / self.H), w / self.W, h / self.H,
                                               boxstyle=f"round,pad=0,rounding_size={rounding / self.W}",
                                               transform=self.fig.transFigure, fc=fc, ec=ec, lw=lw, zorder=z))

    def card(self, x, y, w, h, color, title, subtitle):
        self.box(x, y, w, h, ec=color, lw=1.0)
        self.fig.patches.append(Rectangle((x / self.W, (y + h - 0.2 - self.oy) / self.H), w / self.W, 0.2 / self.H,
                                          transform=self.fig.transFigure, fc=color, ec=color, zorder=-9))
        self.text(x + w / 2, y + h - 0.1, title, ha="center", va="center", color="white", fontweight="bold")
        self.text(x + w / 2, y + h - 0.3, subtitle, ha="center", va="center")

    def img(self, x, y, w, h, im, frame=None):
        ax = self.fig.add_axes(self.r(x, y, w, h))
        im = np.asarray(im); ih_, iw_ = im.shape[:2]; want = w / h          # centre-crop, never stretch
        if iw_ / ih_ > want:
            c = int(ih_ * want); im = im[:, (iw_ - c) // 2:(iw_ - c) // 2 + c]
        else:
            c = int(iw_ / want); im = im[(ih_ - c) // 2:(ih_ - c) // 2 + c]
        ax.imshow(im, aspect="auto"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(frame is not None); sp.set_edgecolor(frame or "none"); sp.set_linewidth(1.2)
        return ax

    def arrow(self, pts, color="0.35", lw=1.2, head=True):
        xs = [p[0] / self.W for p in pts]; ys = [(p[1] - self.oy) / self.H for p in pts]
        for i in range(len(pts) - 1):
            last = head and i == len(pts) - 2
            self.fig.patches.append(FancyArrowPatch((xs[i], ys[i]), (xs[i + 1], ys[i + 1]), transform=self.fig.transFigure,
                                                    arrowstyle="-|>" if last else "-", mutation_scale=9,
                                                    color=color, lw=lw, zorder=5))


def bubble(ax, x, y, s, color, dx=0, dy=0):
    ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", ha="center", va="center", color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=0.9),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8) if (dx or dy) else None, zorder=9)


def main():
    eps = json.load(open(RUN / "batch_metrics.json"))["episodes"]
    snaps = [json.load(open(RUN / f"memory_snapshot_ep{e['episode_id']}.json")) for e in eps]
    paths = []
    for e in eps:
        rec = rd.load_recording(str(RUN / f"episode_{e['episode_id']}.rrd"))
        paths.append(np.array([p[0] for p in col(rec, "/world/agent/position", "Position3D") if p]))
    mapping = json.load(open(T / "topdown_00770.json"))
    top = np.array(Image.open(T / "topdown_00770.png"))
    frame = np.array(Image.open(LG / "raw_frame_matched_3x.png"))   # render_highres_view.py (same pose, 3x)
    entrance = np.array(Image.open("acra_paper/figures/room_survey_final/room1_entrance.png").convert("RGB"))
    tracks = {t["id"]: [3 * v for v in t["bbox"]] for t in json.load(open(LG / "tracks_matched.json"))}   # 640x480 -> 3x
    from PIL import ImageEnhance
    spot = np.array(ImageEnhance.Contrast(ImageEnhance.Brightness(Image.open("acra_paper/figures/spot_third_person.png").convert("RGB")).enhance(1.45)).enhance(1.1))
    sh, sw = spot.shape[:2]; spot = spot[int(sh * 40 / 720):int(sh * 700 / 720), int(sw * 230 / 1280):int(sw * 1050 / 1280)]
    mob = json.load(open("configs/mobility_priors.json"))["categories"]
    rel = yaml.safe_load(open("configs/memory.yaml"))["reliability"]
    reset = [json.load(open(f))["episodes"] for f in sorted(Path("data/runs").glob("phase4_batch_reset5*/batch_metrics.json"))]
    reset_tot = [sum(e["steps_taken"] for e in r) for r in reset]
    rad_fresh = [e["steps_taken"] for r in reset for e in r if e["query"] == "a wall-mounted radiator"]
    ver = [e["verification_detail"] or {} for e in eps]
    # memory after query 4: the last snapshot before DAAAM object-number collisions
    # overwrite entries (memory-only mode restarts DAAAM each query; PROGRESS.md 2026-09-29)
    final = snaps[3]
    ents = [e for e in final["entities"] if e["position_world"]]
    age_min = np.array([(final["now"] - e["last_observed"]) / 60 for e in ents])
    rad_e = next((e for e in snaps[1]["entities"] if e["id"] == ver[1].get("entity_id")), None)

    # real recorded pair (2026-10-03): the radiator search from the same start, memory kept from an
    # earlier search (on) vs memory reset (off), both verifying the SAME radiator (instance-gated,
    # check_same_object.py). Maps: scripts/make_real_onoff_maps.py -> mem_on_real.png / mem_off_real.png
    pair = {k: json.load(open(f"data/runs/{r}/batch_metrics.json"))["episodes"][1]["steps_taken"]
            for k, r in (("on", "phase4_batch_persist_memory_only_instance_on"), ("off", "phase4_batch_instance_off"))}
    on4, off4 = pair["on"], pair["off"]

    cv = Canvas(FULL, 3.97)   # 2 bands (2026-09-29): component band + decay panel dropped
    # ======================= band 1: the loop =======================
    margin, gap, by, bh = 0.14, 0.16, 2.29, 1.62
    cw = (FULL - 2 * margin - 4 * gap) / 5              # five cards fit inside the page with margins
    xs = [margin + i * (cw + gap) for i in range(5)]
    iy, ih = by + 0.37, bh - 0.2 - 0.2 - 0.42
    specs = [(C_OBS, "1. Observe", "RGB-D view"), (C_MEM, "2. Remember", "Add to object memory"),
             (C_NAV, "3. Navigate", "Memory picks the target"), (C_VER, "4. Verify", "Check with fresh view"),
             (C_CON, "5. Continue", "Memory persists")]
    for x, (c, t_, s) in zip(xs, specs):
        cv.card(x, by, cw, bh, c, t_, s)
    for i in range(4):
        cv.arrow([(xs[i] + cw + 0.02, by + bh / 2), (xs[i + 1] - 0.02, by + bh / 2)])
    # 1 observe
    a = cv.img(xs[0] + 0.06, iy, cw - 0.12, ih, frame)
    for tid, bb in tracks.items():
        key = tid in (3, 4)
        a.add_patch(Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False,
                              ec="#ffd700" if key else "#7CFC00", lw=1.0 if key else 0.5))
    for tid, lab, va, off in ((4, "window", "top", 4), (3, "radiator", "bottom", -4)):
        a.text(tracks[tid][0] + 12, tracks[tid][1] + 3 * off, lab, va=va, bbox=dict(fc="#ffd700", ec="none", pad=0.8))
    cv.text(xs[0] + cw / 2, by + 0.2, "Open-vocabulary\ndetection and captions", ha="center", va="center")
    # 2 remember
    bb = tracks[3]; rad_crop = frame[int(bb[1]) - 42:int(bb[3]) + 18, int(bb[0]) - 36:int(bb[2]) + 36]
    cv.img(xs[1] + 0.06, iy, cw - 0.12, ih, rad_crop)   # photo only (field list removed 2026-09-29)
    cv.text(xs[1] + cw / 2, by + 0.2, "Kept after it leaves view", ha="center", va="center")
    # 3 navigate
    n = cv.img(xs[2] + 0.06, iy, cw - 0.12, ih, spot)
    n.text(0.5, 0.9, "Find the radiator", transform=n.transAxes, ha="center", va="center", color=INK,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_NAV, lw=0.9))
    cv.text(xs[2] + cw / 2, by + 0.2, f"Found in {on4} steps\n({off4} without memory)",
            ha="center", va="center")
    # 4 verify
    bw = tracks[4]; win = frame[int(bw[1]):int(bw[3]), int(bw[0]):int(bw[2])]
    for k, (im, ok, lab) in enumerate(((rad_crop, True, "“radiator”"), (win, False, "“mirror”?"))):
        ax = cv.img(xs[3] + 0.08 + k * 0.57, iy + 0.13, 0.5, ih - 0.13, im, frame="#2e8b57" if ok else "#c0392b")
        ax.text(0.92, 0.08, "✓" if ok else "✗", transform=ax.transAxes, ha="right", va="bottom", color="white",
                fontweight="bold", bbox=dict(boxstyle="circle,pad=0.15", fc="#2e8b57" if ok else "#c0392b", ec="none"))
        cv.text(xs[3] + 0.33 + k * 0.57, iy + 0.05, lab, ha="center", va="center")
    cv.text(xs[3] + cw / 2, by + 0.2, "\u2713 Right: accept\n\u2717 Wrong: trust it less", ha="center", va="center")
    # 5 continue
    tot = sum(e["steps_taken"] for e in eps)
    reset_avg = round(sum(reset_tot) / len(reset_tot), -2)   # same 5 searches, memory reset each time (real runs)
    cv.text(xs[4] + cw / 2, iy + ih - 0.05, "The next search\nstarts with it", ha="center", va="top", fontweight="bold")
    cv.text(xs[4] + cw / 2, iy + 0.25, f"5 searches in a row:\n{tot} steps with memory,\nabout {reset_avg:,.0f} without",
            ha="center", va="center", linespacing=1.25)
    cv.text(xs[4] + cw / 2, by + 0.2, "No need to search\nthe house again", ha="center", va="center")
    # loop arrow under band 1
    ly = by - 0.13
    cv.text(FULL / 2, ly, "  A persistent perception–memory–navigation loop  ", ha="center", va="center",
            fontweight="bold", color="#1f3b73", bbox=dict(fc="white", ec="none", pad=2.5), zorder=30)

    my, mh = 0.05, 2.0
    # ======================= band 2: how memory builds up (draft 2, 2026-10-03) =======================
    # One real two-search session in 00770, same start. Search 1 (window): memory empty at the start,
    # 66 objects stored by the end. Search 2 (radiator): memory ON starts with those 66 objects, memory
    # OFF starts empty (reset). Dots = objects stored in memory at the START of that search (panel 1: at the
    # END, i.e. what the search added). Paths = recorded robot trajectories; radiator ring = DAAAM's estimate.
    from make_memory_onoff_figure import load
    import matplotlib.patheffects as pe
    ON_RUN = Path("data/runs/phase4_batch_persist_memory_only_instance_on")
    OFF_RUN = Path("data/runs/phase4_batch_instance_off")
    eps_on, paths_on = load(ON_RUN)
    eps_off, paths_off = load(OFF_RUN)
    mem_pts = np.array([e["position_world"] for e in json.load(open(ON_RUN / "memory_snapshot_ep0.json"))["entities"]
                        if e["position_world"]])
    RAD = np.array([-9.379, 1.121, 6.804])            # radiator entry verified by both runs (DAAAM estimate)
    hr = json.load(open("acra_paper/teaser_assets/highres/map_topdown_house_highres.json"))
    base0 = Image.open("acra_paper/teaser_assets/highres/map_topdown_house_black.png").convert("RGB")
    bk = 1811 / base0.size[0]
    base = np.array(base0.resize((1811, 925), Image.LANCZOS))
    mp = json.load(open("acra_paper/figures/teaser/topdown_00770.json"))

    def px(P):
        u, v = to_px(np.asarray(P, float), mp)
        return (u * hr["scale_vs_1600_render"] - hr["crop_origin_px"][0]) * bk, \
               (v * hr["scale_vs_1600_render"] - hr["crop_origin_px"][1]) * bk

    CROP = (610, 430, 1530, 910)                      # x0, y0, x1, y1 in the 1811 x 925 map
    mb, mt, mgap = 0.14, mh, 0.2
    pw = (FULL - 2 * margin - 2 * 0.08 - 2 * mgap) / 3
    ph = pw * (CROP[3] - CROP[1]) / (CROP[2] - CROP[0])
    NAVY, GREEN, RED = "#1f3b73", "#1e8449", "#c0392b"

    def panel(i, title, sub, colour, path, dots, ring=None, marks=(), end_label=None):
        x = margin + 0.08 + i * (pw + mgap)
        y = my + 0.06
        ax_ = cv.fig.add_axes(cv.r(x, y, pw, ph))
        ax_.imshow(base); ax_.set_xlim(CROP[0], CROP[2]); ax_.set_ylim(CROP[3], CROP[1])
        ax_.set_xticks([]); ax_.set_yticks([]); ax_.grid(False)
        for sp in ax_.spines.values():
            sp.set_visible(False)
        if dots is not None:
            dx, dy = px(dots)
            ax_.scatter(dx, dy, s=7, color="white", ec="black", lw=0.35, zorder=3)
        u, v = px(path)
        ax_.plot(u, v, color="white", lw=3.4, solid_capstyle="round", zorder=4)
        ax_.plot(u, v, color=colour, lw=2.0, solid_capstyle="round", zorder=5)
        ax_.scatter(u[0], v[0], s=22, color="black", ec="white", lw=1.0, zorder=7)
        ax_.text(u[0] - 8, v[0], "start", ha="right", va="center", fontsize=5.5, color="white", fontweight="bold",
                 bbox=dict(fc="black", ec="none", pad=0.8), zorder=8)
        ax_.scatter(u[-1], v[-1], s=34, color=colour, ec="white", lw=1.2, zorder=8)
        if end_label:
            ax_.text(u[-1] + 9, v[-1] + 3, end_label, ha="left", va="center", fontsize=5.5, color="white",
                     fontweight="bold", bbox=dict(fc="black", ec="none", pad=0.8), zorder=9)
        if ring is not None:
            rx, ry = px(ring)
            ax_.scatter(rx, ry, s=60, facecolors="none", edgecolors="white", lw=1.2, zorder=6)
            ax_.scatter(rx, ry, s=60, facecolors="none", edgecolors=colour, lw=0.5, zorder=6)
            ax_.text(rx - 10, ry - 12, "radiator", ha="right", va="bottom", fontsize=5.5, color="white",
                     fontweight="bold", bbox=dict(fc="black", ec="none", pad=0.8), zorder=9)
        for (pt, txt, dxy, ha) in marks:
            mx, my_ = px(pt)
            ax_.scatter(mx, my_, s=20, marker="D", color="#ffd700", ec="black", lw=0.5, zorder=9)
            ax_.text(mx + dxy[0], my_ + dxy[1], txt, ha=ha, va="center", fontsize=5.2, color="black",
                     bbox=dict(fc="#ffd700", ec="none", pad=0.7), zorder=10)
        cv.text(x, my + 0.06 + ph + 0.1, title, va="bottom", fontweight="bold", color=colour, fontsize=7)
        cv.text(x, my + 0.06 + ph + 0.1 + 0.15, sub, va="bottom", color="0.25", fontsize=6.3, linespacing=1.1)
        return x

    cv.box(margin, my, FULL - 2 * margin, mh, fc="#f3f6fa", ec="#c9d3e0")
    x1 = panel(0, "1  Search for the window", "Memory starts empty. Every object\nseen is kept: 66 stored by the end.",
               NAVY, paths_on[0], mem_pts, end_label="window")
    x2 = panel(1, f"2  Radiator, memory on: {eps_on[1]['steps_taken']} steps", "Starts with those 66 objects, so it\nknows where to go and goes there.",
               GREEN, paths_on[1], mem_pts, ring=RAD)
    pass25, enter149 = paths_off[1][25], paths_off[1][149]
    x3 = panel(2, f"2  Radiator, memory off: {eps_off[1]['steps_taken']} steps", "Memory wiped: starts empty, so it\nmust search for the radiator again.",
               RED, paths_off[1], None, ring=RAD,
               marks=[(pass25, "step 25: passes it, not yet remembered", (10, 10), "left"),
                      (enter149, "step 149: now remembered", (10, -10), "left")])
    cv.arrow([(x1 + pw + 0.02, my + 0.06 + ph / 2), (x2 - 0.02, my + 0.06 + ph / 2)], color=GREEN, lw=1.6)
    cv.text(x1 + pw + mgap / 2 + 0.0, my + 0.06 + ph / 2 + 0.07, "keeps\nmemory", ha="center", va="bottom", fontsize=5.5,
            color=GREEN, fontweight="bold", linespacing=1.0)
    cv.text(x2 + pw + mgap / 2, my + 0.06 + ph / 2, "vs", ha="center", va="center", fontsize=7, color="0.3", fontweight="bold")
    finalize(cv.fig, Path("acra_paper/figures/fig_teaser_draft2.pdf"), cv.W, cv.H, tight=False)
    print("objects", len(ents), "age range min", round(age_min.min(), 1), round(age_min.max(), 1),
          "targets with position", sum(bool(v.get("entity_position_world")) for v in ver))


if __name__ == "__main__":
    main()
