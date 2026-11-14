"""Front-page teaser, three bands (2026-09-29). Every element is real except the
band-2 paths (schematic, labelled):
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
    frame = np.array(Image.open(LG / "raw_frame_matched.png"))
    entrance = np.array(Image.open("acra_paper/figures/room_survey_final/room1_entrance.png").convert("RGB"))
    tracks = {t["id"]: t["bbox"] for t in json.load(open(LG / "tracks_matched.json"))}
    spot = np.array(Image.open("acra_paper/figures/spot_third_person.png"))[40:700, 230:1050]
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

    cv = Canvas(FULL, 5.05, oy=0.32)   # footer removed 2026-09-29
    # ======================= band 1: the loop =======================
    cw, gap, by, bh = 1.25, 0.18, 3.62, 1.62
    xs = [0.1 + i * (cw + gap) for i in range(5)]
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
        a.text(tracks[tid][0] + 4, tracks[tid][1] + off, lab, va=va, bbox=dict(fc="#ffd700", ec="none", pad=0.8))
    cv.text(xs[0] + cw / 2, by + 0.2, "open-vocabulary detection\nand captions (DAAAM)", ha="center", va="center")
    # 2 remember
    bb = tracks[3]; rad_crop = frame[int(bb[1]) - 14:int(bb[3]) + 6, int(bb[0]) - 12:int(bb[2]) + 12]
    cv.img(xs[1] + 0.06, iy + 0.1, 0.4, ih - 0.1, rad_crop)
    seen_ago = (snaps[1]["now"] - rad_e["last_observed"]) / 60 if rad_e else None
    fields = ["radiator", "3D position", f"seen {rad_e['n_i'] if rad_e else ver[1].get('n_i')}\u00d7",
              f"mobility {mob['radiator']:.2f}", f"$C_i$ = {ver[1].get('reliability', 0):.2f}"]
    if seen_ago is not None:
        fields.append(f"{seen_ago:.0f} min ago")
    cv.text(xs[1] + 0.52, iy + ih, "\n".join(fields), va="top", linespacing=1.15)
    cv.text(xs[1] + cw / 2, by + 0.2, "kept after it leaves view", ha="center", va="center")
    # 3 navigate
    n = cv.img(xs[2] + 0.06, iy, cw - 0.12, ih, spot)
    n.text(0.5, 0.9, "Find the radiator", transform=n.transAxes, ha="center", va="center", color=INK,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_NAV, lw=0.9))
    cv.text(xs[2] + cw / 2, by + 0.2, f"{eps[1]['steps_taken']} steps from memory\n"
            f"(fresh start {min(rad_fresh)}–{max(rad_fresh)})", ha="center", va="center")
    # 4 verify
    bw = tracks[4]; win = frame[int(bw[1]):int(bw[3]), int(bw[0]):int(bw[2])]
    for k, (im, ok, lab) in enumerate(((rad_crop, True, "“radiator”"), (win, False, "“mirror”?"))):
        ax = cv.img(xs[3] + 0.08 + k * 0.57, iy + 0.13, 0.5, ih - 0.13, im, frame="#2e8b57" if ok else "#c0392b")
        ax.text(0.92, 0.08, "✓" if ok else "✗", transform=ax.transAxes, ha="right", va="bottom", color="white",
                fontweight="bold", bbox=dict(boxstyle="circle,pad=0.15", fc="#2e8b57" if ok else "#c0392b", ec="none"))
        cv.text(xs[3] + 0.33 + k * 0.57, iy + 0.05, lab, ha="center", va="center")
    cv.text(xs[3] + cw / 2, by + 0.2, "\u2713 accept    \u2717 reject, $v_i$\u2193", ha="center", va="center")
    # 5 continue
    tot = sum(e["steps_taken"] for e in eps)
    cv.text(xs[4] + cw / 2, iy + ih - 0.05, "same memory,\nnext query", ha="center", va="top", fontweight="bold")
    cv.text(xs[4] + cw / 2, iy + 0.28, f"5 queries:\n{tot} steps\n(reset: {min(reset_tot)}\u2013{max(reset_tot)})",
            ha="center", va="center", linespacing=1.25)
    cv.text(xs[4] + cw / 2, by + 0.2, "later searches reuse\nearlier observations", ha="center", va="center")
    # loop arrow under band 1
    ly = by - 0.13
    cv.arrow([(xs[4] + cw, by + bh * 0.3), (xs[4] + cw + 0.07, by + bh * 0.3), (xs[4] + cw + 0.07, ly),
              (xs[0] - 0.07, ly), (xs[0] - 0.07, by + bh * 0.3), (xs[0], by + bh * 0.3)], color="#1f3b73", lw=1.3)
    cv.text(FULL / 2, ly, "  A persistent perception–memory–navigation loop  ", ha="center", va="center",
            fontweight="bold", color="#1f3b73", bbox=dict(fc="white", ec="none", pad=2.5), zorder=30)

    # ======================= band 2: the world =======================
    my, mh = 1.98, 1.36
    cv.box(0.1, my, 6.8, mh, fc="#f3f6fa", ec="#c9d3e0")
    # memory on / off maps: author-supplied schematic paths on our high-res 00770 render
    # (acra_paper/figures/mem_on.png, mem_off.png) -- illustration, labelled as such
    # step counts are MEASURED (radiator search, query 2: memory-on teaser session vs the
    # reset runs), not the length of the schematic paths
    rad_on = eps[1]["steps_taken"]
    rad_off = next(e["steps_taken"] for e in json.load(open("data/runs/phase4_batch_reset5_rep2/batch_metrics.json"))["episodes"]
                   if e["query"] == "a wall-mounted radiator")
    mh_img = mh - 0.42
    mw = mh_img * 1811 / 925
    for k, (fn, head, n, col_) in enumerate((("mem_on.png", "Memory on", rad_on, "#1e8449"),
                                            ("mem_off.png", "Memory off", rad_off, "#c0392b"))):
        x = 0.18 + k * (mw + 0.14)
        ax_ = cv.fig.add_axes(cv.r(x, my + 0.2, mw, mh_img))
        ax_.imshow(Image.open(f"acra_paper/figures/{fn}")); ax_.set_xticks([]); ax_.set_yticks([]); ax_.grid(False)
        for sp in ax_.spines.values():
            sp.set_visible(False)
        cv.text(x, my + mh - 0.07, f"{head} \u00b7 {n} steps", va="top", fontweight="bold", color=col_)
    cv.text(0.18, my + 0.06, "Illustrative paths; steps measured in the radiator search.", va="bottom", fontsize=6, color="0.35")
    dx0 = 0.18 + 2 * mw + 0.32
    ox, oz = to_px(np.array([e["position_world"] for e in ents]), mapping)   # used by band 3
    # decay panel
    cv.text(dx0, my + mh - 0.08, "Memory is not static: reliability decays\nwith time, faster for objects likely to move.",
            va="top", linespacing=1.2)
    d = cv.fig.add_axes(cv.r(dx0 + 0.35, my + 0.36, 6.8 - dx0 - 0.4, 0.55))
    t = np.logspace(np.log10(60), np.log10(86400 * 3), 200)
    base = min(1.0, 1 - math.exp(-rel["alpha"]))
    for name, c in (("bag", "#c0392b"), ("cushion", "#2f6db3"), ("radiator", "#2e8b57")):
        d.plot(t / 3600, base * np.exp(-rel["lambda_base"] * mob[name] * t), color=c, lw=1.3, label=f"{name} ({mob[name]:.2f})")
    d.set_xscale("log"); d.set_yscale("log"); d.set_ylim(1e-5, 0.6)
    d.set_xticks([1 / 60, 1, 24]); d.set_xticklabels(["1 min", "1 h", "1 day"]); d.set_yticks([1e-4, 1e-2])
    d.set_ylabel("$C_i$", labelpad=1); d.set_xlabel("time since last seen", labelpad=1)
    d.legend(frameon=False, loc="lower left", handlelength=1, borderaxespad=0.1, labelspacing=0.2)

    # ======================= band 3: the components =======================
    ky, kh, kw, kg = 0.42, 1.46, 1.63, 0.093
    kx = [0.1 + i * (kw + kg) for i in range(4)]
    kspec = [(C_OBS, "Semantic memory", "3D, object-centric"), (C_MEM, "Verification", "Four checks before success"),
             (C_NAV, "Navigation with memory", "Memory changes the route"),
             (C_VER, "Mobility sets decay", "Per-category decay")]
    for x, (c, t_, s) in zip(kx, kspec):
        cv.card(x, ky, kw, kh, c, t_, s)
    # semantic memory: 3D scatter of the real memory, coloured by age
    ax3 = cv.fig.add_axes(cv.r(kx[0] - 0.02, ky + 0.15, 0.88, 0.88), projection="3d")
    P = np.array([e["position_world"] for e in ents])
    ax3.scatter(P[:, 0], P[:, 2], P[:, 1], c=age_min, cmap="turbo", s=3, depthshade=False)
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([]); ax3.view_init(28, -60)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.set_alpha(0.15)
    cv.text(kx[0] + 0.86, ky + kh - 0.42, "caption\n3D position\nembedding\nmobility prior\nobservations $n_i$\n"
            "reliability $C_i$\nlast seen", va="top", linespacing=1.15)
    cv.text(kx[0] + kw / 2, ky + 0.08, f"{len(ents)} objects in memory", ha="center", va="bottom")
    # verification
    checks = [("1", "Fresh evidence", "seen in this query"), ("2", "Relevance", "CLIP score above threshold"),
              ("3", "Confirmation", "enough observations"), ("4", "Lexical", "caption names the query")]
    for j, (num, name, desc) in enumerate(checks):
        yy = ky + kh - 0.47 - j * 0.2
        cv.text(kx[1] + 0.2, yy, num, ha="center", va="center", color="white", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.25", fc=C_MEM, ec="none"))
        cv.text(kx[1] + 0.36, yy + 0.04, name, va="center", fontweight="bold")
        cv.text(kx[1] + 0.36, yy - 0.05, desc, va="center", color=SOFT)
    cv.text(kx[1] + kw / 2, ky + 0.08, "memory is a hypothesis, not a fact", ha="center", va="bottom", style="italic")
    # navigation with memory: zoom on the radiator search
    z = cv.fig.add_axes(cv.r(kx[2] + 0.06, ky + 0.28, kw - 0.12, kh - 0.66))
    z.imshow(top); z.set_xticks([]); z.set_yticks([]); z.grid(False)
    for sp in z.spines.values():
        sp.set_visible(False)
    px, pz = to_px(paths[1], mapping)
    z.scatter(ox, oz, c=age_min, cmap="turbo", s=4, zorder=3)
    z.plot(px, pz, color=QCOL[1], lw=1.8, ls=(0, (3, 1.5)), zorder=5)
    if ver[1].get("entity_position_world"):
        tx, tz = to_px(ver[1]["entity_position_world"], mapping)
        z.scatter(tx, tz, marker="*", s=80, color=QCOL[1], ec="black", lw=0.4, zorder=6)
        bubble(z, tx, tz, "use memory:\ngo here", C_NAV, -30, -16)
        xs_ = np.r_[px, tx]; zs_ = np.r_[pz, tz]; pad = 1.8 * mapping["px_per_m"]
        cx, cz = xs_.mean(), zs_.mean()
        hw = max(np.ptp(xs_) / 2 + pad, (np.ptp(zs_) / 2 + pad) * 1.9); hh = hw / 1.9
        z.set_xlim(cx - hw, cx + hw); z.set_ylim(cz + hh, cz - hh)
    z.scatter(px[0], pz[0], s=30, color="#1f3b73", ec="white", lw=1, zorder=7)
    cv.text(kx[2] + kw / 2, ky + 0.08, "remembered radiator, reached\nwithout searching again", ha="center", va="bottom")
    # mobility card: real thumbnails + real curves
    thumbs = [("bag", entrance[355:455, 275:335], "#c0392b"), ("cushion", entrance[378:440, 130:285], "#2f6db3"),
              ("radiator", rad_crop, "#2e8b57")]
    for j, (name, im, c) in enumerate(thumbs):
        x = kx[3] + 0.05 + j * 0.53
        cv.img(x + 0.08, ky + 0.84, 0.34, 0.27, im)
        cv.text(x + 0.25, ky + 0.72, f"{name}\n{mob[name]:.2f}", ha="center", va="center", linespacing=1.0)
        s_ = cv.fig.add_axes(cv.r(x + 0.04, ky + 0.24, 0.44, 0.34))
        s_.plot(t / 3600, base * np.exp(-rel["lambda_base"] * mob[name] * t), color=c, lw=1.3)
        s_.set_xscale("log"); s_.set_ylim(0, 0.45); s_.set_xticks([]); s_.set_yticks([]); s_.grid(False)
    cv.text(kx[3] + kw / 2, ky + 0.08, "forgets fast → remembers long", ha="center", va="bottom")

    finalize(cv.fig, Path("acra_paper/figures/fig_teaser.pdf"), cv.W, cv.H, tight=False)
    print("objects", len(ents), "age range min", round(age_min.min(), 1), round(age_min.max(), 1),
          "targets with position", sum(bool(v.get("entity_position_world")) for v in ver))


if __name__ == "__main__":
    main()
