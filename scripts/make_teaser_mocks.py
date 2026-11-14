"""Three band-2 DRAFTS for the teaser (2026-10-03), built from the current make_teaser.py (band 1 unchanged):
  A  one map (memory filling, then reused) + chart of steps over five searches in a row
  B  chart only
  C  schematic of memory states (empty -> 66 stored -> kept vs wiped)
All numbers are real (data/runs): the five-search sessions (memory kept: persistmemoryonly5 rep1-3; fresh start each
search: reset5 + rep2 + rep3), the 66 stored objects (memory snapshot after search 1) and the radiator pair
(instance_on / instance_off). Out: acra_paper/figures/fig_teaser_mock{A,B,C}.pdf (not used by main.tex).
Run: python scripts/make_teaser_mocks.py"""
import os
import subprocess
import sys
from pathlib import Path

SRC = Path("scripts/make_teaser.py").read_text()
_c0 = SRC.index("    # 1 observe")
_c1 = SRC.index("    # 2 remember")
# card 1: the corner-of-the-room view with real DAAAM detections (acra_paper/figures/old_scene_card.png, 2026-10-03)
SRC = SRC[:_c0] + """    # 1 observe
    cv.img(xs[0] + 0.06, iy, cw - 0.12, ih, np.array(Image.open("acra_paper/figures/old_scene_card.png").convert("RGB")))
    cv.text(xs[0] + cw / 2, by + 0.2, "Open-vocabulary\\ndetection and captions", ha="center", va="center")
""" + SRC[_c1:]
a = SRC.index("    # ======================= band 2: the world =======================")
b = SRC.index('    finalize(cv.fig, Path("acra_paper/figures/fig_teaser.pdf")')

PRELUDE = '''    import os
    my, mh = 0.05, 2.0
    import matplotlib.patheffects as pe
    from make_memory_onoff_figure import load
    GREEN, RED, NAVY = "#1e8449", "#c0392b", "#1f3b73"
    ON5 = [f"phase4_batch_persist_memory_only_persistmemoryonly5{s}" for s in ("_rep1", "_rep2", "_rep3")]
    OFF5 = ["phase4_batch_reset5", "phase4_batch_reset5_rep2", "phase4_batch_reset5_rep3"]
    NAMES = ["window", "radiator", "cabinet", "wall vent", "bench"]
    steps_of = lambda r: [e["steps_taken"] for e in json.load(open(f"data/runs/{r}/batch_metrics.json"))["episodes"]]
    S_on = np.array([steps_of(r) for r in ON5]); S_off = np.array([steps_of(r) for r in OFF5])
    TS = "data/runs/phase4_batch_persist_memory_only_teaser_memonly5"
    stored = [len([e for e in json.load(open(f"{TS}/memory_snapshot_ep{k}.json"))["entities"] if e["position_world"]])
              for k in range(5)]
    ON_RUN = Path("data/runs/phase4_batch_persist_memory_only_instance_on")
    OFF_RUN = Path("data/runs/phase4_batch_instance_off")
    eps_on, paths_on = load(ON_RUN); eps_off, paths_off = load(OFF_RUN)
    mem_pts = np.array([e["position_world"] for e in json.load(open(ON_RUN / "memory_snapshot_ep0.json"))["entities"]
                        if e["position_world"]])
    RAD = np.array([-9.379, 1.121, 6.804])
    hr = json.load(open("acra_paper/teaser_assets/highres/map_topdown_house_highres.json"))
    base0 = Image.open("acra_paper/teaser_assets/highres/map_topdown_house_black.png").convert("RGB")
    bk = 1811 / base0.size[0]
    BS = 2                                   # embed the map at 2x (3622 x 1850 before cropping): ~1,700 ppi at print size
    base = np.array(base0.resize((1811 * BS, 925 * BS), Image.LANCZOS))
    mp = json.load(open("acra_paper/figures/teaser/topdown_00770.json"))
    CROP = (610, 430, 1530, 910)

    def px(P):
        u, v = to_px(np.asarray(P, float), mp)
        return (u * hr["scale_vs_1600_render"] - hr["crop_origin_px"][0]) * bk, \\
               (v * hr["scale_vs_1600_render"] - hr["crop_origin_px"][1]) * bk

    def tag(ax_, x, y, s, ha="left", fc="black", tc="white", fs=5.5):
        ax_.text(x, y, s, ha=ha, va="center", fontsize=fs, color=tc, fontweight="bold", bbox=dict(fc=fc, ec="none", pad=0.8), zorder=9)

    def mapax(x, y, w, tint=1.0):
        ph = w * (CROP[3] - CROP[1]) / (CROP[2] - CROP[0])
        ax_ = cv.fig.add_axes(cv.r(x, y, w, ph))
        ax_.imshow(base[CROP[1] * BS:CROP[3] * BS, CROP[0] * BS:CROP[2] * BS], extent=(CROP[0], CROP[2], CROP[3], CROP[1]))
        ax_.set_xlim(CROP[0], CROP[2]); ax_.set_ylim(CROP[3], CROP[1]); ax_.set_xticks([]); ax_.set_yticks([]); ax_.grid(False)
        for sp in ax_.spines.values():
            sp.set_visible(False)
        return ax_, ph

    def dots(ax_, pts, s=7, alpha=1.0):
        dx, dy = px(pts); ax_.scatter(dx, dy, s=s, color="white", ec="black", lw=0.35, zorder=3, alpha=alpha)

    def path(ax_, p, col, lw=2.0, alpha=1.0):
        u, v = px(p)
        ax_.plot(u, v, color="white", lw=lw + 1.4, solid_capstyle="round", zorder=4, alpha=alpha)
        ax_.plot(u, v, color=col, lw=lw, solid_capstyle="round", zorder=5, alpha=alpha)
        return u, v

    def steps_chart(ax):
        x = np.arange(5); w = 0.36
        ax.bar(x - w / 2, S_on.mean(0), w, color=GREEN, label="Memory kept between searches")
        ax.bar(x + w / 2, S_off.mean(0), w, color="#d98880", label="Fresh start every search")
        for j in range(3):
            ax.scatter(x - w / 2, S_on[j], s=5, color="white", ec="black", lw=0.5, zorder=4)
            ax.scatter(x + w / 2, S_off[j], s=5, color="white", ec="black", lw=0.5, zorder=4)
        for i in range(5):
            ax.text(i - w / 2, S_on.mean(0)[i] + 14, f"{S_on.mean(0)[i]:.0f}", ha="center", fontsize=5.5, color=GREEN, fontweight="bold")
            ax.text(i + w / 2, S_off.mean(0)[i] + 14, f"{S_off.mean(0)[i]:.0f}", ha="center", fontsize=5.5, color=RED, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([f"{i + 1} {n}\\n{stored[i]} stored" for i, n in enumerate(NAMES)], fontsize=5.5)
        ax.set_ylim(0, 600); ax.set_yticks([0, 250, 500]); ax.tick_params(labelsize=5.5, length=2)
        ax.set_ylabel("steps to find it", fontsize=6, labelpad=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=5.5, loc="upper right", handlelength=1, borderaxespad=0, ncol=1)

    cv.box(margin, my, FULL - 2 * margin, mh, fc="#f3f6fa", ec="#c9d3e0")
'''

BLOCKS = {}
BLOCKS["A"] = '''
    mw = 3.0
    ax_, ph = mapax(margin + 0.1, my + 0.1, mw)
    dots(ax_, mem_pts); path(ax_, paths_on[0], NAVY, 1.1, 0.7)
    u, v = path(ax_, paths_on[1], GREEN, 2.2)
    ax_.scatter(u[0], v[0], s=22, color="black", ec="white", lw=1.0, zorder=7); tag(ax_, u[0] - 9, v[0], "start", "right")
    rx, ry = px(RAD); ax_.scatter(rx, ry, s=60, facecolors="none", edgecolors="white", lw=1.3, zorder=6)
    tag(ax_, rx - 9, ry - 12, "radiator", "right")
    ax_.scatter(u[-1], v[-1], s=34, color=GREEN, ec="white", lw=1.2, zorder=8)
    cv.text(margin + 0.1, my + 0.1 + ph + 0.1, "Memory fills up during search 1 (66 objects, dots) and is\\nreused in search 2: straight to the radiator, 12 steps.",
            va="bottom", fontsize=6.3, color="0.2", linespacing=1.15)
    cv.text(margin + 0.1, my + mh - 0.07, "What memory holds", va="top", fontweight="bold", color=NAVY, fontsize=7)
    cx = margin + 0.1 + mw + 0.5
    cax = cv.fig.add_axes(cv.r(cx, my + 0.4, FULL - margin - 0.15 - cx, 1.3))
    steps_chart(cax)
    cv.text(cx, my + mh - 0.07, "What it buys: five searches in a row (3 repeats)", va="top", fontweight="bold", color=NAVY, fontsize=7)
    cv.text(cx, my + 0.06, "Search 1 is the same: memory starts empty. Searches 2\\u20135 average 46 steps vs 348.", va="bottom",
            fontsize=6, color="0.25")
'''
BLOCKS["B"] = '''
    cax = cv.fig.add_axes(cv.r(margin + 0.45, my + 0.42, FULL - 2 * margin - 0.65, 1.2))
    steps_chart(cax)
    cv.text(margin + 0.1, my + mh - 0.07, "Memory is kept between searches, so every search after the first is much cheaper", va="top",
            fontweight="bold", color=NAVY, fontsize=7)
    cv.text(margin + 0.1, my + 0.06, "Steps to find each object over five searches in a row (bars: mean of 3 runs; dots: runs; 500 = step budget). "
            "Searches 2\\u20135 average 46 steps with memory vs 348 with a fresh start.", va="bottom", fontsize=5.8, color="0.25")
'''
BLOCKS["C"] = '''
    OX = float(os.environ.get("OX", "0"))
    gx = margin + 0.15 + OX
    mw = 1.1
    py = my + 0.5
    def mini(x, y, pts, border, p=None, pcol=None):
        ax_, ph = mapax(x, y, mw)          # full colour (no fade)
        if pts is not None:
            dots(ax_, pts, s=4, alpha=0.6)
        if p is not None:                  # recorded robot path: thin line, light -> dark with time so overlaps stay readable
            from matplotlib.collections import LineCollection
            from matplotlib.colors import to_rgb
            u, v = px(p)
            pts = np.column_stack([u, v]); segs = np.stack([pts[:-1], pts[1:]], axis=1)
            c = np.array(to_rgb(pcol)); t = np.linspace(0.0, 1.0, len(segs))[:, None] ** 0.8
            cols = (1 - (0.55 + 0.45 * t)) * np.ones(3) + (0.55 + 0.45 * t) * c      # 55% -> 100% colour
            ax_.add_collection(LineCollection(segs, colors="white", linewidths=2.5, capstyle="round", zorder=4))
            ax_.add_collection(LineCollection(segs, colors=cols, linewidths=1.35, capstyle="round", zorder=5))
            ax_.scatter(u[0], v[0], s=18, color="black", ec="white", lw=0.9, zorder=7)
            ax_.scatter(u[-1], v[-1], s=26, color=pcol, ec="white", lw=1.1, zorder=8)
        for sp in ax_.spines.values():
            sp.set_visible(True); sp.set_edgecolor(border); sp.set_linewidth(1.2)
        return ph
    ph = mini(gx, py + 0.2, None, "0.5")
    cv.text(gx, py + 0.2 + ph + 0.05, "Search 1: window\\nMemory starts empty", va="bottom", fontsize=6.3, fontweight="bold", color="0.25", linespacing=1.1)
    x2 = gx + mw + 0.35
    mini(x2, py + 0.2, mem_pts, NAVY)
    cv.text(x2, py + 0.2 + ph + 0.05, "After search 1\\n66 objects stored", va="bottom", fontsize=6.3, fontweight="bold", color=NAVY, linespacing=1.1)
    cv.arrow([(gx + mw + 0.03, py + 0.2 + ph / 2), (x2 - 0.03, py + 0.2 + ph / 2)], color="0.4", lw=1.4)
    x3 = x2 + mw + 0.55
    ya, yb = py + 0.62, py - 0.18
    mini(x3, ya, mem_pts, GREEN, paths_on[1], GREEN); mini(x3, yb, None, RED, paths_off[1], RED)
    cv.arrow([(x2 + mw + 0.03, py + 0.2 + ph / 2 + 0.05), (x3 - 0.03, ya + ph / 2)], color=GREEN, lw=1.4)
    cv.arrow([(x2 + mw + 0.03, py + 0.2 + ph / 2 - 0.05), (x3 - 0.03, yb + ph / 2)], color=RED, lw=1.4)
    t_on = cv.text(x3 + mw + 0.12, ya + ph / 2, f"Search 2: radiator\\n\\u2713 Memory kept: starts with the\\n66 objects \\u2192 {eps_on[1]['steps_taken']} steps", va="center", fontsize=6.3, color=GREEN, fontweight="bold", linespacing=1.15)
    t_off = cv.text(x3 + mw + 0.12, yb + ph / 2, f"Search 2: radiator\\n\\u2717 Fresh start: memory wiped,\\nstarts empty \\u2192 {eps_off[1]['steps_taken']} steps", va="center", fontsize=6.3, color=RED, fontweight="bold", linespacing=1.15)
    _r = cv.fig.canvas.get_renderer()
    print("GROUP", margin + 0.15, max(t.get_window_extent(_r).x1 for t in (t_on, t_off)) / cv.fig.dpi, flush=True)
    cv.text(margin + 0.1, my + mh - 0.07, "Memory is built while searching, and either kept or wiped before the next search", va="top", fontweight="bold", color=NAVY, fontsize=7)
    cv.text(margin + 0.1, my + 0.06, "Dots: objects stored in memory (real data, scene 00770). Steps are from one recorded two-search session.", va="bottom", fontsize=5.8, color="0.25")
'''

for k, blk in BLOCKS.items():
    out = SRC[:a] + PRELUDE + blk + SRC[b:]
    out = out.replace('Path("acra_paper/figures/fig_teaser.pdf")', f'Path("acra_paper/figures/fig_teaser_mock{k}.pdf")')
    p = Path(f"/tmp/claude-1004/-home-ksharma-mem-ws/228ff7f4-4e94-4d56-9832-54bf4b393403/scratchpad/mock{k}.py")
    p.write_text(out)
    env = dict(os.environ, OX="0")
    r = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, env=env)
    if k == "C" and r.returncode == 0:
        left, right = [float(v) for v in next(l for l in r.stdout.splitlines() if l.startswith("GROUP")).split()[1:3]]
        ox = 7.0 / 2 - (left + right) / 2          # centre of the 7.0 in page == centre of the band-2 box
        env["OX"] = f"{ox:.4f}"
        print("C group", round(left, 2), "->", round(right, 2), "in; shifting by", round(ox, 3), "in")
        r = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, env=env)
    print(k, "ok" if r.returncode == 0 else "FAILED", (r.stderr or "").strip().splitlines()[-1:] if r.returncode else "")
