"""MOCK D (2026-10-03): band 2 = (top) two top-down maps, memory off (empty) vs memory on (every remembered object
as a dot, coloured by type) and (bottom) the same memory in an orthographic 3D (isometric) view, dots at their
estimated 3D positions. Memory shown = the scripted house tour (1,193 objects, real DAAAM captions/positions),
i.e. what memory looks like after exploring the whole house -- labelled as such. Band 1 unchanged.
Out: acra_paper/figures/fig_teaser_mockD.pdf (not used by main.tex)."""
import subprocess
import sys
from pathlib import Path

SRC = Path("scripts/make_teaser.py").read_text()
a = SRC.index("    # ======================= band 2: the world =======================")
b = SRC.index('    finalize(cv.fig, Path("acra_paper/figures/fig_teaser.pdf")')
X = 2.25   # extra height vs the 2-band teaser

BLOCK = '''    my, mh = 0.05, 4.25
    import re
    GREEN, RED, NAVY = "#1e8449", "#c0392b", "#1f3b73"
    TOUR = "data/runs/phase4_batch_persist_memory_only_house_tour/memory_snapshot_ep0.json"
    ents3 = [e for e in json.load(open(TOUR))["entities"] if e["position_world"]]
    CLASSES = [("fixture", ["radiator", "lamp", "light", "chandelier", "fixture", "vent", "switch", "outlet", "socket", "sink", "toilet", "bathtub", "faucet", "thermostat", "oven", "pipe", "tripod", "camera", "heater", "handle", "knob", "hinge"]),
               ("furniture", ["table", "chair", "cabinet", "bench", "drawer", "shelf", "bed", "sofa", "desk", "stool", "wardrobe", "dresser", "headboard", "bookcase", "countertop", "suitcase", "box"]),
               ("opening", ["window", "door", "doorway", "curtain", "arch", "blind"]),
               ("decor", ["painting", "mirror", "picture", "sculpture", "vase", "rug", "carpet", "clock", "plaque", "ornament", "flower", "canvas", "tapestry", "fireplace", "shield", "cross", "bear", "medallion"]),
               ("surface", ["wall", "floor", "ceiling", "panel", "tile", "column", "pillar", "molding", "cornice", "baseboard", "wallpaper", "stair", "beam", "frame", "trim"])]
    COL = {"fixture": "#f28e2b", "furniture": "#4e79a7", "opening": "#17becf", "decor": "#b07aa1", "surface": "#a0a0a0", "other": "#a0a0a0"}
    LABEL = {"fixture": "Fixtures (radiator, lamp, sink)", "furniture": "Furniture (cabinet, table)", "opening": "Openings (window, door)",
             "decor": "Decor (painting, mirror)", "surface": "Walls, floors, trim"}
    def klass(d):
        s = d.lower()[:70]
        words = re.findall(r"[a-z]+", s.split(".")[0].split(" with ")[0].split(" featuring ")[0])
        for cls, keys in CLASSES:
            if words and words[-1] in keys:
                return cls
        for cls, keys in CLASSES:
            if any(k in s[:45] for k in keys):
                return cls
        return "other"
    cls3 = np.array([klass(e["description"]) for e in ents3])
    P3 = np.array([e["position_world"] for e in ents3])
    hr = json.load(open("acra_paper/teaser_assets/highres/map_topdown_house_highres.json"))
    base0 = Image.open("acra_paper/teaser_assets/highres/map_topdown_house_black.png").convert("RGB")
    bk = 1811 / base0.size[0]
    base = np.array(base0.resize((1811, 925), Image.LANCZOS))
    mp = json.load(open("acra_paper/figures/teaser/topdown_00770.json"))
    def px(P):
        u, v = to_px(np.asarray(P, float), mp)
        return (u * hr["scale_vs_1600_render"] - hr["crop_origin_px"][0]) * bk, \\
               (v * hr["scale_vs_1600_render"] - hr["crop_origin_px"][1]) * bk
    def draw_dots(ax_, u, v, big=14, small=4):
        for cls in ("surface", "other", "decor", "furniture", "opening", "fixture"):
            m = cls3 == cls
            if m.any():
                ax_.scatter(u[m], v[m], s=small if cls in ("surface", "other") else big, color=COL[cls], ec="white", lw=0.25,
                            alpha=0.55 if cls in ("surface", "other") else 0.95, zorder=3)
    cv.box(margin, my, FULL - 2 * margin, mh, fc="#f3f6fa", ec="#c9d3e0")
    mw = (FULL - 2 * margin - 0.16 - 0.2) / 2
    ph1 = mw * 925 / 1811
    iw = 4.0
    ih = iw / 2.675
    y0 = my + 0.1
    y1 = y0 + ih + 0.5
    # ---- row 1: top-down, memory off vs memory on
    for k, (head, sub, col_) in enumerate((("Memory off: starts empty", "Nothing is remembered, so every search starts from scratch.", RED),
                                           ("Memory on: objects remembered across the house", "One dot per stored object: caption, 3D position, time seen.", GREEN))):
        x = margin + 0.08 + k * (mw + 0.2)
        ax_ = cv.fig.add_axes(cv.r(x, y1, mw, ph1))
        ax_.imshow(base); ax_.set_xticks([]); ax_.set_yticks([]); ax_.grid(False)
        for sp in ax_.spines.values():
            sp.set_visible(False)
        if k == 1:
            u, v = px(P3); draw_dots(ax_, u, v); ax_.set_xlim(0, 1811); ax_.set_ylim(925, 0)
            picked = []
            for kw, off in (("bathtub", (-60, -26)), ("fireplace", (-70, 40)), ("radiator", (50, 40)), ("cabinet", (-60, 40)), ("mirror", (30, -30))):
                cand = [i for i, e in enumerate(ents3) if kw in e["description"].lower()[:40]]
                if not cand:
                    continue
                i = max(cand, key=lambda j: (ents3[j]["n_i"], -abs(P3[j][0] % 7)))
                ax_.annotate(kw.capitalize(), (u[i], v[i]), xytext=(u[i] + off[0], v[i] + off[1]), fontsize=5.5, color="white", fontweight="bold",
                             ha="center", va="center", bbox=dict(fc="black", ec="none", pad=1.0), zorder=9,
                             arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
        cv.text(x, y1 + ph1 + 0.30, head, va="bottom", fontweight="bold", color=col_, fontsize=7)
        cv.text(x, y1 + ph1 + 0.07, sub, va="bottom", color="0.25", fontsize=6)
    # ---- row 2: orthographic 3D (isometric) view of the same memory
    iso = np.array(Image.open("acra_paper/teaser_assets/isometric_00770_yaw0.png").convert("RGB"))
    mi = json.load(open("acra_paper/teaser_assets/isometric_00770_yaw0.json"))
    rows_ = np.where(iso.sum((1, 2)) > 0)[0]; cols_ = np.where(iso.sum((0, 2)) > 0)[0]
    c0, c1, r0, r1 = cols_.min() - 30, cols_.max() + 30, rows_.min() - 20, rows_.max() + 20
    axi = cv.fig.add_axes(cv.r(margin + 0.08, y0, iw, ih))
    axi.imshow(iso); axi.set_xlim(c0, c1); axi.set_ylim(r1, r0); axi.set_xticks([]); axi.set_yticks([]); axi.grid(False)
    for sp in axi.spines.values():
        sp.set_visible(False)
    d_ = P3 - np.array(mi["cam"])
    ic = mi["W"] / 2 + d_ @ np.array(mi["right"]) * mi["px_per_m"] + mi["dcol"]
    ir = mi["H"] / 2 - d_ @ np.array(mi["up"]) * mi["px_per_m"] + mi["drow"]
    Pf = P3.copy(); Pf[:, 1] = 0.107
    df = Pf - np.array(mi["cam"])
    fc_ = mi["W"] / 2 + df @ np.array(mi["right"]) * mi["px_per_m"] + mi["dcol"]
    fr_ = mi["H"] / 2 - df @ np.array(mi["up"]) * mi["px_per_m"] + mi["drow"]
    keep = (cls3 != "surface") & (cls3 != "other")
    for i in np.where(keep)[0]:
        axi.plot([fc_[i], ic[i]], [fr_[i], ir[i]], color="white", lw=0.35, alpha=0.5, zorder=2)
    for cls in ("surface", "other", "decor", "furniture", "opening", "fixture"):
        m = cls3 == cls
        axi.scatter(ic[m], ir[m], s=3 if cls in ("surface", "other") else 11, color=COL[cls], ec="white", lw=0.25,
                    alpha=0.5 if cls in ("surface", "other") else 0.95, zorder=3)
    cv.text(margin + 0.08, y0 + ih + 0.07, "The same memory in 3D (orthographic view)", va="bottom",
            fontweight="bold", color=NAVY, fontsize=7)
    # ---- legend
    lx = margin + 0.08 + iw + 0.25
    cv.text(lx, y0 + ih + 0.07, "What the dots are", va="bottom", fontweight="bold", color=NAVY, fontsize=7)
    for j, cls in enumerate(("fixture", "furniture", "opening", "decor", "surface")):
        yy = y0 + ih - 0.05 - j * 0.2
        cv.fig.add_artist(plt.Line2D([(lx + 0.05) / cv.W], [yy / cv.H], marker="o", color=COL[cls], markersize=5, ls="", markeredgecolor="white", markeredgewidth=0.4))
        cv.text(lx + 0.16, yy, f"{LABEL[cls]}  {int((cls3 == cls).sum())}", va="center", fontsize=6, color="0.2")
    cv.text(lx, y0 + 0.02, f"{len(ents3):,} objects stored after exploring the house\\n(scripted tour, scene 00770; real captions\\nand positions from the perception system)",
            va="bottom", fontsize=5.6, color="0.3", linespacing=1.2)
'''
out = SRC[:a] + BLOCK + SRC[b:]
out = out.replace("cv = Canvas(FULL, 3.97)", f"cv = Canvas(FULL, {3.97 + X})")
out = out.replace("margin, gap, by, bh = 0.14, 0.16, 2.29, 1.62", f"margin, gap, by, bh = 0.14, 0.16, {2.29 + X}, 1.62")
out = out.replace('Path("acra_paper/figures/fig_teaser.pdf")', 'Path("acra_paper/figures/fig_teaser_mockD.pdf")')
if "import matplotlib.pyplot as plt" not in out:
    out = out.replace("import matplotlib\n", "import matplotlib\n", 1)
p = Path("/tmp/claude-1004/-home-ksharma-mem-ws/228ff7f4-4e94-4d56-9832-54bf4b393403/scratchpad/mockD.py")
p.write_text(out)
r = subprocess.run([sys.executable, str(p)], capture_output=True, text=True)
print("D", "ok" if r.returncode == 0 else "FAILED", (r.stderr or "").strip().splitlines()[-3:] if r.returncode else "")
