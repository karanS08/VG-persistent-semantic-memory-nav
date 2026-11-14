"""Temporal-decay figure (2026-10-03): four panels of the SAME memory on the HM3D 00770 map at increasing time since last seen.
Memory: the scripted house-tour memory (data/runs/phase4_batch_persist_memory_only_house_tour/memory_snapshot_ep0.json, 1,193 entries
with real DAAAM captions and estimated 3D positions). Each dot is one entry; its opacity is its reliability relative to its initial value,
C_i(t)/C_i(0) = exp(-lambda_i t) with lambda_i = lambda_base * mobility(category_i) (Eq. 1; lambda_base, alpha from configs/memory.yaml, mobility from
configs/mobility_priors.json -- the paper's actual weights, not adjusted for display). Colour = mobility prior. A dot is hidden once its reliability
is below 5% of its initial value; entries are never deleted. The 268 entries whose mobility prior is exactly 0 (static; they never decay) are
not drawn. Illustration of the equation, not an experiment.
Out: acra_paper/figures/fig_decay_map.pdf (+ png preview)"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

sys.path[:0] = ["packages/memory_layer", "scripts"]
from memory_layer.mobility import MobilityPriors  # noqa: E402
from make_teaser import to_px  # noqa: E402
from paper_figstyle import FULL, finalize  # noqa: E402

TOUR = "data/runs/phase4_batch_persist_memory_only_house_tour/memory_snapshot_ep0.json"
HIDE = 0.05                                             # hidden below 5% of initial reliability (display only)
TIMES = [(0.0, "Just seen"), (6 * 3600.0, "6 hours"), (24 * 3600.0, "24 hours"), (40 * 3600.0, "40 hours")]

lam_base = yaml.safe_load(open("configs/memory.yaml"))["reliability"]["lambda_base"]
mob = MobilityPriors.load()
E_all = [e for e in json.load(open(TOUR))["entities"] if e["position_world"]]
M_all = np.array([mob.mobility_for(e["category"]) for e in E_all])
N_ZERO = int((M_all == 0).sum())                          # prior exactly 0 -> never decays; left out of the figure (22.5% of entries)
keep = M_all > 0
E = [e for e, k in zip(E_all, keep) if k]
P = np.array([e["position_world"] for e in E])
M = M_all[keep]
N_NONZERO = len(E)
lam = lam_base * M

hr = json.load(open("acra_paper/teaser_assets/highres/map_topdown_house_highres.json"))
mp = json.load(open("acra_paper/figures/teaser/topdown_00770.json"))
base = Image.open("acra_paper/teaser_assets/highres/map_topdown_house_white.png").convert("RGB")
bk = 1811.0 / base.width                                                                  # high-res crop pixels -> the 1811 x 925 coordinate system
base = base.resize((base.width // 2, base.height // 2), Image.LANCZOS)                    # 3168 px wide: ~1,900 ppi per 1.65 in panel
u, v = to_px(P, mp)
X = (u * hr["scale_vs_1600_render"] - hr["crop_origin_px"][0]) * bk
Y = (v * hr["scale_vs_1600_render"] - hr["crop_origin_px"][1]) * bk
W, H = 1811, 925
# display filter (disclosed in the caption): drop entries whose estimated position is more than 2 m outside the floor plan
from scipy.ndimage import distance_transform_edt
_mask = np.array(Image.open("acra_paper/teaser_assets/highres/map_topdown_house_black.png").convert("L").resize((W, H))) > 25
_ppm = mp["px_per_m"] * hr["scale_vs_1600_render"] * bk
_d = distance_transform_edt(~_mask) / _ppm
_ix, _iy = np.clip(X.astype(int), 0, W - 1), np.clip(Y.astype(int), 0, H - 1)
near = (X >= 0) & (X < W) & (Y >= 0) & (Y < H) & (_d[_iy, _ix] <= 2.0)
N_FAR = int((~near).sum())
X, Y, M, P, lam = X[near], Y[near], M[near], P[near], lam[near]
E = [e for e, k in zip(E, near) if k]

cmap = LinearSegmentedColormap.from_list("mob", ["#2c7bb6", "#7fa8c9", "#f2b36b", "#d7191c"])
norm = Normalize(0, 0.9)
FH = 1.55
fig = plt.figure(figsize=(FULL, FH))
order = np.argsort(M)                                                                       # draw mobile dots last (on top)
axes = [fig.add_axes([0.005 + k * 0.2488, 0.22, 0.2415, 0.66]) for k in range(4)]
for ax, (t, title) in zip(axes, TIMES):
    rel = np.exp(-lam * t)                                                                  # C_i(t) / C_i(0)
    vis = rel >= HIDE
    ax.imshow(np.asarray(Image.blend(base, Image.new("RGB", base.size, "white"), 0.62)), extent=(0, W, H, 0), zorder=0)   # map faded so dots read
    idx = [i for i in order if vis[i]]
    cols = cmap(norm(M[idx])); cols[:, 3] = 0.15 + 0.85 * rel[idx]
    ax.scatter(X[idx], Y[idx], s=3.0 + 9.0 * M[idx], color=cols, linewidths=0, zorder=3)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=8, pad=2, fontweight="bold")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cax = fig.add_axes([0.37, 0.075, 0.26, 0.04])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.set_ticks([0, 0.45, 0.9]); cb.set_ticklabels(["static", "mobility prior", "mobile"])
cb.outline.set_visible(False); cax.tick_params(length=0, labelsize=7, pad=1)
finalize(fig, Path("acra_paper/figures/fig_decay_map.pdf"), FULL, FH, tight=False)
fig.savefig("/tmp/claude-1004/-home-ksharma-mem-ws/228ff7f4-4e94-4d56-9832-54bf4b393403/scratchpad/decay_map_preview.png", dpi=200)
print("ENTRIES: total", len(E_all), "| prior 0 (not drawn):", N_ZERO, "| prior>0:", N_NONZERO, "| >2 m outside plan (not drawn):", N_FAR, "| DRAWN:", len(E), "|" "| visible per panel:", [int((np.exp(-lam * t) >= HIDE).sum()) for t, _ in TIMES], "| lambda_base", lam_base, "| hide below", HIDE)
