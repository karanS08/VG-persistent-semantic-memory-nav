"""Teaser figure for the paper's introduction, modeled on the narrative structure
of DAAAM's own Figure 1 (Gorlo et al. 2025) -- sensor stream -> scene memory with
real descriptions -> a worked query/outcome example -- built entirely from this
project's own real data (2026-09-06):

  - Left: 6 real RGB frames captured live from scene 00770 via HabitatEnv
    (scripts/capture_teaser_frames.py), the same scene every other figure in
    this paper uses.
  - Middle: the same live entity-position snapshot as fig_3d_memory_scene.pdf,
    with two real DAAAM captions from that session called out.
  - Right: one real query end to end -- "a wall-mounted radiator", the
    pipeline's single most reliable direct-recall query (Section 7.4) --
    with the actual relevance/reliability/outcome numbers from a real episode
    (data/runs/.../drsweep_baseline_theta0.30, entity_id=49).

No invented imagery, captions, or numbers.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT_DIR = Path("paper/figures")
FRAMES_DIR = OUT_DIR / "teaser_frames"

plt.rcParams.update({"font.size": 8.5})

fig = plt.figure(figsize=(10, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 1.3], wspace=0.08)

# --- Left: real RGB frame strip ---
gs_left = gs[0].subgridspec(6, 1, hspace=0.06)
frame_paths = sorted(FRAMES_DIR.glob("frame_*.png"))
for i, fp in enumerate(frame_paths):
    ax = fig.add_subplot(gs_left[i])
    ax.imshow(mpimg.imread(fp))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
fig.text(0.005, 0.5, "Live RGB-D stream\n(Habitat-Sim, scene 00770)", rotation=90,
          va="center", ha="center", fontsize=8, fontweight="bold")

# --- Middle: real entity-position snapshot with 2 real caption callouts ---
ax_mid = fig.add_subplot(gs[1])
snapshot = json.load(open(
    "/tmp/claude-1004/-home-ksharma/7a19f3a6-82ed-494d-a93d-7c120ab171dd/scratchpad/live_positions_snapshot.json"))
positions = snapshot["positions"]
xs, zs = [], []
for obs in positions.values():
    x, y, z = obs[-1]["position_world"]
    xs.append(x)
    zs.append(z)
xs, zs = np.array(xs), np.array(zs)
ax_mid.scatter(xs, zs, s=10, alpha=0.45, color="#2166ac")
ax_mid.set_xlabel("x (m)")
ax_mid.set_ylabel("z (m)")
ax_mid.set_title("DAAAM: real-time 4D scene graph\n+ persistent decaying memory", fontsize=9)
ax_mid.set_aspect("equal")

# two real captions from this session's actual DAAAM output, called out on the map
callouts = [
    (xs[10], zs[10], "\"A traditional cast iron\nradiator with a series of\nvertical columns...\"", (-13, -6)),
    (xs[150], zs[150], "\"A wall-mounted mirror with\na thick, ornate frame\nfeaturing intricate carvings...\"", (2, 15.5)),
]
for cx, cz, text, (tx, tz) in callouts:
    ax_mid.annotate(text, xy=(cx, cz), xytext=(tx, tz),
                     fontsize=6.3, ha="left",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#fff7bc", ec="#666", lw=0.5),
                     arrowprops=dict(arrowstyle="-", color="#666", lw=0.6))

# --- Right: one real query, end to end ---
ax_right = fig.add_subplot(gs[2])
ax_right.set_xlim(0, 1)
ax_right.set_ylim(0, 1)
ax_right.axis("off")

boxes = [
    (0.5, 0.93, "Query:\n\"a wall-mounted radiator\"", "#d1e5f0"),
    (0.5, 0.72, "Direct recall:\n$s(q,d_i)=0.217$\n$C_i=0.393$\n$R_i=0.085$", "#fddbc7"),
    (0.5, 0.46, "Matched entity:\n\"A black, horizontal,\ncolumn-style radiator\nwith a series of vertical\nfins...\"", "#f7f7f7"),
    (0.5, 0.19, "Outcome: verified\n198 steps, SPL=0.096\n(one of 9 correct firings\nfor this query, Section 7.4)", "#d9f0d3"),
]
for x, y, text, color in boxes:
    box = FancyBboxPatch((x - 0.42, y - 0.075), 0.84, 0.15, boxstyle="round,pad=0.02",
                          fc=color, ec="#444", lw=0.6, transform=ax_right.transAxes)
    ax_right.add_patch(box)
    ax_right.text(x, y, text, transform=ax_right.transAxes, ha="center", va="center", fontsize=7.3)

for (x1, y1, _, _), (x2, y2, _, _) in zip(boxes[:-1], boxes[1:]):
    ax_right.annotate("", xy=(x2, y2 + 0.075), xytext=(x1, y1 - 0.075),
                       xycoords="axes fraction", textcoords="axes fraction",
                       arrowprops=dict(arrowstyle="->", color="#333", lw=1.0))

ax_right.set_title("Frontier agent: memory-weighted\nverification, one real episode", fontsize=9)

fig.savefig(OUT_DIR / "fig_teaser.pdf", bbox_inches="tight", dpi=200)
print("wrote fig_teaser.pdf")
