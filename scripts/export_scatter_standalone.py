"""Standalone export of the 3D-memory-scene scatter panel (with its real caption
callouts) as its own high-res PNG, for manual composition in Canva. Same real
data as generate_teaser_figure.py's middle panel."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/tmp/claude-1004/-home-ksharma/7a19f3a6-82ed-494d-a93d-7c120ab171dd/scratchpad/canva_export/scatter_plot.png"

plt.rcParams.update({"font.size": 11})

snapshot = json.load(open(
    "/tmp/claude-1004/-home-ksharma/7a19f3a6-82ed-494d-a93d-7c120ab171dd/scratchpad/live_positions_snapshot.json"))
positions = snapshot["positions"]
xs, zs = [], []
for obs in positions.values():
    x, y, z = obs[-1]["position_world"]
    xs.append(x)
    zs.append(z)
xs, zs = np.array(xs), np.array(zs)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(xs, zs, s=22, alpha=0.5, color="#2166ac")
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title("DAAAM: real-time 4D scene graph\n+ persistent decaying memory")
ax.set_aspect("equal")

callouts = [
    (xs[10], zs[10], "\"A traditional cast iron\nradiator with a series of\nvertical columns...\"", (-13, -6)),
    (xs[150], zs[150], "\"A wall-mounted mirror with\na thick, ornate frame\nfeaturing intricate carvings...\"", (2, 15.5)),
]
for cx, cz, text, (tx, tz) in callouts:
    ax.annotate(text, xy=(cx, cz), xytext=(tx, tz),
                fontsize=9, ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff7bc", ec="#666", lw=0.5),
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.7))

fig.savefig(OUT, dpi=250, bbox_inches="tight", facecolor="white")
print("wrote", OUT)

# also a plain version with no callouts, in case that's more useful in Canva
fig2, ax2 = plt.subplots(figsize=(7, 6))
ax2.scatter(xs, zs, s=22, alpha=0.5, color="#2166ac")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("z (m)")
ax2.set_title("Remembered entities in 3D scene space")
ax2.set_aspect("equal")
fig2.savefig(OUT.replace("scatter_plot.png", "scatter_plot_plain.png"), dpi=250,
             bbox_inches="tight", facecolor="white")
print("wrote", OUT.replace("scatter_plot.png", "scatter_plot_plain.png"))
