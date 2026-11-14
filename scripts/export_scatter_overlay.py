"""Transparent-background scatter overlay for Canva compositing on top of a real
RGB frame (2026-09-06, per user request). No axes/ticks/title -- a labeled
coordinate plot wouldn't make sense floating on a photo. Same real entity-
position snapshot used throughout this session's teaser work.

"Positive" (green) = entities this session actually verified as a real query
match, traceable to a real episode: the radiator (index 10) and mirror
(index 150) called out earlier in fig_teaser.pdf, both real verified matches
from this project's runs. Everything else (gray) is other real remembered
entities from the same snapshot, shown as context, not claimed as positive.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/tmp/claude-1004/-home-ksharma/7a19f3a6-82ed-494d-a93d-7c120ab171dd/scratchpad/canva_export/scatter_overlay_transparent.png"

snapshot = json.load(open(
    "/tmp/claude-1004/-home-ksharma/7a19f3a6-82ed-494d-a93d-7c120ab171dd/scratchpad/live_positions_snapshot.json"))
positions = snapshot["positions"]
xs, zs = [], []
for obs in positions.values():
    x, y, z = obs[-1]["position_world"]
    xs.append(x)
    zs.append(z)
xs, zs = np.array(xs), np.array(zs)

positive_idx = [10, 150]  # radiator, mirror -- real verified matches
mask = np.ones(len(xs), dtype=bool)
mask[positive_idx] = False

fig, ax = plt.subplots(figsize=(8, 6.5))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# context entities: small, muted, low alpha
ax.scatter(xs[mask], zs[mask], s=40, color="#888888", alpha=0.35, edgecolors="none")

# positive (verified) matches: large, bright, ringed
ax.scatter(xs[positive_idx], zs[positive_idx], s=260, color="#2ecc71", alpha=0.95,
           edgecolors="white", linewidths=2.5, zorder=5)
ax.scatter(xs[positive_idx], zs[positive_idx], s=420, facecolors="none",
           edgecolors="#2ecc71", linewidths=1.5, alpha=0.5, zorder=4)

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_aspect("equal")

fig.savefig(OUT, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0)
print("wrote", OUT)
