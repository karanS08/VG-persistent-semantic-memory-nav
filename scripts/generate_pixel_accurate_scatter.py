"""Scatter-style overlay using REAL pixel-space detection centers from the live
grounding capture (2026-09-06), not the earlier mismatched 3D-world-coordinate
data. Each dot is a real FastSAM/BoT-SORT track's bbox center in that exact
frame's own pixel coordinates -- fully registered by construction, no
coordinate-system mismatch.

The 3 largest-area tracks (window frame, whole building facade, wall panel --
areas 20584/58166/52614 px^2, an order of magnitude bigger than every other
track) are background/compositional regions, not discrete objects, and are
excluded from the scatter so the dots represent genuine object-scale detections
(individual windows, the car, door/window hardware) rather than being dominated
by three giant boxes. All excluded/included tracks are real data either way --
this is a display choice, not a data change (kept explicit in AREA_CUTOFF below).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--suffix", default="", help="e.g. '_matched' to use raw_frame_matched.png/tracks_matched.json")
parser.add_argument("--area-cutoff", type=float, default=10000,
                     help="px^2 -- excludes background/compositional regions bigger than this")
parser.add_argument("--size-divisor", type=float, default=6,
                     help="marker area = bbox_area / size_divisor, before clipping")
parser.add_argument("--size-clip", type=float, nargs=2, default=[40, 260],
                     help="min/max marker size after scaling")
args = parser.parse_args()

RAW_FRAME = f"paper/figures/live_grounding/raw_frame{args.suffix}.png"
TRACKS = f"paper/figures/live_grounding/tracks{args.suffix}.json"
OUT = f"paper/figures/live_grounding/live_grounding_scatter{args.suffix}.png"

AREA_CUTOFF = args.area_cutoff

tracks = json.load(open(TRACKS))
img = mpimg.imread(RAW_FRAME)
h, w = img.shape[0], img.shape[1]

centers, areas, ids = [], [], []
for t in tracks:
    x1, y1, x2, y2 = t["bbox"]
    area = (x2 - x1) * (y2 - y1)
    if area > AREA_CUTOFF:
        continue
    centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
    areas.append(area)
    ids.append(t["id"])

centers = np.array(centers)
areas = np.array(areas)
# marker size scaled to real bbox area, clipped to a readable range
sizes = np.clip(areas / args.size_divisor, args.size_clip[0], args.size_clip[1])

fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
ax.imshow(img)
ax.scatter(centers[:, 0], centers[:, 1], s=sizes, color="#2166ac", alpha=0.55,
           edgecolors="white", linewidths=1.2, zorder=5)
ax.set_xlim(0, w)
ax.set_ylim(h, 0)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(OUT, dpi=100)
plt.close(fig)
print(f"wrote {OUT} -- {len(centers)} real detections plotted "
      f"({len(tracks) - len(centers)} background regions excluded, area>{AREA_CUTOFF})")
