"""3D scene-space figure: real remembered-entity positions, colored by real
elapsed time since last observation. Data: a live snapshot pulled from the
bridge's /positions endpoint (GET http://localhost:8765/positions) during the
2026-09-05 lexical_policy_probe batch run, saved to
scratchpad/live_positions_snapshot.json -- 265 real entities from one episode's
actual DAAAM grounding output on scene 00770.

This is the one 3D figure in the paper, and deliberately so: the other results
(decay curves, ablation bars, threshold sweeps) are genuinely 2D and forcing
them into 3D would only obscure them. This one is different -- the pipeline's
whole contribution is a memory anchored in real 3D scene positions that decays
with elapsed time, so a 3D scatter colored by elapsed time is what that concept
actually looks like, not a decorative add-on.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SNAPSHOT = "/tmp/claude-1004/-home-ksharma/7a19f3a6-82ed-494d-a93d-7c120ab171dd/scratchpad/live_positions_snapshot.json"
OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

d = json.load(open(SNAPSHOT))
positions = d["positions"]

xs, ys, zs, delta_ts = [], [], [], []
all_last_ts = [obs[-1]["timestamp"] for obs in positions.values()]
snapshot_now = max(all_last_ts)  # "now" relative to this snapshot, not wall-clock

for sid, obs_list in positions.items():
    last = obs_list[-1]
    x, y, z = last["position_world"]
    xs.append(x)
    ys.append(y)
    zs.append(z)
    delta_ts.append(snapshot_now - last["timestamp"])

xs, ys, zs, delta_ts = map(np.array, (xs, ys, zs, delta_ts))

fig = plt.figure(figsize=(6.4, 5.2))
ax = fig.add_subplot(111, projection="3d")
# Habitat's y-axis is up; plot as (x, z, y) so the "up" direction reads vertically.
sc = ax.scatter(xs, zs, ys, c=delta_ts, cmap="viridis_r", s=22, alpha=0.85,
                 edgecolors="none", vmin=0, vmax=delta_ts.max())
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Time since last observation (s)")
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_zlabel("y / height (m)")
ax.set_title(f"Remembered entities in 3D scene space\n(n={len(xs)}, one live episode, scene 00770)")
ax.view_init(elev=22, azim=-60)
fig.savefig(OUT_DIR / "fig_3d_memory_scene.pdf")
plt.close(fig)
print(f"wrote fig_3d_memory_scene.pdf ({len(xs)} entities, delta_t range "
      f"{delta_ts.min():.1f}-{delta_ts.max():.1f}s)")
