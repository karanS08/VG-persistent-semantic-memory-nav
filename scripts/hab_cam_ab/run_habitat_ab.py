"""Run the UNMODIFIED orchestrator.run_phase4_batch on Habitat with an optional DAAAM camera-convention fix
(HAB_CAM_FIX=1) applied to the pushed transform, to test whether the pipeline's remembered positions are
mirrored behind the agent (see scripts/thor/run_thor_batch.py for the derivation). HAB_CAM_FIX=0 = pipeline as published.
Run in the rpv_mem env from the repo root."""
import os, runpy, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in ("sim_bridge", "daaam_adapter", "frontier_agent", "memory_layer", "orchestrator"):
    sys.path.insert(0, str(ROOT / "packages" / p))
import numpy as np
from scipy.spatial.transform import Rotation as _R
import daaam_adapter.bridge_client as _bc
_orig = _bc.DaaamBridgeClient.push_frame
def _push(self, frame_id, timestamp, rgb, depth=None, transform=None, camera_intrinsics=None):
    if transform is not None and os.environ.get("HAB_CAM_FIX", "0") == "1":
        x, y, z, qx, qy, qz, qw = transform
        q = _R.from_matrix(_R.from_quat([qx, qy, qz, qw]).as_matrix() @ np.diag([1.0, -1.0, -1.0])).as_quat()
        transform = [x, y + 1.5, z, float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    return _orig(self, frame_id, timestamp, rgb, depth, transform, camera_intrinsics)
_bc.DaaamBridgeClient.push_frame = _push
runpy.run_module("orchestrator.run_phase4_batch", run_name="__main__")
