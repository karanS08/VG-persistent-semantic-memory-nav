"""Run the UNMODIFIED orchestrator.run_phase4_batch against AI2-THOR/ProcTHOR instead of Habitat.

Mechanism: import-time shims. `habitat_sim` -> stub exposing ShortestPath; `sim_bridge.habitat_env`
-> module whose HabitatEnv builds a ThorEnv (house chosen by THOR_HOUSE index into procthor-10k train).
On every env.close() the world-frame (Habitat convention) ground-truth object list is written to
THOR_GT_DIR/gt_<tag>.json so memory positions can be scored against ground truth afterwards.

Usage (rpv_thor env, repo root):
  THOR_HOUSE=1 THOR_GT_DIR=data/runs/<run> python scripts/thor/run_thor_batch.py <run_phase4_batch args...>
"""
import json, os, runpy, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("sim_bridge", "daaam_adapter", "frontier_agent", "memory_layer", "orchestrator"):
    sys.path.insert(0, str(ROOT / "packages" / p))
from sim_bridge import thor_env  # noqa: E402

hs = types.ModuleType("habitat_sim"); hs.ShortestPath = thor_env.ShortestPath
sys.modules["habitat_sim"] = hs

_HOUSE = None
def _house():
    global _HOUSE
    if _HOUSE is None:
        import prior, time
        for attempt in range(5):  # prior does a git fetch; a transient network failure raises AssertionError
            try:
                _HOUSE = prior.load_dataset("procthor-10k")["train"][int(os.environ.get("THOR_HOUSE", "1"))]
                break
            except AssertionError:
                print(f"[thor] prior.load_dataset failed (attempt {attempt + 1}/5), retrying in 20 s", flush=True)
                time.sleep(20)
        else:
            raise RuntimeError("prior.load_dataset failed 5 times")
    return _HOUSE

class HabitatEnv(thor_env.ThorEnv):  # same name/signature the runner expects
    _n = 0      # closes so far (names GT dumps)
    _made = 0   # envs constructed so far == query index in per-query-env modes (e.g. --persist-memory-only)
    def __init__(self, config, episode_id="episode_0"):
        super().__init__(config, episode_id=episode_id, house=_house())
        # THOR_START="x,z,yaw" (THOR coords): override the house's default spawn pose used by reset()
        st = os.environ.get("THOR_START")
        if st:
            x, z, yaw = (float(v) for v in st.split(","))
            self._spawn = ({"x": x, "y": self._spawn[0]["y"], "z": z}, yaw)
        k, HabitatEnv._made = HabitatEnv._made, HabitatEnv._made + 1
        # THOR_RELOCATE="<first_episode>:<objectId>:<x_thor>:<z_thor>" -- from episode <first_episode> on, the object is
        # teleported to the reachable cell nearest (x,z) BEFORE the agent acts (moved while the agent was away).
        spec = os.environ.get("THOR_RELOCATE")
        if spec:
            for one in spec.split(";"):  # several objects: "ep:oid:x:z;ep:oid:x:z"
                first, oid, x, z = one.split(":"); x, z = float(x), float(z)
                if k >= int(first) and os.environ.get("THOR_RELOCATE_CLEAR") == "1":
                    # clear-floor placement + navgrid rebuild (thor_env.relocate_clear); never run with a bad placement
                    pos = self.relocate_clear(oid, x, z)
                    print(f"[thor] episode {k}: relocated {oid} -> clear spot near ({x},{z}): {pos}", flush=True)
                    if pos is None:
                        sys.exit(f"no clear placement for {oid} near ({x},{z})")
                elif k >= int(first):
                    ok = self.relocate(oid, x, z)
                    print(f"[thor] episode {k}: relocated {oid} -> nearest reachable to ({x},{z}) ok={ok}", flush=True)
    def close(self):
        d = os.environ.get("THOR_GT_DIR")
        if d:
            HabitatEnv._n += 1
            objs = [{"id": o["objectId"], "type": o["objectType"], "pickupable": o["pickupable"],
                     "pos_h": thor_env.t2h_point(o["axisAlignedBoundingBox"]["center"]),
                     "size": o["axisAlignedBoundingBox"]["size"]}
                    for o in self._c.last_event.metadata["objects"]]
            Path(d).mkdir(parents=True, exist_ok=True)
            json.dump(objs, open(Path(d) / f"gt_{HabitatEnv._n}.json", "w"))
        super().close()

m = types.ModuleType("sim_bridge.habitat_env"); m.HabitatEnv = HabitatEnv; m.Frame = thor_env.Frame
sys.modules["sim_bridge.habitat_env"] = m
# --- DAAAM camera-convention switch (THOR_CAM_FIX=1 default, 0 = push the Habitat pose unchanged) ---
# DAAAM unprojects with the OpenCV camera convention (x right, y down, +z = depth/forward) and multiplies by the
# pushed pose unchanged (utils/geometry.pose_to_matrix). The runner pushes the Habitat agent pose (OpenGL
# convention: forward = -z, y up), so without a correction unprojected points land mirrored behind the agent.
# Fix: R_world_cam = R_habitat @ diag(1,-1,-1); translation = base + camera height.
import numpy as np
from scipy.spatial.transform import Rotation as _R
import daaam_adapter.bridge_client as _bc
_orig_push = _bc.DaaamBridgeClient.push_frame
def _push(self, frame_id, timestamp, rgb, depth=None, transform=None, camera_intrinsics=None):
    if transform is not None and os.environ.get("THOR_CAM_FIX", "1") == "1":
        x, y, z, qx, qy, qz, qw = transform
        R = _R.from_quat([qx, qy, qz, qw]).as_matrix() @ np.diag([1.0, -1.0, -1.0])
        q = _R.from_matrix(R).as_quat()
        transform = [x, y + 1.5, z, float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    return _orig_push(self, frame_id, timestamp, rgb, depth, transform, camera_intrinsics)
_bc.DaaamBridgeClient.push_frame = _push
runpy.run_module("orchestrator.run_phase4_batch", run_name="__main__")
