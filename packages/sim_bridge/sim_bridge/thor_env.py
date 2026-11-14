"""AI2-THOR / ProcTHOR drop-in for HabitatEnv (2026-10-28, camera-ready reviewer-2 test).

Same contract as habitat_env.HabitatEnv -- reset / step / current_frame / close,
`.pathfinder`, `._config` -- so frontier_agent, memory_layer and the DAAAM bridge run
unchanged. Frames are emitted in HABITAT's convention:

  * right-handed, y up, camera looks along -z, +x to the agent's right;
  * pose.position = agent BASE position [x, y_floor, z];
  * pose.rotation = [w, x, y, z] quaternion about +y.

THOR is left-handed (forward = +z at yaw 0, yaw grows clockwise). Mapping used:
    x_h = x_t,  z_h = -z_t,  yaw_h = -yaw_t  (rotation about +y by -yaw_t).
Derivation: THOR forward at yaw t is (sin t, cos t) in (x, z); Habitat forward for a
rotation phi about +y is (-sin phi, -cos phi); equating gives phi = -t.

THOR has no navmesh, so ThorPathfinder rebuilds the three pathfinder calls the agent
uses (snap_point, find_path, get_random_navigable_point_near) from THOR's reachable
grid. See validate_thor_env.py for the projection test of the pose/FOV conventions.
"""
from __future__ import annotations

import heapq
import math
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

GRID = 0.25


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    pose: Dict[str, Any]
    timestamp: float
    episode_id: str
    step_id: int

    def as_dict(self) -> Dict[str, Any]:
        return {"rgb": self.rgb, "depth": self.depth, "pose": self.pose,
                "timestamp": self.timestamp, "episode_id": self.episode_id, "step_id": self.step_id}


def t2h_point(p: Dict[str, float], floor_y: float = 0.0) -> List[float]:
    """THOR world point -> Habitat world point (floor-level y)."""
    return [float(p["x"]), float(floor_y), -float(p["z"])]


def h2t_xz(x: float, z: float) -> Tuple[float, float]:
    return x, -z


def yaw_to_wxyz(yaw_thor_deg: float) -> List[float]:
    phi = math.radians(-yaw_thor_deg)
    return [math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0]


class ThorPathfinder:
    """Habitat-PathFinder-shaped view of THOR's reachable grid (8-connected A*)."""

    def __init__(self, reachable: List[Dict[str, float]], seed: int = 0):
        self._rng = random.Random(seed)
        # grid index <-> THOR (x, z); store Habitat-frame points
        self._cells: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for p in reachable:
            ix, iz = round(p["x"] / GRID), round(p["z"] / GRID)
            self._cells[(ix, iz)] = (p["x"], p["z"])
        self._keys = list(self._cells)
        self._arr = np.array([self._cells[k] for k in self._keys], dtype=np.float64)  # THOR x,z
        self.num_islands = self._count_islands()

    def block_point(self, point) -> bool:
        """Drop the grid cell at a Habitat-frame point the agent collided at. THOR's reachable grid can admit cells the
        off-grid agent (snapToGrid=False) cannot enter; without this the planner re-routes into them forever."""
        k = self._nearest_key(float(point[0]), float(point[2]))
        x_t, z_t = self._cells[k]
        if math.hypot(x_t - point[0], -z_t - point[2]) > GRID * 0.75 or len(self._cells) < 2:
            return False
        del self._cells[k]
        self._keys = list(self._cells)
        self._arr = np.array([self._cells[k] for k in self._keys], dtype=np.float64)
        return True

    # -- helpers ----------------------------------------------------------------
    def _h_point(self, x_t: float, z_t: float) -> np.ndarray:
        return np.array([x_t, 0.0, -z_t], dtype=np.float32)

    def _nearest_key(self, x_h: float, z_h: float) -> Optional[Tuple[int, int]]:
        x_t, z_t = h2t_xz(x_h, z_h)
        d = (self._arr[:, 0] - x_t) ** 2 + (self._arr[:, 1] - z_t) ** 2
        i = int(np.argmin(d))
        return self._keys[i]

    def _neighbors(self, k):
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue
                n = (k[0] + dx, k[1] + dz)
                if n in self._cells:
                    # no corner cutting through a blocked orthogonal neighbour
                    if dx != 0 and dz != 0 and ((k[0] + dx, k[1]) not in self._cells or (k[0], k[1] + dz) not in self._cells):
                        continue
                    yield n, math.hypot(dx, dz) * GRID

    def _count_islands(self) -> int:
        seen, n = set(), 0
        for k in self._keys:
            if k in seen:
                continue
            n += 1
            stack = [k]
            while stack:
                c = stack.pop()
                if c in seen:
                    continue
                seen.add(c)
                stack.extend(m for m, _ in self._neighbors(c) if m not in seen)
        return n

    # -- Habitat PathFinder surface used by the agent ---------------------------------
    def snap_point(self, point) -> np.ndarray:
        k = self._nearest_key(float(point[0]), float(point[2]))
        return self._h_point(*self._cells[k])

    def is_navigable(self, point, max_y_delta: float = 0.5) -> bool:
        k = self._nearest_key(float(point[0]), float(point[2]))
        x_t, z_t = self._cells[k]
        return math.hypot(x_t - point[0], -z_t - point[2]) <= GRID * 0.75

    def get_random_navigable_point_near(self, circle_center, radius: float, **_):
        cx_t, cz_t = h2t_xz(float(circle_center[0]), float(circle_center[2]))
        d = np.hypot(self._arr[:, 0] - cx_t, self._arr[:, 1] - cz_t)
        idx = np.nonzero(d <= radius)[0]
        if len(idx) == 0:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        i = int(idx[self._rng.randrange(len(idx))])
        return self._h_point(*self._arr[i])

    def find_path(self, sp) -> bool:
        s = self._nearest_key(float(sp.requested_start[0]), float(sp.requested_start[2]))
        g = self._nearest_key(float(sp.requested_end[0]), float(sp.requested_end[2]))
        if s is None or g is None:
            return False
        openq, best, prev = [(0.0, s)], {s: 0.0}, {}
        while openq:
            f, c = heapq.heappop(openq)
            if c == g:
                break
            for n, w in self._neighbors(c):
                ng = best[c] + w
                if ng < best.get(n, 1e18):
                    best[n], prev[n] = ng, c
                    h = math.hypot(n[0] - g[0], n[1] - g[1]) * GRID
                    heapq.heappush(openq, (ng + h, n))
        if g not in best:
            sp.points, sp.geodesic_distance = [], float("inf")
            return False
        path, c = [g], g
        while c != s:
            c = prev[c]
            path.append(c)
        path.reverse()
        pts = [self._h_point(*self._cells[k]) for k in path]
        # keep the exact requested start/goal as end points, like Habitat's find_path
        pts[0] = np.array(sp.requested_start, dtype=np.float32)
        pts[-1] = np.array(sp.requested_end, dtype=np.float32)
        sp.points, sp.geodesic_distance = pts, best[g]
        return True


class ShortestPath:
    """Stand-in for habitat_sim.ShortestPath (plain attribute bag)."""

    def __init__(self):
        self.requested_start = None
        self.requested_end = None
        self.points: List[np.ndarray] = []
        self.geodesic_distance = float("inf")


_ACTIONS = {"move_forward": ("MoveAhead", {"moveMagnitude": GRID}),
            "turn_left": ("RotateLeft", {"degrees": 10.0}),
            "turn_right": ("RotateRight", {"degrees": 10.0})}


class ThorEnv:
    """Scripted-agent THOR wrapper with HabitatEnv's interface."""

    def __init__(self, config: Dict[str, Any], episode_id: str = "episode_0", house: Optional[dict] = None,
                 scene: str = "FloorPlan1"):
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering

        self._config = config
        self.episode_id = episode_id
        self._step_id = 0
        h, w = config["sensor"]["rgb_resolution"]
        # THOR's fieldOfView is the VERTICAL FOV (validated: scripts/thor/validate_thor_env.py, 9 cm median
        # back-projection error with vertical-FOV intrinsics vs 40-50 cm with horizontal). The runner's
        # camera_intrinsics() assumes a horizontal hfov, so request the vertical FOV that yields it.
        vfov = math.degrees(2 * math.atan(math.tan(math.radians(config["sensor"]["hfov"]) / 2) * h / w))
        self._c = Controller(
            platform=CloudRendering, scene=house if house is not None else scene, width=w, height=h,
            fieldOfView=vfov, renderDepthImage=True, gridSize=GRID,
            rotateStepDegrees=config["action_space"]["turn_left"]["amount"], snapToGrid=False,
            renderInstanceSegmentation=config.get("render_instance_segmentation", False))
        self._spawn = None
        self._pathfinder: Optional[ThorPathfinder] = None
        self._build_navgrid()

    def _build_navgrid(self):
        ev = self._c.step("GetReachablePositions")
        self._pathfinder = ThorPathfinder(ev.metadata["actionReturn"])
        a = self._c.last_event.metadata["agent"]
        self._spawn = (dict(a["position"]), a["rotation"]["y"])

    def _frame(self) -> Frame:
        e = self._c.last_event
        a = e.metadata["agent"]
        pose = {"position": t2h_point(a["position"]), "rotation": yaw_to_wxyz(a["rotation"]["y"])}
        return Frame(rgb=e.frame.copy(), depth=e.depth_frame.copy(), pose=pose, timestamp=time.time(),
                     episode_id=self.episode_id, step_id=self._step_id)

    def reset(self) -> Frame:
        self._step_id = 0
        pos, yaw = self._spawn
        # horizon=0: level camera, matching Habitat's forward-looking sensor (THOR defaults to 30 deg down)
        self._c.step("Teleport", position=pos, rotation={"x": 0, "y": yaw, "z": 0}, horizon=0, standing=True)
        return self._frame()

    def current_frame(self) -> Frame:
        return self._frame()

    def step(self, action: str) -> Frame:
        name, kw = _ACTIONS[action]
        self._c.step(name, **kw)
        self._step_id += 1
        return self._frame()

    def close(self) -> None:
        self._c.stop()

    @property
    def pathfinder(self) -> ThorPathfinder:
        return self._pathfinder

    def relocate(self, object_id: str, x_t: float, z_t: float) -> bool:
        """Teleport a movable object to the reachable grid cell nearest (x_t, z_t) (THOR coords), floor level."""
        pf = self._pathfinder
        d = (pf._arr[:, 0] - x_t) ** 2 + (pf._arr[:, 1] - z_t) ** 2
        for i in np.argsort(d)[:20]:  # a few nearby cells in case the object does not fit at the first
            ev = self._c.step("TeleportObject", objectId=object_id, position={"x": float(pf._arr[i][0]), "y": 0.0, "z": float(pf._arr[i][1])},
                              rotation={"x": 0, "y": 0, "z": 0}, forceAction=True, forceKinematic=True)
            if ev.metadata["lastActionSuccess"]:
                return True
        return False

    def relocate_clear(self, object_id: str, x_t: float, z_t: float, clearance: float = 0.3,
                       max_drift: float = 2.5) -> Optional[Tuple[float, float]]:
        """Like relocate(), but only accepts a placement in clear floor space: the object's box (grown by `clearance`)
        overlaps no other object except the floor and items resting on it, the house stays one connected navigable
        region, and the agent can stand within 1 m of the object. relocate() teleports with forceAction, so it could
        bury the object in a bed or block a doorway. Rebuilds the reachable grid IN PLACE (the agent holds a reference
        to the pathfinder) so the planner sees the moved object. Returns the new THOR (x, z), or None if no cell fits."""
        pf = self._pathfinder
        islands0 = pf.num_islands
        d = np.hypot(pf._arr[:, 0] - x_t, pf._arr[:, 1] - z_t)
        for i in np.argsort(d):
            if d[i] > max_drift:
                break
            ev = self._c.step("TeleportObject", objectId=object_id, position={"x": float(pf._arr[i][0]), "y": 0.0, "z": float(pf._arr[i][1])},
                              rotation={"x": 0, "y": 0, "z": 0}, forceAction=True, forceKinematic=True)
            if not ev.metadata["lastActionSuccess"]:
                continue
            objs = ev.metadata["objects"]
            me = next(o for o in objs if o["objectId"] == object_id)
            c, s = me["axisAlignedBoundingBox"]["center"], me["axisAlignedBoundingBox"]["size"]
            # walls report a whole-house AABB in ProcTHOR, so they are checked via the footprint instead: every corner
            # of the object's box (grown by clearance/2) must lie within one grid cell of a reachable cell (open floor)
            corners = [(c["x"] + sx * (s["x"] / 2 + clearance / 2), c["z"] + sz * (s["z"] / 2 + clearance / 2))
                       for sx in (-1, 1) for sz in (-1, 1)]
            if any(np.hypot(pf._arr[:, 0] - cx, pf._arr[:, 1] - cz).min() > GRID for cx, cz in corners):
                continue
            clash = False
            for o in objs:
                if (o["objectId"] == object_id or o["objectType"] in ("Floor", "Wall")
                        or object_id in (o.get("parentReceptacles") or [])):
                    continue
                oc, os_ = o["axisAlignedBoundingBox"]["center"], o["axisAlignedBoundingBox"]["size"]
                if (abs(oc["x"] - c["x"]) < (os_["x"] + s["x"]) / 2 + clearance
                        and abs(oc["z"] - c["z"]) < (os_["z"] + s["z"]) / 2 + clearance
                        and abs(oc["y"] - c["y"]) < (os_["y"] + s["y"]) / 2):
                    clash = True
                    break
            if clash:
                continue
            reach = self._c.step("GetReachablePositions").metadata["actionReturn"]
            trial = ThorPathfinder(reach)
            near = np.hypot(np.maximum(np.abs(trial._arr[:, 0] - c["x"]) - s["x"] / 2, 0),
                            np.maximum(np.abs(trial._arr[:, 1] - c["z"]) - s["z"] / 2, 0)).min()
            if trial.num_islands != islands0 or near > 1.0:
                continue
            pf.__init__(reach)
            return (c["x"], c["z"])
        return None

    @property
    def controller(self):
        return self._c
