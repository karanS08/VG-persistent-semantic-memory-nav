"""Camera projection helpers shared by revisit evidence and the image check (pipeline-v8).

Conventions are the runner's (Habitat frame, also emitted by sim_bridge.thor_env):
  * pose["position"] = agent BASE position [x, y_floor, z]; the camera sits `cam_height` above it;
  * pose["rotation"] = [w, x, y, z] quaternion, a pure rotation phi about +y;
  * camera frame: +x right, +y up, forward = -z; depth images hold planar (z) depth in metres.
The forward model matches scripts/thor/validate_thor_env.py (median back-projection error checked there).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def yaw_from_wxyz(rotation_wxyz) -> float:
    w, _, y, _ = rotation_wxyz
    return 2.0 * math.atan2(y, w)


def project_point(point_world: List[float], pose: Dict, intrinsics: Dict, cam_height: float
                  ) -> Optional[Tuple[float, float, float]]:
    """World point -> (u, v, planar_depth) in pixels/metres, or None if behind the camera."""
    pos = pose["position"]
    phi = yaw_from_wxyz(pose["rotation"])
    dx, dz = point_world[0] - pos[0], point_world[2] - pos[2]
    xc = dx * math.cos(phi) - dz * math.sin(phi)
    zc = dx * math.sin(phi) + dz * math.cos(phi)
    depth = -zc
    if depth <= 1e-3:
        return None
    yc = point_world[1] - (pos[1] + cam_height)
    u = intrinsics["cx"] + intrinsics["fx"] * xc / depth
    v = intrinsics["cy"] - intrinsics["fy"] * yc / depth
    return u, v, depth


def expected_visible(point_world: List[float], pose: Dict, depth_image: np.ndarray, intrinsics: Dict,
                     cam_height: float, min_range: float = 0.5, max_range: float = 3.0,
                     margin_px: int = 40, occlusion_tol: float = 0.4, clamp_vertical: bool = False) -> bool:
    """True if the camera should currently see `point_world`: in front, within [min_range, max_range] metres
    (floor-plane distance), projected inside the image with a `margin_px` border, and not hidden behind a
    nearer surface (median depth in a 7x7 window around the pixel >= predicted depth - occlusion_tol).

    clamp_vertical=True (image check only): the level camera's ~74 deg vertical FOV cuts off the floor closer than
    ~2 m, so at verify distance a piece of furniture's centre projects at or below the bottom edge while its upper
    part is in view; the row is then clamped into the image and the occlusion test runs there. Revisit evidence
    keeps the strict test, so an object merely below the field of view is never counted as 'seen empty'."""
    pos = pose["position"]
    rng = math.hypot(point_world[0] - pos[0], point_world[2] - pos[2])
    if not (min_range <= rng <= max_range):
        return False
    p = project_point(point_world, pose, intrinsics, cam_height)
    if p is None:
        return False
    u, v, depth = p
    d = np.asarray(depth_image)
    if d.ndim == 3:
        d = d[:, :, 0]
    h, w = d.shape
    if clamp_vertical:
        v = min(max(v, margin_px), h - margin_px - 1)
    if not (margin_px <= u < w - margin_px and margin_px <= v < h - margin_px):
        return False
    ui, vi = int(u), int(v)
    win = d[max(vi - 3, 0):vi + 4, max(ui - 3, 0):ui + 4]
    win = win[np.isfinite(win) & (win > 0)]
    if win.size == 0:
        return False
    return float(np.median(win)) >= depth - occlusion_tol


def depth_probe(point_world: List[float], pose: Dict, depth_image: np.ndarray, intrinsics: Dict, cam_height: float,
                min_range: float = 0.5, max_range: float = 3.0, margin_px: int = 40
                ) -> Optional[Tuple[float, float]]:
    """(predicted_depth, measured_depth) along the camera ray through `point_world`, or None if the point is out of
    range or outside the image (with margin). measured = median of a 7x7 window of the depth image. Comparing the
    two is free-space evidence, as in an occupancy grid: measured >> predicted means the ray passes THROUGH the
    remembered position (nothing there); measured ~ predicted means something occupies it; measured << predicted
    means it is hidden."""
    pos = pose["position"]
    if not (min_range <= math.hypot(point_world[0] - pos[0], point_world[2] - pos[2]) <= max_range):
        return None
    p = project_point(point_world, pose, intrinsics, cam_height)
    if p is None:
        return None
    u, v, pred = p
    d = np.asarray(depth_image)
    if d.ndim == 3:
        d = d[:, :, 0]
    h, w = d.shape
    if not (margin_px <= u < w - margin_px and margin_px <= v < h - margin_px):
        return None
    ui, vi = int(u), int(v)
    win = d[max(vi - 3, 0):vi + 4, max(ui - 3, 0):ui + 4]
    win = win[np.isfinite(win) & (win > 0)]
    if win.size == 0:
        return None
    return pred, float(np.median(win))


def crop_object(rgb: np.ndarray, point_world: List[float], pose: Dict, intrinsics: Dict, cam_height: float,
                up_m: float = 0.6, down_m: float = 0.4, half_width_m: float = 0.6) -> Optional[np.ndarray]:
    """Crop the image region an object around `point_world` would occupy: from up_m above to down_m below the
    point vertically and +-half_width_m horizontally, all projected at the point's depth and clipped to the image.
    Unlike a square around the projected point, this still frames the visible upper body of furniture whose
    centre projects below the bottom edge (common at verify distance with a level camera)."""
    p = project_point(point_world, pose, intrinsics, cam_height)
    if p is None:
        return None
    u, _, d = p
    top = project_point([point_world[0], point_world[1] + up_m, point_world[2]], pose, intrinsics, cam_height)
    bot = project_point([point_world[0], point_world[1] - down_m, point_world[2]], pose, intrinsics, cam_height)
    h, w = rgb.shape[:2]
    half = intrinsics["fx"] * half_width_m / d
    x0, x1 = max(int(u - half), 0), min(int(u + half), w)
    y0, y1 = max(int(top[1]), 0), min(int(bot[1]), h)
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    return rgb[y0:y1, x0:x1]


def crop_around(rgb: np.ndarray, u: float, v: float, depth: float, fx: float, half_extent_m: float = 0.75,
                min_half_px: int = 48, max_half_px: int = 220) -> Optional[np.ndarray]:
    """Square crop centred on (u, v) covering about +-half_extent_m at `depth`, clipped to the image."""
    half = int(max(min_half_px, min(max_half_px, fx * half_extent_m / max(depth, 1e-3))))
    h, w = rgb.shape[:2]
    x0, x1 = max(int(u) - half, 0), min(int(u) + half, w)
    y0, y1 = max(int(v) - half, 0), min(int(v) + half, h)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return rgb[y0:y1, x0:x1]
