"""Phase 2 Step 2.4 verification: run a scripted episode streaming RGB/depth/pose
(Phase 1's sim_bridge) AND DAAAM's grounded object detections (Phase 2's
daaam_adapter bridge) into ONE Rerun recording, so both are visible together.

Caveat (logged, not fixed here): the `transform` pushed to DAAAM is the agent's raw
pose, not offset by the camera sensor's +1.5m height (configs/habitat.yaml
sensor.height) -- so DAAAM-computed 3D object positions are ~1.5m off from this
script's own world/agent frame. Fine for this smoke test's purpose (proving the two
streams merge into one synced recording); worth fixing before Phase 3/4 relies on
positions being metrically exact.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import rerun as rr
import yaml

from daaam_adapter.bridge_client import DaaamBridgeClient
from sim_bridge.habitat_env import HabitatEnv
from sim_bridge import rerun_stream

ACTIONS = ["move_forward", "turn_left", "turn_right"]


def camera_intrinsics(width: int, height: int, hfov_deg: float) -> dict:
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return {"fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2}


def wxyz_to_xyzw_transform(position, rotation_wxyz) -> list:
    w, x, y, z = rotation_wxyz
    return [position[0], position[1], position[2], x, y, z, w]


def main() -> None:
    random.seed(1)
    out_dir = Path("data/runs/phase2_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    rrd_path = out_dir / "episode_daaam.rrd"

    with open("configs/habitat.yaml") as f:
        habitat_cfg = yaml.safe_load(f)

    client = DaaamBridgeClient()
    print("health:", client.health())
    start_resp = client.start_episode()
    print("episode started:", start_resp)

    intrinsics = camera_intrinsics(
        habitat_cfg["sensor"]["rgb_resolution"][1],
        habitat_cfg["sensor"]["rgb_resolution"][0],
        habitat_cfg["sensor"]["hfov"],
    )

    rerun_stream.init(app_id="rpv_mem_phase2", serve=False)

    env = HabitatEnv(habitat_cfg)
    frame = env.reset()
    rerun_stream.log_frame(frame)

    n_steps = 60
    seen_semantic_ids = set()
    for i in range(n_steps):
        transform = wxyz_to_xyzw_transform(frame.pose["position"], frame.pose["rotation"])
        client.push_frame(
            frame_id=frame.step_id,
            timestamp=frame.timestamp,
            rgb=frame.rgb[:, :, :3],
            depth=frame.depth,
            transform=transform,
            camera_intrinsics=intrinsics,
        )

        for upd in client.pull_updates():
            rr.set_time_sequence("step", frame.step_id)
            rr.log(
                f"world/daaam/live_updates/{upd['semantic_id']}",
                rr.TextLog(upd["semantic_label"]),
            )

        action = random.choice(ACTIONS)
        frame = env.step(action)
        rerun_stream.log_frame(frame)

    print("waiting for grounding to finish...")
    time.sleep(15)

    positions = client.get_positions()
    stop_resp = client.stop_episode()
    corrections = {str(c["semantic_id"]): c for c in stop_resp.get("corrections", [])}
    print(f"final corrections: {len(corrections)}, objects with 3D positions: {len(positions)}")

    rr.set_time_sequence("step", n_steps)
    points = []
    labels = []
    for semantic_id, obs_list in positions.items():
        if not obs_list:
            continue
        pos = obs_list[-1]["position_world"]
        label = corrections.get(semantic_id, {}).get("semantic_label", f"semantic_id={semantic_id}")
        points.append(pos)
        labels.append(label[:60])
        rr.log(f"world/daaam/objects/{semantic_id}", rr.TextLog(label))

    if points:
        rr.log("world/daaam/objects", rr.Points3D(points, labels=labels, radii=0.08))

    env.close()
    rr.save(str(rrd_path))

    with open(out_dir / "positions.json", "w") as f:
        json.dump(positions, f, indent=2)

    print(f"saved recording to {rrd_path}")
    print(f"logged {len(points)} objects with 3D positions")


if __name__ == "__main__":
    main()
