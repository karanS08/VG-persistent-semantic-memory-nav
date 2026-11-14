"""Phase 2 Step 2.2/2.3 verification: push real HM3D frames from HabitatEnv through
the daaam_bridge_server, confirm DAAAM's async grounding actually produces
ObjectAnnotation-derived updates, save raw output for evidence.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import yaml

from daaam_adapter.bridge_client import DaaamBridgeClient
from sim_bridge.habitat_env import HabitatEnv

ACTIONS = ["move_forward", "turn_left", "turn_right"]


def camera_intrinsics(width: int, height: int, hfov_deg: float) -> dict:
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return {"fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2}


def wxyz_to_xyzw_transform(position, rotation_wxyz) -> list:
    w, x, y, z = rotation_wxyz
    return [position[0], position[1], position[2], x, y, z, w]


def main() -> None:
    random.seed(0)
    out_dir = Path("data/runs/phase2_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

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

    env = HabitatEnv(habitat_cfg)
    frame = env.reset()

    n_steps = 40
    all_updates = []
    push_log = []
    for i in range(n_steps):
        transform = wxyz_to_xyzw_transform(frame.pose["position"], frame.pose["rotation"])
        resp = client.push_frame(
            frame_id=frame.step_id,
            timestamp=frame.timestamp,
            rgb=frame.rgb[:, :, :3],
            depth=frame.depth,
            transform=transform,
            camera_intrinsics=intrinsics,
        )
        push_log.append(resp)
        print(f"step {i}: pushed frame {frame.step_id}, num_tracks={resp.get('num_tracks')}")

        updates = client.pull_updates()
        if updates:
            print(f"  -> {len(updates)} new update(s)")
            all_updates.extend(updates)

        action = random.choice(ACTIONS)
        frame = env.step(action)

    # give async grounding workers a little time to finish after the last frame
    print("waiting for late updates...")
    time.sleep(15)
    all_updates.extend(client.pull_updates())

    env.close()
    stop_resp = client.stop_episode()
    print("episode stopped:", {k: v for k, v in stop_resp.items() if k != "corrections"})
    print(f"final corrections: {len(stop_resp.get('corrections', []))}")

    with open(out_dir / "push_log.json", "w") as f:
        json.dump(push_log, f, indent=2)
    with open(out_dir / "updates.json", "w") as f:
        json.dump(all_updates, f, indent=2)
    with open(out_dir / "final_corrections.json", "w") as f:
        json.dump(stop_resp.get("corrections", []), f, indent=2)

    print(f"saved raw output to {out_dir}/")
    print(f"total streamed updates: {len(all_updates)}")


if __name__ == "__main__":
    main()
