"""Phase 1 verification entrypoint: scripted random-walk episode in Habitat-Sim,
streamed to Rerun, saved as .rrd for remote/offline inspection.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import rerun as rr
import yaml

from sim_bridge.habitat_env import HabitatEnv
from sim_bridge import rerun_stream

ACTIONS = ["move_forward", "turn_left", "turn_right"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/habitat.yaml")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rrd-out", default="data/runs/phase1_smoke/episode.rrd")
    parser.add_argument("--serve", action="store_true", help="also serve a live web viewer")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    Path(args.rrd_out).parent.mkdir(parents=True, exist_ok=True)

    rerun_stream.init(serve=args.serve)

    env = HabitatEnv(config)
    frame = env.reset()
    rerun_stream.log_frame(frame)
    shapes = [(frame.rgb.shape, frame.depth.shape)]

    for _ in range(args.steps):
        action = random.choice(ACTIONS)
        frame = env.step(action)
        rerun_stream.log_frame(frame)
        shapes.append((frame.rgb.shape, frame.depth.shape))

    env.close()
    rr.save(args.rrd_out)

    print(f"ran {args.steps} steps, episode_id={frame.episode_id}, final step_id={frame.step_id}")
    print(f"first frame shapes: rgb={shapes[0][0]} depth={shapes[0][1]}")
    print(f"last frame shapes:  rgb={shapes[-1][0]} depth={shapes[-1][1]}")
    print(f"saved recording to {args.rrd_out}")


if __name__ == "__main__":
    main()
