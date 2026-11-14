"""Streams sim_bridge Frames to Rerun. Entity path convention chosen to be
consistent with the user's existing rosbag -> Rerun workflow: world-frame
transform tree under `world/agent`, sensor imagery under `world/agent/rgb`
and `world/agent/depth`, matching how a robot-mounted camera would be logged.
"""
from __future__ import annotations

import rerun as rr

from .habitat_env import Frame


def init(app_id: str = "rpv_mem_sim_bridge", serve: bool = True, connect_addr: str | None = None) -> None:
    """serve=True opens a web viewer server (rr.serve, open_browser=False since this
    box has no local display — the user connects a remote client to it).
    connect_addr, if given, instead streams to an already-running rerun viewer/server
    at that address (overrides serve).
    """
    rr.init(app_id, spawn=False)
    if connect_addr:
        rr.connect(connect_addr)
    elif serve:
        rr.serve(open_browser=False)


def log_frame(frame: Frame) -> None:
    rr.set_time_sequence("step", frame.step_id)
    rr.set_time_seconds("sim_time", frame.timestamp)

    rr.log(
        "world/agent",
        rr.Transform3D(
            translation=frame.pose["position"],
            rotation=rr.Quaternion(
                xyzw=[*frame.pose["rotation"][1:], frame.pose["rotation"][0]]
            ),
        ),
    )
    rr.log("world/agent/rgb", rr.Image(frame.rgb[:, :, :3]))
    rr.log("world/agent/depth", rr.DepthImage(frame.depth))
    rr.log("world/agent/position", rr.Points3D([frame.pose["position"]]))
