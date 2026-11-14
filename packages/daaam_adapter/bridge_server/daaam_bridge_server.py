"""Runs INSIDE the daaam_dev container (Python 3.12, DAAAM importable there).
Exposes DAAAM's PipelineOrchestrator over HTTP so daaam_adapter (a different
Python interpreter, in the rpv_mem conda env) can drive it without sharing a
process — mirrors rvp_semnav's own vlfm/vlm/server_wrapper.py CLIP-server pattern,
adapted for DAAAM's async/batched grounding (see PROGRESS.md Phase 2 Step 2.1):
frames are pushed synchronously, but ObjectAnnotation results arrive later and are
pulled separately rather than returned from the push call.
"""
from __future__ import annotations

import base64
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from flask import Flask, jsonify, request

from daaam.config import PipelineConfig
from daaam.pipeline.orchestrator import PipelineOrchestrator
from daaam.pipeline.models import Frame
from daaam.utils.logging import get_default_logger

# Upstream bug workaround: PipelineOrchestrator.__init__ constructs its
# WorkerHealthMonitor with the raw `logger` *parameter* (None if not passed),
# not `self.logger` (which defaults to get_default_logger()) -- so
# WorkerHealthMonitor.logger ends up None and .start()/.stop() crash with
# AttributeError. Passing an explicit logger here sidesteps it. See
# PROGRESS.md Phase 2 for the traceback that surfaced this.
_logger = get_default_logger()

app = Flask(__name__)

_state_lock = threading.Lock()
_orchestrator: PipelineOrchestrator | None = None
_pending_updates: List[Dict[str, Any]] = []

# 2026-09-02: synchronous VQA-style verification, added alongside the existing
# async captioning pipeline (see PROGRESS.md "verification redesign"). A
# SEPARATE DAMAgentPanoptic instance from the one(s) the async orchestrator's
# grounding workers use -- deliberately not reaching into orchestrator's
# worker-process internals (real risk of destabilizing the existing, already-
# validated async pipeline for an unrelated feature). Lazily initialized on
# first /verify call so normal episodes that never use it pay no extra
# startup cost or GPU memory. Guarded by its own lock: GPU inference on a
# single model instance is not safely concurrent, and this pipeline only ever
# has one agent verifying one candidate at a time anyway, so serializing here
# costs nothing real.
_verify_lock = threading.Lock()
_verify_agent = None


def _get_verify_agent():
    global _verify_agent
    if _verify_agent is None:
        from daaam.query_manager.dam.services import DAMAgentPanoptic
        # Same model/conv_mode/prompt_mode as the main pipeline
        # (bridge_pipeline_config.yaml's grounding.dam_model_path etc.) --
        # reusing the already-validated config rather than a new guess.
        _verify_agent = DAMAgentPanoptic(
            model_path="nvidia/DAM-3B", conv_mode="v1", prompt_mode="focal_prompt",
        )
    return _verify_agent


def _decode_array(payload: Dict[str, Any], key: str) -> np.ndarray | None:
    if payload.get(key) is None:
        return None
    raw = base64.b64decode(payload[key]["data"])
    arr = np.frombuffer(raw, dtype=payload[key]["dtype"]).reshape(payload[key]["shape"])
    return arr


def _semantic_update_callback(update) -> None:
    """Registered on the orchestrator; buffers ObjectAnnotation-derived updates for
    /updates to drain. `update` is a daaam.pipeline.models.SemanticUpdate."""
    with _state_lock:
        for semantic_id, label in update.semantic_labels.items():
            entry: Dict[str, Any] = {
                "semantic_id": semantic_id,
                "semantic_label": label,
                "timestamp": update.timestamp,
            }
            temporal = update.temporal_observations.get(semantic_id)
            if temporal is not None:
                entry["observation_count"] = temporal.observation_count
                entry["first_observed"] = temporal.first_observed
                entry["last_observed"] = temporal.last_observed
            features = update.features.get(semantic_id)
            # `semantic_embedding_feature` needs a sentence_embedding_model_name
            # (unconfigured, always empty -- same gap as /episode/stop had before
            # its earlier fix, see PROGRESS.md Phase 3). `clip_feature` is what's
            # actually populated (sourced from ObjectAnnotation.selectframe_clip_feature
            # via orchestrator._trigger_semantic_update) -- this callback was missed
            # when /episode/stop got the equivalent fix; caught via a real 0/N
            # embedding-coverage diagnostic in a Phase 4 batch run.
            if features is not None and features.clip_feature:
                entry["embedding"] = features.clip_feature
            _pending_updates.append(entry)


@app.route("/episode/start", methods=["POST"])
def episode_start():
    global _orchestrator
    with _state_lock:
        if _orchestrator is not None:
            return jsonify({"error": "episode already running, call /episode/stop first"}), 409
        payload = request.get_json(force=True)
        config_path = payload.get("config_path", "/tmp/bridge_pipeline_config.yaml")
        config = PipelineConfig.from_yaml(config_path)
        _orchestrator = PipelineOrchestrator(config, logger=_logger)
        _orchestrator.set_semantic_update_callback(_semantic_update_callback)
        _orchestrator.start()
        _pending_updates.clear()
    return jsonify({"status": "started", "output_dir": str(_orchestrator.output_dir)})


@app.route("/frame", methods=["POST"])
def push_frame():
    with _state_lock:
        if _orchestrator is None:
            return jsonify({"error": "no episode running, call /episode/start first"}), 409
        orchestrator = _orchestrator

    payload = request.get_json(force=True)
    rgb = _decode_array(payload, "rgb")
    depth = _decode_array(payload, "depth")

    frame = Frame(
        frame_id=payload.get("frame_id", 0),
        timestamp=payload["timestamp"],
        rgb_image=rgb,
        depth_image=depth,
        transform=np.array(payload["transform"]) if payload.get("transform") is not None else None,
        camera_intrinsics=payload.get("camera_intrinsics"),
    )
    try:
        orchestrator.process_frame(frame)
    except Exception as e:
        # Known failure mode: boxmot's BotSort.update() can return a malformed
        # 1D array (instead of (M,8)) after its internal ECC/CMC affine estimation
        # hits a NaN (see the "Returning identity" warning preceding this in the
        # log) -- daaam/tracking/services.py only guards the zero-detections case,
        # not this one. Not something to patch blind in an unfamiliar third-party
        # tracker; skip the frame defensively here instead of crashing the episode.
        app.logger.warning(f"process_frame failed for frame_id={frame.frame_id}, skipping: {e}")
        return jsonify({"status": "skipped", "frame_id": frame.frame_id, "error": str(e)}), 200
    return jsonify({"status": "ok", "frame_id": frame.frame_id, "num_tracks": len(frame.tracks)})


@app.route("/updates", methods=["GET"])
def pull_updates():
    with _state_lock:
        updates = list(_pending_updates)
        _pending_updates.clear()
    return jsonify({"updates": updates})


@app.route("/positions", methods=["GET"])
def get_positions():
    """Exposes orchestrator.object_3d_positions (already computed internally from
    mask centroid + median depth + camera pose, see orchestrator._process_tracks)
    -- not returned by /updates or /episode/stop, added specifically so Rerun
    visualization can place detections in 3D rather than just a timeline."""
    with _state_lock:
        if _orchestrator is None:
            return jsonify({"error": "no episode running"}), 409
        orchestrator = _orchestrator
        positions = {
            str(semantic_id): [
                {
                    "position_world": [float(x) for x in obs["position_world"]],
                    "frame_id": obs["frame_id"],
                    "timestamp": obs["timestamp"],
                }
                for obs in obs_list
            ]
            for semantic_id, obs_list in orchestrator.object_3d_positions.items()
        }
    return jsonify({"positions": positions})


@app.route("/episode/stop", methods=["POST"])
def episode_stop():
    global _orchestrator
    with _state_lock:
        if _orchestrator is None:
            return jsonify({"error": "no episode running"}), 409
        orchestrator = _orchestrator
        _orchestrator = None

    orchestrator.stop()
    corrections = orchestrator.scene_graph_service.corrections
    summary = [
        {
            "semantic_id": sid,
            "semantic_label": c.semantic_label,
            "confidence": c.confidence,
            # `embedding` needs a sentence_embedding_model_name configured (it isn't,
            # by default) and is always None as a result -- `selectframe_clip_feature`
            # (a real CLIP ViT-L-14 image-crop embedding) is what's actually populated
            # with `enable_selectframe_clip_features: true`. See PROGRESS.md Phase 3.
            "embedding": c.selectframe_clip_feature,
            "timestamp": c.timestamp,
            # c.timestamp is arrival time (Annotation's default time.time() at
            # correction creation), not when the object was actually seen -- same
            # gap as the /updates callback had, see PROGRESS.md 2026-08-24.
            # first_observed/last_observed are the real, per-frame-tracked window.
            "first_observed": c.first_observed,
            "last_observed": c.last_observed,
        }
        for sid, c in corrections.items()
    ]
    return jsonify({"status": "stopped", "output_dir": str(orchestrator.output_dir), "corrections": summary})


@app.route("/verify", methods=["POST"])
def verify_region():
    """Synchronous VQA-style verification: 'is the object in this image a
    {query}?' -- a direct semantic judgment from DAM-3B, not a CLIP-similarity
    threshold. See PROGRESS.md "verification redesign" for why: CLIP cosine
    similarity between a short query and a stored caption's embedding sits
    inside CLIP's well-known 0.1-0.3 noise floor at the values this pipeline
    actually produces, and no amount of gating on top of that raw number
    fully closes the false-positive risk (four distinct mechanisms found and
    fixed this project). A direct yes/no judgment on the actual current view
    is a categorically different, more reliable signal for this exact
    decision. Whole-image query (no per-object mask) -- deliberately simple:
    the caller is expected to only call this once physically close to a
    specific candidate, so the target object should be prominent in view
    without needing DAAAM's internal per-track segmentation mask, which the
    Python 3.9 side does not have access to anyway (masks stay inside the
    container's async tracking pipeline)."""
    payload = request.get_json(force=True)
    rgb = _decode_array(payload, "rgb")
    if rgb is None:
        return jsonify({"error": "rgb required"}), 400
    query_text = payload.get("query_text")
    if not query_text:
        return jsonify({"error": "query_text required"}), 400

    from PIL import Image as PILImage
    image = PILImage.fromarray(rgb).convert("RGB")
    full_mask = PILImage.new("L", image.size, 255)  # whole image, no per-object mask
    prompt = f"Is there a {query_text} clearly visible in this image? Answer only 'yes' or 'no', then briefly say what you actually see."

    with _verify_lock:
        agent = _get_verify_agent()
        answer = agent.query_single(image=image, mask=full_mask, query=prompt, max_new_tokens=64)

    answer_lower = (answer or "").strip().lower()
    verified = answer_lower.startswith("yes")
    return jsonify({"verified": verified, "raw_answer": answer})


@app.route("/health", methods=["GET"])
def health():
    with _state_lock:
        running = _orchestrator is not None
    return jsonify({"status": "ok", "episode_running": running})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8765, threaded=True)
