"""Stand-alone VQA verification server (2026-10-30), port 8766.

Same /verify contract, model (nvidia/DAM-3B, conv_mode v1, focal_prompt), prompt and yes/no parsing as
daaam_bridge_server.py's /verify -- moved into its own process because loading DAM-3B inside the bridge's
main process stopped grounding updates from reaching /updates for the rest of the session (THOR v5
diagnosis: captions kept being created by the grounding worker but never arrived after the first /verify).
Clients use it when DAAAM_VERIFY_URL is set (see daaam_adapter.bridge_client.DaaamBridgeClient.verify_region).
"""
from __future__ import annotations

import base64
import os
import threading

import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)
_verify_lock = threading.Lock()
_verify_agent = None


def _get_verify_agent():
    global _verify_agent
    if _verify_agent is None:
        from daaam.query_manager.dam.services import DAMAgentPanoptic
        _verify_agent = DAMAgentPanoptic(model_path="nvidia/DAM-3B", conv_mode="v1", prompt_mode="focal_prompt")
    return _verify_agent


def _decode_array(payload, key):
    if payload.get(key) is None:
        return None
    raw = base64.b64decode(payload[key]["data"])
    return np.frombuffer(raw, dtype=payload[key]["dtype"]).reshape(payload[key]["shape"])


@app.route("/verify", methods=["POST"])
def verify_region():
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

    global _verify_agent
    with _verify_lock:
        agent = _get_verify_agent()
        answer = agent.query_single(image=image, mask=full_mask, query=prompt, max_new_tokens=64)
        if os.environ.get("VERIFY_UNLOAD", "1") == "1":
            # free the ~9 GB again right away: with this copy resident next to the grounding worker's own DAM-3B
            # the GPU sits at ~21.5/24 GB and captioning throughput collapses (43 vs 141 captions per episode)
            del agent
            _verify_agent = None
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()

    answer_lower = (answer or "").strip().lower()
    return jsonify({"verified": answer_lower.startswith("yes"), "raw_answer": answer})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "loaded": _verify_agent is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8766, threaded=True)
