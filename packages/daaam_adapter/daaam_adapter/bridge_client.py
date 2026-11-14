"""Client for daaam_bridge_server (runs inside daaam_dev, see
../bridge_server/daaam_bridge_server.py). Talks HTTP across the Python 3.9 (this
env) <-> Python 3.12 (daaam_dev) interpreter split — see PROGRESS.md Phase 2 for
why this split exists and why it's a push-frame / pull-updates split rather than a
single synchronous call (DAAAM's grounding is async and batched).
"""
from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional

import numpy as np
import requests


def _encode_array(arr: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    return {
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


class DaaamBridgeClient:
    """Real-time-optimization note (see PROGRESS.md / TODO.md): this reuses a
    single requests.Session so the TCP connection to daaam_bridge_server is kept
    alive across calls (HTTP keep-alive + connection pooling) instead of a fresh
    handshake per push_frame/pull_updates/get_positions -- a real, measurable
    per-call saving, though small next to DAAAM's own ~7s grounding latency. The
    bigger transport wins (binary payloads instead of base64 JSON, collapsing the
    3 per-step calls into 1) are documented but NOT implemented here since they
    require coordinated changes to daaam_bridge_server, which needs live testing
    against the running container -- out of scope for this pass, see
    docs/realtime_optimization_notes.md.
    """

    def __init__(self, base_url: str = "http://localhost:8765", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def health(self) -> Dict[str, Any]:
        r = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def start_episode(self, config_path: str = "/tmp/bridge_pipeline_config.yaml") -> Dict[str, Any]:
        r = self._session.post(f"{self.base_url}/episode/start", json={"config_path": config_path}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def push_frame(
        self,
        frame_id: int,
        timestamp: float,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        transform: Optional[List[float]] = None,
        camera_intrinsics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "rgb": _encode_array(rgb),
            "depth": _encode_array(depth),
            "transform": transform,
            "camera_intrinsics": camera_intrinsics,
        }
        r = self._session.post(f"{self.base_url}/frame", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        r = self._session.get(f"{self.base_url}/positions", timeout=self.timeout)
        r.raise_for_status()
        return r.json()["positions"]

    def pull_updates(self) -> List[Dict[str, Any]]:
        r = self._session.get(f"{self.base_url}/updates", timeout=self.timeout)
        r.raise_for_status()
        return r.json()["updates"]

    def stop_episode(self) -> Dict[str, Any]:
        r = self._session.post(f"{self.base_url}/episode/stop", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def verify_region(self, rgb: np.ndarray, query_text: str) -> Dict[str, Any]:
        """2026-09-02: synchronous VQA-style verification -- 'is the object in
        this image a {query_text}?', a direct DAM-3B yes/no judgment on the
        current frame, not a CLIP-similarity threshold. See
        daaam_bridge_server.py's /verify docstring for the full motivation.
        Real measured latency: ~10s on first call (model load), ~0.4s on
        subsequent calls once warm. Returns {"verified": bool, "raw_answer": str}."""
        payload = {"rgb": _encode_array(rgb), "query_text": query_text}
        # DAAAM_VERIFY_URL (2026-10-30): send /verify to the stand-alone verify server instead of the bridge,
        # see bridge_server/daaam_verify_server.py for why. Unset = unchanged behaviour.
        verify_url = os.environ.get("DAAAM_VERIFY_URL", self.base_url)
        r = self._session.post(f"{verify_url}/verify", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
