# Real-time optimization notes

Scope: this covers the `daaam_bridge_server`/`DaaamBridgeClient` transport layer
specifically. See `PROGRESS.md`/`TODO.md` for the broader real-time-capability
picture (segmentation+tracking already runs ~150-200ms/frame; DAM-3B grounding is
~6.6-7.2s/query and is not, and should not be, on the per-frame critical path).

## What's implemented (this pass)

- **`DaaamBridgeClient` now reuses a single `requests.Session`** instead of the
  module-level `requests.get`/`requests.post` (which open a fresh TCP connection,
  full 3-way handshake, per call). Verified against a real local test server
  (`packages/daaam_adapter/tests/test_bridge_client.py`) that 6 sequential calls
  through one client share exactly 1 TCP connection, not 6. This is a real,
  measurable per-call saving (handshake + slow-start avoided), though small next
  to DAAAM's own multi-second grounding latency — it matters more for the
  steady per-frame `push_frame`/`pull_updates`/`get_positions` cadence than for
  the grounding path itself.
- **Cached the query CLIP text embedding on `FrontierAgent`** (keyed by query
  string, not cleared by `reset()`), so `decide()` no longer re-embeds the same
  query text on every single call within (or across) episodes.

## What's documented here, not implemented (needs live testing against daaam_dev)

Both of these require coordinated client+server changes and real testing against
the running container/GPU, which was explicitly out of scope for this pass
(another process was using the live bridge for value-map calibration/ablation
runs concurrently). Recommended for a dedicated follow-up pass with the bridge
free to restart.

### 1. Binary payload instead of base64-JSON for RGB/depth

Current: `_encode_array` base64-encodes the raw frame bytes and embeds them in a
JSON body. Rough numbers for the actual resolution in use (480x640):

- RGB (480x640x3, uint8): ~921KB raw -> ~1.23MB base64 (+33% size) -> JSON
  string-escaping/parsing overhead on top of that on both ends.
- Depth (480x640, float32): ~1.23MB raw -> ~1.64MB base64.
- Combined: **~1.5MB raw -> ~2.9MB of base64+JSON per frame pushed.**

At the loop rates actually observed this session (DAAAM's own segmentation+
tracking sync path: ~150-200ms/frame, i.e. ~5-6fps ceiling before grounding
latency even enters the picture), this is a few MB/s of avoidable
serialization + copy overhead — real, but not dominant compared to DAM-3B's
~7s/query grounding calls, which stay off the per-frame path regardless. Worth
fixing for genuine sustained-fps operation (e.g. an eventual real Spot/RealSense
deployment streaming continuously), not urgent for the current batch/episode
testing cadence.

Recommended fix: switch `/frame` to `multipart/form-data` (or a raw
`application/octet-stream` body with a small fixed-size binary header encoding
shape/dtype/frame_id/timestamp/transform/intrinsics, avoiding JSON's text
overhead entirely) for the RGB+depth payload specifically. Keep JSON for the
low-frequency, small-payload endpoints (`/positions`, `/updates`,
`/episode/start`, `/episode/stop`) where base64/JSON overhead is negligible.

### 2. Collapse 3 round-trips/step into 1

Current per-step pattern (see `run_phase4_batch.py`'s loop): `push_frame` (POST),
`pull_updates` (GET), `get_positions` (GET) -- 3 separate HTTP request/response
cycles every step, even though `pull_updates`+`get_positions` are always called
together immediately after `push_frame` in every caller (`run_phase4_batch.py`,
`run_phase4_closed_loop.py`, `run_phase2_rerun.py` all follow this exact
sequence). With Session reuse (this pass) each round-trip avoids a new TCP
handshake, but still pays a full HTTP request/response cycle (Flask routing,
JSON encode/decode, GIL handoff) 3x per step instead of 1x.

Recommended fix: add a single combined endpoint (e.g. `POST /step`) that does
`process_frame` + drains `_pending_updates` + snapshots `object_3d_positions` in
one server-side call, returning all three in one JSON response. This is a
larger change than it looks: `daaam_bridge_server.py`'s three routes currently
have independent locking or none at all around `_orchestrator`/`_pending_updates`
access -- combining them needs the same care taken with the existing
`_state_lock` usage, and should be tested against a live episode (does draining
updates less granularly change any timing-sensitive behavior, e.g. the
already-documented live-stream-vs-stop()-drain race from Phase 2?) before
trusting it. Flagging as the higher-value fix of the two, but the riskier one to
make blind.
