# Spot Deployment Plan

Status: planning only, nothing built. Grounded against actual repo files where
noted (`[verified: <path>]`); everything else is a design proposal, not yet
implemented or tested against real hardware.

## 0. Why this document exists

The rest of this project (`~/mem_ws`) runs the full loop — sim → DAAAM → memory →
frontier agent → action — against Habitat-Sim. This document is the plan for
replacing Habitat-Sim with a real Boston Dynamics Spot robot, compute offboard on
a powerful external machine (not Spot's own payload compute), everything else in
the pipeline unchanged.

## 1. Sensor payload

**Recommendation: Intel RealSense D455** (D435i is the fallback if depth range at
the wide end matters less than robustness/cost).

Spot's 5 built-in stereo camera pairs are greyscale, fisheye, low-resolution, and
tuned for local obstacle avoidance — not suitable as FastSAM/DAM-3B input. This
isn't a guess: `rvp_semnav`'s own real-Spot deployment code (`reality_policies.py`,
`RealityMixin`) doesn't use Spot's native depth at all — it runs ZoeDepth for
monocular depth *estimation* from RGB instead, real evidence the original team
found Spot's stock sensing insufficient for this class of pipeline too.

Depth range needed: the bridge's currently-configured depth gate is
`depth_lb: 0.1`, `depth_ub: 10.0` meters `[verified:
packages/daaam_adapter/bridge_pipeline_config.yaml]` — though that file's own
comment flags this as "widened as a precaution, not cross-checked against a real
depth scale" for Habitat's synthetic sensor, so treat 0.1–10m as directional, not
final. For comparison, `daaam_ros`'s own CODa launch defaults are wider still —
`depth_lb: 0.05`, `depth_ub: 20.0` — and the Hydra input config additionally gates
sensor range to `min_range: 0.3, max_range: 15.0` `[verified:
workspace/src/daaam_ros/launch/coda_daaam_hydra.launch.yaml,
workspace/src/daaam_ros/config/hydra_ros_config/coda_dataset_input_config.yaml]`.
D455 covers roughly 0.4–20m (better far-range accuracy than D435i's ~0.3–3m
practical useful range), so it's the better match to the *wider* range DAAAM's own
default config actually expects, not just the currently-narrower Habitat-tuned
bridge config.

ZED2i (stereo, not active-IR) was considered and rejected as first choice: better
outdoor/sunlight robustness, but no onboard active depth means worse close-range
(<0.5m) accuracy indoors, which matters more for this pipeline's typical object-
search distances than outdoor range does. Revisit if deployment is outdoor-heavy.

Mounting: Spot's payload rails (front or top depending on FOV needs relative to
walking gait — front-mounted risks the robot's own legs/shadow entering frame at
downward angles). Power: D455 draws ~1-3W via USB3 from the payload's own power,
well within Spot's ~20W-per-payload-port budget; not a real constraint here.

**Depth encoding gotcha, real and easy to miss**: `daaam_ros`'s launch exposes a
`depth_scale` arg, default `'1000.0'`, described as "depth image scale factor (e.g.
1000 for mm to m)" `[verified: coda_daaam_hydra.launch.yaml]` — i.e. the pipeline
expects raw depth images as **uint16 millimeters**, the standard RealSense/ROS
`sensor_msgs/Image` depth convention, not float32 meters. A RealSense ROS2 wrapper
publishes exactly this by default, so this should just work — but it's a real,
specific format contract to verify on first bring-up (a float-meters depth stream
would silently produce depth values 1000x too small and fail the `depth_lb`/
`depth_ub` gate entirely, reading as "no valid depth anywhere").

## 2. Topic/network architecture

Confirmed exact topics `daaam_ros`'s coda launch expects `[verified:
coda_daaam_hydra.launch.yaml]`:

| Topic (launch arg default) | Type | Source |
|---|---|---|
| `/cam0/rgb_image` | `sensor_msgs/Image` | RealSense RGB stream |
| `/cam0/camera_info` | `sensor_msgs/CameraInfo` | RealSense RGB intrinsics |
| `/cam0/depth_image` | `sensor_msgs/Image` (uint16 mm) | RealSense depth stream, registered to RGB |
| `/segmentation/image_raw`, `/segmentation/color/image_raw` | published BY `daaam_node`, not input | n/a |
| pose | via `/tf`, NOT a topic arg — `odom_topic` defaults to `/none` `[verified: same file]` | robot localization, see §3 |

These are launch **args**, not hardcoded — so a `realsense-ros` launch simply needs
to publish under these names (or remap to them), and the existing DAAAM/DSG launch
needs zero code changes on the DAAAM side. This is the whole point of matching the
existing contract rather than inventing a new one.

**Network topology.** DAAAM requires an NVIDIA GPU with CUDA 12.8+, confirmed
against this exact RTX 3090/24GB deployment `[verified: ~/daaam_docker/README.md]`,
and DAM-3B alone holds ~8.6GB VRAM once loaded (confirmed via this session's real
GPU logs). No companion computer realistically small enough to ride on Spot's
payload rails (Spot CORE I/O, a NUC-class board) carries a GPU anywhere near that
class. So DAAAM (and this project's `rpv_mem` stack) **cannot** run on Spot's own
compute — it must run on the offboard machine, exactly as already planned. The
companion computer's only job is: run the RealSense driver + Spot SDK client,
publish `/cam0/*` + `/tf` over the network to the offboard machine's ROS2 graph
(same `network_mode: host` + shared `ROS_DOMAIN_ID` pattern `daaam_docker` already
uses for its own container `[verified: same README]` — the offboard machine's
`daaam_dev` container just needs to be reachable on the same DDS domain as the
companion computer, e.g. both on the same WiFi/Ethernet segment or bridged via
Zerotier/Tailscale the way this session's own Rerun viewer was exposed).

Given DAM-3B's own ~6.6-7.2s/query latency (confirmed via this session's real
server logs, `daaam_bridge_server.log`) dwarfs any plausible RGB-D-over-WiFi
transfer time for a 640x480 frame at a few fps, network bandwidth/latency between
companion computer and offboard machine is not expected to be the bottleneck —
consistent with what was already concluded earlier this session, now grounded in
the actual topic/format contract rather than just the general architecture shape.

## 3. Pose / localization

Spot's onboard state estimator (visual-inertial, `frame_helpers.py` in the Spot
SDK) publishes a pose in Spot's own `vision` or `odom` frame — genuinely good
quality, this is core to how Spot walks at all. Two real gotchas, not hand-waved:

1. **`Frame.transform` format** (`sim_bridge.habitat_env.Frame`, this project's own
   contract) is `[x, y, z, qx, qy, qz, qw]` — Spot's SDK returns poses as
   `SE3Pose` (`position` + `rotation` as a quaternion with `.w/.x/.y/.z` field
   order, protobuf convention) — a `spot_bridge` package needs a small, explicit
   reorder (`w,x,y,z` → `x,y,z,w`), the exact same class of quaternion-order gotcha
   this session already hit once for real inside `frontier_agent/heading.py`
   (verified empirically against a real Habitat step before trusting it — same
   discipline applies here, verify against a real Spot SDK response before
   shipping this reorder blind).

2. **Frame-name mismatch, real and specific**: the Hydra input config hardcodes
   `extrinsics: {type: ros, robot_frame: os1}` `[verified:
   coda_dataset_input_config.yaml]` — `os1` is CODa's lidar-derived body-frame
   name, not a Spot concept at all. The launch's `robot_frame` arg (default `os1`)
   and `sensor_frame` arg (default `cam0_optical_frame`) both need overriding to
   whatever the actual TF tree calls Spot's body frame (`body`, per Spot SDK
   convention) and the RealSense's optical frame (whatever `realsense-ros`
   publishes, typically `camera_color_optical_frame` by default — needs an
   explicit remap or launch-arg override either way, not a silent assumption).
   `odom_frame` (default `world`) and `map_frame` (default `map`) are generic
   enough to likely need no change, but confirm once a real Spot TF tree exists to
   compare against.

## 4. What changes in this project's own code

`sim_bridge.HabitatEnv` (`packages/sim_bridge/sim_bridge/habitat_env.py`) is
Habitat-specific in two real ways that a live deployment can't reuse directly:

- **`reset()`/`step(action)`** — scripted, synchronous, blocking calls into
  `habitat_sim.Simulator`. A live robot has no `reset()` (an episode doesn't
  "restart" the physical world) and `step()` needs to become async (issue a Spot
  SDK command, then poll/wait for completion or timeout) rather than a blocking
  call that returns the next frame immediately.
- **`self.pathfinder`** (`HabitatEnv.pathfinder` property, used throughout
  `frontier_agent/frontier.py`'s `sample_frontier_candidates` for navmesh-based
  candidate sampling) — Spot has no navmesh. Two real options, not a coin flip:
  - **(a) Spot's own local grid/obstacle-avoidance API** (`local_grid_pb2` /
    `LocalGridClient` in the Spot SDK) — gives real, live-sensed traversability
    immediately around the robot, but only in a fairly short radius (Spot's local
    grid is a local costmap, not a persistent map of the whole space) — good for
    short-range candidate sampling, not for candidates far from the robot's
    current position.
  - **(b) Sample from Hydra's own TRAVERSABILITY layer** — DAAAM/Hydra is already
    building this live as part of its own scene-graph construction (confirmed
    present as a real layer in the DSG hierarchy, per this session's earlier
    scene-graph research: `SEGMENTS → OBJECTS → AGENTS → PLACES/TRAVERSABILITY →
    ROOMS → BUILDINGS`). This is the more architecturally consistent choice — it
    reuses data DAAAM is already computing instead of adding a second,
    independent traversability source — but requires pulling that layer out of
    the DSG binary (`spark_dsg.DynamicSceneGraph`, same deserialization pattern
    `daaam_bridge_server.py` already uses for other layers) rather than calling a
    ready-made Spot SDK function, so it's more implementation work up front for a
    more consistent long-term architecture. **Recommended**: (b) for real
    candidate sampling once a live DSG exists, with (a) as a cheap fallback/sanity
    check during early bring-up before DAAAM's TRAVERSABILITY layer is trusted.

  Either way, `sample_frontier_candidates`'s current signature (pathfinder +
  agent_position → list of `FrontierCandidate`) is the right abstraction boundary
  to preserve — a `spot_bridge`-side implementation just needs to satisfy the same
  interface, not a rewrite of `frontier_agent` itself.

- **New `spot_bridge` package** (mirroring `sim_bridge`'s shape): wraps the Spot
  SDK connection + RealSense subscription, exposes the same `Frame`-shaped data
  (`rgb_image`, `depth_image`, `transform`, `camera_intrinsics`, `timestamp`) that
  `daaam_adapter.bridge_client.push_frame` already consumes — meaning
  `daaam_adapter`, `memory_layer`, and `frontier_agent` need **zero** changes; only
  `sim_bridge` gets a live-deployment sibling, not a replacement (Habitat-Sim stays
  for continued offline development/testing).

## 5. Phased test plan

Each phase has a real go/no-go gate, not a vague milestone.

1. **Bench test** (Spot standing still, tethered/on a stand). Validate: RealSense
   publishes `/cam0/*` at expected rate/format, `depth_scale` convention correct
   (spot-check a few raw depth pixel values against a tape-measure distance, not
   just "it renders something"), companion computer reaches the offboard machine's
   `daaam_dev` container over the network, `daaam_node` ingests real frames without
   crashing. **Go/no-go**: 5 minutes of continuous real frames processed with zero
   crashes and depth values within ~5% of measured ground truth at 2-3 known
   distances.
2. **Teleoperated walk-around** (human drives Spot via controller, pipeline only
   observes — no autonomous action). Validate: DAAAM grounding produces real,
   plausible captions on real-world objects (not just Habitat's clean synthetic
   renders — real-world clutter, lighting, motion blur are all genuinely
   untested); memory populates with real position+embedding coverage, matching
   this session's Habitat-side finding that both fields need explicit tracking to
   reach ~100% coverage. **Go/no-go**: a 10-minute walk-around session produces at
   least 10 real, sensible grounded object descriptions with valid embeddings and
   positions logged.
3. **Supervised closed-loop** (frontier_agent drives Spot's move commands, a human
   safety operator holds an emergency stop and can override). Validate: the
   `choose_action` → Spot SDK command path actually moves the robot correctly
   (same turn-direction-sign class of bug this session hit once already for real —
   re-verify empirically against actual Spot movement, don't assume the Habitat
   fix transfers), stopping condition and verification logic behave sensibly on
   real (noisier) depth/pose. **Go/no-go**: 3 consecutive supervised episodes with
   no safety-operator intervention required and at least one real
   `target_visually_verified` success.
4. **Unsupervised**. Only after phase 3's go/no-go is met consistently (not once) —
   and only with Spot's own built-in collision/fall-safety systems (independent of
   this pipeline) still active as the actual safety backstop, not this pipeline's
   own stopping-condition logic.

## 6. Open risks / unknowns (honest, not resolved by this document)

- **Real-world segmentation/depth quality is genuinely untested.** Every real
  result this session has is against Habitat's clean synthetic renders (see
  `data/runs/phase1_smoke/frame0_hm3d.png`) or the CODa dataset's own curated
  bags — never a live, noisy, motion-blurred real-time feed. FastSAM's mask
  quality and DAM-3B's caption quality under real conditions are unverified.
- **Command-rate/latency mismatch, real possibility.** This pipeline's perception
  loop runs ~150-200ms/frame (confirmed via real server logs). Spot SDK command
  latency and the robot's own physical response time haven't been characterized
  against that — if Spot's command pipeline is meaningfully slower, the
  `choose_action` → single discrete action-per-decide() model (built for
  Habitat's instant scripted stepping) may need to become closed-loop/continuous
  instead of one-shot per decision.
- **Battery/thermal budget for continuous companion-computer + payload-sensor
  operation** is unestimated — Spot's own battery life (~90 min typical) already
  bounds session length regardless; whether the RealSense + companion computer
  meaningfully shortens that further needs a real bench measurement, not a guess.
- **`selectframe_clip_backend`/model mismatch, worth resolving before Spot
  bring-up, not during it**: this project's bridge config uses `openclip`/
  `ViT-L-14` `[verified: bridge_pipeline_config.yaml]`, while `daaam_ros`'s own
  CODa launch defaults to `pe`/`PE-Core-L14-336` (Perception Encoder, a different
  backend entirely) `[verified: coda_daaam_hydra.launch.yaml]` — decide
  deliberately which to standardize on for a real deployment rather than
  inheriting whichever one happened to be convenient for the Habitat smoke tests.
