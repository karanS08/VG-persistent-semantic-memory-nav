# Recording and presenting a demo episode

This walks through producing a self-contained Rerun recording of a real closed-loop
episode — real HM3D scene, real DAAAM grounding, real memory, real navigation
decisions — and how to describe what's in it.

## Recording a fresh episode

1. **Check the DAAAM bridge is healthy and the GPU is clear.** Repeated restarts
   during development have previously left orphaned GPU processes (see
   `PROGRESS.md` Phase 2/4) — always check before a run you care about:
   ```bash
   curl -s http://localhost:8765/health
   nvidia-smi --query-gpu=memory.used --format=csv
   ```

2. **(Optional) Set your demo query.** Edit `QUERIES` near the top of
   `packages/orchestrator/orchestrator/run_phase4_batch.py`. Episode 0 (the
   first query in the list) is the one that gets recorded.

3. **Run it**, from the repo root, in the `rpv_mem` conda environment:
   ```bash
   conda activate rpv_mem
   python -m orchestrator.run_phase4_batch --max-steps 200
   ```
   Takes roughly 8-12 minutes (200 steps + a 30s grounding-wait per episode).
   Worth backgrounding with `nohup ... &` for a longer run.

4. **The recording lands at** `data/runs/phase4_batch/episode_0.rrd` (only
   episode 0 gets one — `log_rerun=(episode_id == 0)` in the script).

5. **Verify it's real before trusting it for a demo** (headless, no GUI needed):
   ```bash
   rerun rrd print data/runs/phase4_batch/episode_0.rrd | grep "^Chunk" \
     | sed -E 's/.*- (\/[^ ]+) - .*/\1/' | sort | uniq -c
   ```
   Expect real counts under `/world/agent`, `/world/agent/rgb`,
   `/world/agent/depth`, `/world/daaam/objects`, and
   `/world/daaam/live_captions/<id>` per detected object.

6. **Serve it for viewing:**
   ```bash
   rerun data/runs/phase4_batch/episode_0.rrd --web-viewer \
     --web-viewer-port 9090 --ws-server-port 9877 --bind 0.0.0.0 &
   ```
   Then open it with the WebSocket target **explicitly** included in the URL —
   the default page bakes in `ws://localhost:9877`, which resolves to the
   *viewer's own machine* when connecting remotely (e.g. over Tailscale), not
   the server. Use:
   ```
   http://<host>:9090?url=ws://<host>:9877
   ```

## What's actually in the recording

| Entity path | What it is |
|---|---|
| `world/agent` | The agent's pose (position + rotation) over time — its trajectory. |
| `world/agent/rgb` | The agent's RGB camera view, frame by frame — literally what it saw. |
| `world/agent/depth` | The corresponding depth map. |
| `world/agent/position` | Trajectory points. |
| `world/daaam/objects` | Every object DAAAM has grounded so far, at its real 3D position, labeled with its real caption. Updates over time as memory grows. |
| `world/daaam/live_captions/<id>` | A text-log entry the moment each object's caption arrives, so you can scrub to exactly when a new detection landed. |

Nothing here is placeholder or pre-scripted: the RGB/depth stream is the actual
Habitat-Sim render, the captions are real DAM-3B (a 3B-parameter vision-language
model) output generated during the run, and the 3D positions come from DAAAM's
own depth-based back-projection — not hand-placed.

## How to describe it to an audience

Scrub through the timeline (bottom of the Rerun viewer) and narrate roughly in
this order:

1. **The view.** "This is literally what the robot's camera saw, frame by
   frame, as it explored the space." — point at `world/agent/rgb`.
2. **The path.** "This trace is everywhere it walked. It's not scripted — it's
   making a navigation decision every single step based on what it currently
   knows." — point at the trajectory under `world/agent`.
3. **The detections.** "Each of these labels is a real caption from a
   3-billion-parameter vision-language model, generated live during
   exploration — not pre-written, not a lookup table." — click through a few
   `world/daaam/objects` points, read the captions aloud.
4. **The outcome.** Pair the visual with the actual run output (printed to
   console, or `data/runs/phase4_batch/batch_metrics.json`): the query, the
   step count, `outcome: "success"` / `reason: "target_visually_verified"`, and
   the SPL value — "Success weighted by Path Length: how efficient its path
   was relative to the shortest possible route to where it ended up. 1.0 would
   be a perfectly direct path; ours here is real but not yet optimized."

If asked "does the memory actually help it decide where to go" — the honest,
current answer (see `PROGRESS.md`'s Phase 4 "does memory help" investigation)
is: not yet demonstrably, in the runs so far. A per-step instrumented probe
(score every navigation decision with and without remembered signals, check
whether the choice changes) shows memory isn't yet the thing driving these
particular successes — a real, precisely-diagnosed timing issue (grounding
results arrive in a late burst relative to when the agent bothers to check
memory), not a flaw in the underlying scoring math, which is separately
verified correct against real data in Phase 3. This is presented as the
current state of an active investigation, not a hidden gap — the point of
building the probe was to be able to answer this question honestly rather than
assume the answer.
