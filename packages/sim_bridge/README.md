# sim_bridge

Phase 1. Drives a Habitat-Sim (HM3D) episode headless (EGL), steps an agent,
and streams RGB/depth/pose to Rerun for remote viewing, plus emits per-step
frames downstream to `daaam_adapter`.

## Run tests standalone

```
pytest packages/sim_bridge/tests
```

## Contract

Output (per step, to `daaam_adapter`):

```
{rgb, depth, pose, timestamp, episode_id, step_id}
```

No upstream input in Phase 1 (agent is scripted/random-walk); in Phase 4 this
becomes the sink for `frontier_agent`'s chosen action.
