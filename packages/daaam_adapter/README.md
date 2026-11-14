# daaam_adapter

Phase 2. Feeds `sim_bridge`'s per-step RGB/depth/pose frames into DAAAM
(`MIT-SPARK/DAAAM`, pinned commit — see `configs/daaam.yaml`) and normalizes
its captioning/scene-graph output into the detection contract consumed by
`memory_layer`.

## Run tests standalone

```
pytest packages/daaam_adapter/tests
```

## Contract

Input (per step, from `sim_bridge`): `{rgb, depth, pose, timestamp, episode_id, step_id}`

Output (per detection, to `memory_layer`):

```
{entity_description, bbox_or_mask, embedding, confidence c_i, room/context_label, timestamp}
```
