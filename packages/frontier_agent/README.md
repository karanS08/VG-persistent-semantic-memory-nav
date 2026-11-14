# frontier_agent

Phase 4. Fuses live detections with `memory_layer`'s weighted remembered
anchors into a value map, scores frontiers, selects the next action, and
evaluates the continuous stopping condition (success / budget-failure) per
`CLAUDE.md` §6.

## Run tests standalone

```
pytest packages/frontier_agent/tests
```

## Contract

Input (per candidate, from `memory_layer`): `{entity_id, relevance, reliability C_i, category, mobility_i, standoff_pose_or_none}` + fused value map

Output (to `sim_bridge`): next action in Habitat's action space (`move_forward` / `turn_left` / `turn_right` / `stop` / `look_at`, etc.)
