# memory_layer

Phase 3. Implements the persistent-memory scoring equations (reliability
`C_i`, relevance `s(q, d_i)`, direct-recall criterion, per-category
mobility-scaled temporal decay, memory update `M_{t+1} = Update(M_t, O_t, V_t)`)
from `CLAUDE.md` §6. Consumes `daaam_adapter` detections, produces scored
candidates for `frontier_agent`.

## Run tests standalone

```
pytest packages/memory_layer/tests
```

## Contract

Input (per detection, from `daaam_adapter`): `{entity_description, bbox_or_mask, embedding, confidence c_i, room/context_label, timestamp}`

Output (per queried candidate, to `frontier_agent`):

```
{entity_id, relevance, reliability C_i, category, mobility_i, standoff_pose_or_none}
```

plus the fused value map used for contextual anchor search.
