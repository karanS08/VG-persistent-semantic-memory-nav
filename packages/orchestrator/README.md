# orchestrator

Ties `sim_bridge`, `daaam_adapter`, `memory_layer`, and `frontier_agent`
together into the closed loop, and hosts the top-level `run_phase{1,2,3,4}.sh`
entrypoints (see `../../scripts/`). Not independently testable in isolation —
its correctness is the end-to-end phase verification gates in `CLAUDE.md` §9.

## Contract

Wires the per-package contracts documented in each package's own `README.md`
into one loop: `sim_bridge -> daaam_adapter -> memory_layer -> frontier_agent -> sim_bridge`.
