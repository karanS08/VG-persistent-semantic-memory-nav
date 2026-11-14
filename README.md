# Verification-Gated Persistent Semantic Memory for Embodied Object-Search

Code for the ACRA 2026 paper by **Karan Sharma, Adam Scicluna and Gavin Paul**
(Robotics Institute, University of Technology Sydney).

**Project page:** https://claude.ai/artifact/V4MsRbsnzL6bpLeKhVZpcT

> Memory proposes. Perception verifies. New evidence updates the memory.

A robot searching for one object sees many others along the way. Task-specific
navigation stacks usually throw that information away. This repository keeps it
as a persistent semantic memory, a record of what was seen, where and when, and
uses it in later searches. Each memory gets a reliability score, and a search
only counts as a success after a fresh observation confirms the object.

The memory layer sits on top of [DAAAM](https://github.com/MIT-SPARK/DAAAM)
(open-vocabulary segmentation, tracking and captioning) and a frontier-based
explorer in Habitat-Sim. Neither of those is modified.

## Results in brief

- **Memory between searches:** in a memory × session factorial on HM3D scene
  00770, keeping memory between queries cut steps per search from 348 to 46
  and raised strict-audited SPL from 0.10 to 0.77.
- **Verification:** with all four verification checks removed, reported success
  rises to 1.0 while audited success falls to 0/10.
- **Moved objects:** in ProcTHOR houses where a remembered sofa is moved,
  requiring fresh perceptual evidence reduces searches ending at the old location
  from 11 of 12 episodes to 0 of 12.
- **Influence:** memory pulled the robot away from a correct live target in 0.8%
  of scored decisions (5 of 636).

## How it works

```
Habitat-Sim / ProcTHOR      RGB-D + pose
        │
        ▼
sim_bridge ──frames──▶ daaam_adapter ──captions, CLIP embeddings, 3D positions──▶ memory_layer
        ▲                  (HTTP bridge to DAAAM)                                   │
        │                                                                           │ candidates:
        └──────────── action ◀──── frontier_agent ◀── relevance × reliability ──────┘
                                   (direct recall / contextual anchor / exploration)
```

Each search runs four stages: **see** (DAAAM captions the object and an entry is
created or reinforced), **lose sight** (the entry is kept and starts to decay),
**recall** (memory proposes locations weighted by relevance × reliability), and
**verify** (the robot re-observes the spot; four checks must pass). A failed check
discounts the entry instead of deleting it.

### Reliability

```
C_i = v_i · clip[0,1]( c_i · (1 − exp(−α n_i)) ) · exp(−λ_i Δt_i),    λ_i = λ_base · mobility(category_i)
```

| Term | Meaning |
|---|---|
| `c_i` | quality of the original observation |
| `n_i` | number of confirming observations (`α = 0.5`) |
| `Δt_i` | time since the object was last seen |
| `v_i` | verification multiplier, discounted on each failed check |
| `λ_i` | per-category decay; `λ_base = 1e-3 s⁻¹` scaled by an offline mobility prior |

The mobility priors for 118 categories (a microwave at 0.03, a phone at 0.85) are in
`configs/mobility_priors.json`. Relevance `s(q, d_i)` is the CLIP similarity
between the query text and the stored entry. Direct recall requires
`s · C_i ≥ θ_direct` **and** a reachable standoff pose with line of sight; these
two conditions are logged separately.

## Repository layout

```
packages/
  sim_bridge/        Habitat-Sim wrapper (EGL headless), ProcTHOR drop-in, Rerun streaming
  daaam_adapter/     HTTP client + the bridge server that runs inside DAAAM's container
  memory_layer/      reliability, relevance, recall, memory store and update
  frontier_agent/    value map, frontier candidates, standoff poses, verification, actions
  orchestrator/      episode runners and batch experiments
configs/             habitat.yaml, memory.yaml, mobility_priors.json, run_defaults.yaml, house-tour waypoints
scripts/             run_phase1.sh; thor/ runs the same pipeline in AI2-THOR / ProcTHOR houses
docs/                demo recording, real-time notes, Spot deployment plan
logs/failures.md     failure-triage log
```

Each package has its own `README.md` describing its input/output contract and
how to test it on its own.

## Setup

The pipeline runs two processes because DAAAM and Habitat-Sim need incompatible
Python versions.

1. **Simulator side:** a conda environment with Python 3.9. It was tested with
   `habitat-sim 0.3.3` (headless, EGL), `habitat-lab 0.3.3`, `torch 2.8.0`,
   `open_clip_torch 3.3.0`, `rerun-sdk 0.19.1` and `numpy 1.26.4`.
   Then install the packages:
   ```bash
   for p in sim_bridge daaam_adapter memory_layer frontier_agent orchestrator; do
     pip install -e packages/$p
   done
   ```
2. **DAAAM side:** install [MIT-SPARK/DAAAM](https://github.com/MIT-SPARK/DAAAM)
   in its own container and start the bridge server from
   `packages/daaam_adapter` inside it (see that package's README).
3. **Scenes:** download [HM3D](https://aihabitat.org/datasets/hm3d/) under its
   licence into `data/hm3d/`. Scene data is not included here.

Tested on Ubuntu with one RTX 3090 (24 GB).

## Running

```bash
# Unit tests (no simulator or DAAAM needed)
pytest packages/memory_layer/tests packages/frontier_agent/tests

# Simulator + Rerun only, no DAAAM
bash scripts/run_phase1.sh

# Closed-loop batch with DAAAM grounding and memory
python -m orchestrator.run_phase4_batch --max-steps 200

# Keep memory across queries in one session
python -m orchestrator.run_phase4_batch --persistent-memory

# Ablations
python -m orchestrator.run_phase4_batch --memory-off
python -m orchestrator.run_phase4_batch --no-mobility-scaling
python -m orchestrator.run_phase4_batch --no-verification-discount
python -m orchestrator.run_phase4_batch --no-fresh-evidence-guard

# Same pipeline in a ProcTHOR house (needs ai2thor and prior installed)
THOR_HOUSE=1 THOR_GT_DIR=data/runs/thor_demo python scripts/thor/run_thor_batch.py --max-steps 200
```

Every run writes its resolved config, per-step JSONL logs and a Rerun recording
(`.rrd`) to `data/runs/<run>/`. Defaults live in `configs/run_defaults.yaml`; run
`python -m orchestrator.run_phase4_batch --help` for all flags.

## Citation

The proceedings are not out yet. Citation details will be added once the paper
is published.

## Acknowledgements

Built on [DAAAM](https://github.com/MIT-SPARK/DAAAM),
[Habitat-Sim](https://aihabitat.org/), [HM3D](https://aihabitat.org/datasets/hm3d/),
[AI2-THOR / ProcTHOR](https://procthor.allenai.org/) and [Rerun](https://rerun.io/).
