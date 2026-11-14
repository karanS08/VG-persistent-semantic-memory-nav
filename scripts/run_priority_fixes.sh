#!/bin/bash
# 2026-09-05: addresses the 3 highest-leverage gaps identified in a quality
# review of the paper's current state, in priority order, fully detached and
# resumable (skips any rep whose batch_metrics.json already exists):
#
#  1. Reset-vs-continue (Table 5, the paper's single strongest/most-quoted
#     result -- 27% faster, 3x SPL) has only ever been run once per condition,
#     unlike the primary ablation battery which was explicitly re-run for
#     variance. Adds 2 more repeats each (rep1 = the original reset5/continue5
#     runs, this script starts at rep2).
#  2. Multi-anchor fusion (w=1, sigma=6m) has been live in every batch run
#     with zero independent ablation against w=0 (pre-fusion solo-argmax).
#     Runs 3 fresh repeats each on the standard retuned-default 8-query
#     battery (the same set Table 2 uses), now that --anchor-weight exists.
#  3. Single-scene evaluation: this script only runs the MINING episode on
#     scene 00337 (33 rooms, 4 floors, verified connected navmesh -- see
#     PROGRESS.md 2026-09-04) to harvest real captions. Selecting genuine,
#     verified queries from that corpus (same rigor as the 00770 query-mining
#     work) needs human judgment on the actual caption text, so the real
#     00337 evaluation batch is a manual follow-up once this mining episode
#     completes -- not scripted here.
#
# Usage: nohup bash scripts/run_priority_fixes.sh > logs/priority_fixes_driver.log 2>&1 &
#        disown

set -u
cd "$(dirname "$0")/.." || exit 1

QUERIES5='a large window,a wall-mounted radiator,a cabinet,a wall vent,a bench'
QUERIES8='a soft cushion,a large window,a wooden chair,a wall-mounted mirror,a dining table,a wall-mounted radiator,a doorway,a framed painting'
SCENE_00337='data/hm3d/example/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb'
BRIDGE_HEALTH_URL="http://localhost:8765/health"
CONDA_ENV=rpv_mem

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# 2026-09-05: switched to the shared flock-based run_batch (see PROGRESS.md
# "driver race condition found and fixed") after this script's own
# not-yet-locked run_batch raced with a concurrently-running driver.
# run_batch already skips any rep whose batch_metrics.json exists, so
# restarting this script mid-flight is safe and redoes no completed work.
source "$(dirname "$0")/_driver_lib.sh"

log "=== Priority 1: reset-vs-continue repeats (rep2, rep3; rep1 = existing reset5/continue5) ==="
for rep in 2 3; do
    run_batch "phase4_batch_reset5_rep${rep}" \
        --max-steps 500 --queries "$QUERIES5" --run-tag "reset5_rep${rep}"
    run_batch "phase4_batch_persistent_continue5_rep${rep}" \
        --max-steps 500 --queries "$QUERIES5" --persistent-memory --run-tag "continue5_rep${rep}"
done

log "=== Priority 3: multi-anchor-fusion ablation, w=0 vs w=1, 3 reps each ==="
for rep in 1 2 3; do
    run_batch "phase4_batch_anchorw0.0_anchorw0_rep${rep}" \
        --max-steps 500 --queries "$QUERIES8" --anchor-weight 0.0 --run-tag "anchorw0_rep${rep}"
    run_batch "phase4_batch_anchorw1.0_anchorw1_rep${rep}" \
        --max-steps 500 --queries "$QUERIES8" --anchor-weight 1.0 --run-tag "anchorw1_rep${rep}"
done

log "=== Priority 2 (partial): scene 00337 mining episode ==="
run_batch "phase4_batch_scene00337mining" \
    --max-steps 1000 --queries "a chair" --scene-path "$SCENE_00337" --run-tag "scene00337mining"

log "=== priority fixes driver complete ==="
