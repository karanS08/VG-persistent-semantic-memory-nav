#!/bin/bash
# 2026-09-06: genuine held-out test-set evaluation (critique.md SS9 / critique_2
# SS9/SS42's highest-priority undone item). Scene 00337's query set was selected
# BEFORE this run, from real mined captions, with zero threshold re-tuning --
# see results/held_out_test_scene00337_query_selection.md. Uses the exact same
# production defaults every scene-00770 result in this paper uses
# (theta_direct=0.2, min_confirming_observations=1, anchor_weight=1.0), frozen
# from scene-00770-only development well before this file was written.
#
# Usage: nohup bash scripts/run_heldout_test_scene00337.sh > logs/heldout_test_00337_driver.log 2>&1 &

set -u
cd "$(dirname "$0")/.." || exit 1

QUERIES='a framed painting,a lamp,a soft cushion,a cabinet,a curtain,a rug,a pillow,a sofa,a bicycle,a laptop'
SCENE_00337='data/hm3d/example/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb'
CONDA_ENV=rpv_mem

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
source "$(dirname "$0")/_driver_lib.sh"

log "=== Held-out test, scene 00337: baseline (reset) condition ==="
run_batch "phase4_batch_heldout00337_baseline" \
    --max-steps 500 --queries "$QUERIES" --scene-path "$SCENE_00337" --run-tag heldout00337_baseline

# 2026-09-06: run_phase4_batch.py prepends "persistent" to its real output
# dir whenever --persistent-memory is set (ablation_suffixes.append
# ("persistent") happens before the run-tag append) -- the real directory is
# phase4_batch_persistent_heldout00337_persistent, not
# phase4_batch_heldout00337_persistent. This run already completed
# successfully before the mismatch was caught (0.80 success, 0.5348 SPL,
# real data intact) and got a false FAILED; fixed here so any future rerun
# of this script correctly recognizes it as done.
log "=== Held-out test, scene 00337: persistent-across-queries condition ==="
run_batch "phase4_batch_persistent_heldout00337_persistent" \
    --max-steps 500 --queries "$QUERIES" --scene-path "$SCENE_00337" --persistent-memory --run-tag heldout00337_persistent

log "=== held-out test scene00337 driver complete ==="
