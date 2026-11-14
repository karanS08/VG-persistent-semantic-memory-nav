#!/bin/bash
# 2026-09-07: coverage-vs-step-budget probe on the 33-room, 4-floor held-out
# scene (00337). The existing 500-step held-out result
# (data/runs/phase4_batch_heldout00337_baseline) never gets close to the real
# stopping condition's coverage_threshold=0.95 (configs/run_defaults.yaml):
# every episode that timed out at 500 steps -- including both adversarial
# queries, which are guaranteed to exhaust the budget -- topped out at
# coverage_ratio_at_end 0.40-0.50. This run raises the budget to 2000 steps,
# same scene, same real 10-query list (8 confirmed-present + 2 adversarial,
# mined the same way as the original held-out test -- see
# results/held_out_test_scene00337_query_selection.md), baseline (reset)
# condition only, to see where coverage actually plateaus on this scene
# rather than assuming a step count.
#
# Baseline (not persistent) chosen deliberately: persistent mode would let
# later queries shortcut via remembered anchors from earlier ones, which is
# exactly the thing that would muddy a clean per-episode read of how much of
# the scene genuine fresh exploration covers at this budget.
#
# Usage: nohup bash scripts/run_heldout00337_2000step_probe.sh > logs/heldout00337_2000step_probe_driver.log 2>&1 &

set -u
cd "$(dirname "$0")/.." || exit 1

QUERIES='a framed painting,a lamp,a soft cushion,a cabinet,a curtain,a rug,a pillow,a sofa,a bicycle,a laptop'
SCENE_00337='data/hm3d/example/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb'
CONDA_ENV=rpv_mem

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
source "$(dirname "$0")/_driver_lib.sh"

log "=== Scene 00337 (33 rooms/4 floors), 2000-step coverage probe, baseline condition ==="
run_batch "phase4_batch_heldout00337_2000step_baseline" \
    --max-steps 2000 --queries "$QUERIES" --scene-path "$SCENE_00337" --run-tag heldout00337_2000step_baseline

log "=== 2000-step coverage probe complete ==="
