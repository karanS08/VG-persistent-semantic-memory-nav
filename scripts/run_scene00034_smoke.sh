#!/bin/bash
# 2026-09-29: first closed-loop run on HM3D train scene 00034-6imZUJGRUq4 (user
# request: run the sim once). One query, default settings, recording on.
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rpv_mem
source scripts/_driver_lib.sh
unset RPV_NO_RRD
run_batch phase4_batch_scene00034_smoke --max-steps 500 --min-confirming-observations 1 --queries 'a bicycle' \
    --scene-path data/hm3d_train/versioned_data/hm3d-0.2/hm3d/train/00034-6imZUJGRUq4/6imZUJGRUq4.basis.glb --run-tag scene00034_smoke
log "SCENE00034 SMOKE DONE"
