#!/bin/bash
# 2026-09-07: memory-age injection sweep -- the direct, live-pipeline test of
# whether temporal decay (Eq. 1) actually changes real outcomes at timescales
# (hours) a single ~250s episode could never reach on its own. Uses the new
# --inject-memory-age-s flag (jumps every remembered entity's clock backward
# by N seconds at each query boundary in --persistent-memory mode, simulating
# "N seconds/hours passed since the previous query"). Standard 8-query
# battery, persistent condition (the only condition where memory carries
# across query boundaries, so the only one this flag can affect), one rep
# per value -- a controlled-manipulation sweep, matching the same convention
# the latency-injection sweep used.
#
# NOTE on real output directory names: --persistent-memory prepends
# "persistent" to the ablation-suffix list BEFORE --run-tag (same real
# behavior documented in run_heldout_test_scene00337.sh's own note) -- so the
# real directory for run-tag "memage1h" is
# data/runs/phase4_batch_persistent_memage1h, not phase4_batch_memage1h.
# Getting this right here the first time, from that earlier lesson.
#
# Usage: nohup bash scripts/run_memory_age_injection_sweep.sh > logs/memory_age_injection_driver.log 2>&1 &

set -u
cd "$(dirname "$0")/.." || exit 1

QUERIES8='a soft cushion,a large window,a wooden chair,a wall-mounted mirror,a dining table,a wall-mounted radiator,a doorway,a framed painting'
CONDA_ENV=rpv_mem

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
source "$(dirname "$0")/_driver_lib.sh"

log "=== Memory-age injection sweep (persistent-across-queries condition) ==="
# tag:seconds pairs -- 0, 1h, 3h, 6h, 24h
for pair in "memage0s:0" "memage1h:3600" "memage3h:10800" "memage6h:21600" "memage24h:86400"; do
    tag="${pair%%:*}"
    secs="${pair##*:}"
    run_batch "phase4_batch_persistent_${tag}" \
        --max-steps 500 --queries "$QUERIES8" --persistent-memory \
        --inject-memory-age-s "$secs" --run-tag "$tag"
done

log "=== memory-age injection sweep driver complete ==="
