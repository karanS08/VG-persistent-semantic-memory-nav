#!/bin/bash
# 2026-09-06: DAAAM latency injection experiment (critique.md/critique_2
# "asynchronous DAAAM latency needs experimental validation"). Sweeps
# additional synthetic latency on top of DAAAM's real grounding delay, per
# the critique's own suggested values, on the standard 8-query battery,
# non-persistent condition (the paper's primary reported condition), one
# rep per value -- a controlled-manipulation sweep, not a variance study.
#
# Usage: nohup bash scripts/run_latency_injection_sweep.sh > logs/latency_injection_driver.log 2>&1 &

set -u
cd "$(dirname "$0")/.." || exit 1

QUERIES8='a soft cushion,a large window,a wooden chair,a wall-mounted mirror,a dining table,a wall-mounted radiator,a doorway,a framed painting'
CONDA_ENV=rpv_mem

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
source "$(dirname "$0")/_driver_lib.sh"

log "=== DAAAM latency injection sweep (baseline/non-persistent condition) ==="
for lat in 0 2 5 10 20 30; do
    run_batch "phase4_batch_latency${lat}s" \
        --max-steps 500 --queries "$QUERIES8" --inject-latency-s "$lat" --run-tag "latency${lat}s"
done

log "=== latency injection sweep driver complete ==="
