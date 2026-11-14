#!/bin/bash
# 2026-10-03: memory-off half of the instance-gated pair, re-run after moving the gate
# before verification (first attempt kept in data/runs/_superseded_instance_off_postgate).
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rpv_mem
source scripts/_driver_lib.sh
unset RPV_NO_RRD
Q='a large window,a wall-mounted radiator'
T='{"query": "a wall-mounted radiator", "xyz": [-9.379, 1.121, 6.804], "radius": 1.5}'
run_batch phase4_batch_instance_off --max-steps 1500 --min-confirming-observations 1 \
    --queries "$Q" --instance-target "$T" --run-tag instance_off
python scripts/check_same_object.py data/runs/phase4_batch_persist_memory_only_instance_on data/runs/phase4_batch_instance_off
log "INSTANCE OFF RETRY DONE"
