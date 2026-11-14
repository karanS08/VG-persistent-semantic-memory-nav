#!/bin/bash
# 2026-10-03: real memory-on / memory-off pair for the teaser, with the demo-only
# instance gate: "a wall-mounted radiator" succeeds only at the radiator near the start
# (target = position verified in phase4_batch_persist_memory_only_teaser_memonly5, query 2),
# so both conditions must end at the SAME radiator. Query 1 ("a large window") is the
# shared exploring search; memory-on keeps memory into query 2, memory-off resets.
# Memory-off gets a 1500-step budget (demo only) so it can reach that radiator.
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rpv_mem
source scripts/_driver_lib.sh
unset RPV_NO_RRD
Q='a large window,a wall-mounted radiator'
T='{"query": "a wall-mounted radiator", "xyz": [-9.379, 1.121, 6.804], "radius": 1.5}'
run_batch phase4_batch_persist_memory_only_instance_on --max-steps 500 --min-confirming-observations 1 \
    --queries "$Q" --persist-memory-only --instance-target "$T" --run-tag instance_on
run_batch phase4_batch_instance_off --max-steps 1500 --min-confirming-observations 1 \
    --queries "$Q" --instance-target "$T" --run-tag instance_off
python scripts/check_same_object.py data/runs/phase4_batch_persist_memory_only_instance_on data/runs/phase4_batch_instance_off
log "INSTANCE PAIR DONE"
