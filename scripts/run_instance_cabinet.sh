#!/bin/bash
# 2026-10-03: instance-gated memory-on/off pair for the teaser, cabinet version.
# Target = the cabinet verified in phase4_batch_persist_memory_only_teaser_memonly5
# (query 3) at (-11.861, 1.142, 6.619); radius 1.5 m. Same 3-query lead-up as the
# earlier sessions (window, radiator, cabinet); only the cabinet query is gated.
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rpv_mem
source scripts/_driver_lib.sh
unset RPV_NO_RRD
Q='a large window,a wall-mounted radiator,a cabinet'
T='{"query": "a cabinet", "xyz": [-11.861, 1.142, 6.619], "radius": 1.5}'
run_batch phase4_batch_persist_memory_only_instance_cab_on --max-steps 500 --min-confirming-observations 1 \
    --queries "$Q" --persist-memory-only --instance-target "$T" --run-tag instance_cab_on
run_batch phase4_batch_instance_cab_off --max-steps 1500 --min-confirming-observations 1 \
    --queries "$Q" --instance-target "$T" --run-tag instance_cab_off
python scripts/check_same_object.py data/runs/phase4_batch_persist_memory_only_instance_cab_on data/runs/phase4_batch_instance_cab_off
log "INSTANCE CABINET DONE"
