#!/bin/bash
# 2026-09-29 spread-object demo, stage 2: five rare objects at the user's five
# spots (NE shield, north-annex tripod, NW cardboard box, SW whiteboard, SE
# fireplace; counts in tour memory 1/2/3/4/5). Memory ON starts from the house-tour
# memory (condition B, position resets to spawn each query); memory OFF = reset.
# ON runs first so the tour memory is as fresh as possible (ages in wall-clock time).
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rpv_mem
source scripts/_driver_lib.sh
unset RPV_NO_RRD   # recordings needed for the path plots
Q='a shield,a tripod,a cardboard box,a whiteboard,a fireplace'
run_batch phase4_batch_persist_memory_only_spread_on --max-steps 500 --min-confirming-observations 1 \
    --queries "$Q" --persist-memory-only \
    --init-memory-snapshot data/runs/phase4_batch_persist_memory_only_house_tour/memory_snapshot_ep0.json --run-tag spread_on
run_batch phase4_batch_spread_off --max-steps 500 --min-confirming-observations 1 --queries "$Q" --run-tag spread_off
log "SPREAD DEMO DONE"
