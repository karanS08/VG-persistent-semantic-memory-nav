#!/bin/bash
# 2026-09-29 spread infographic: real memory-OFF searches for the five spread objects
# (NE carved fireplace, north-annex tripod, NW mirror, SW radiator, SE tapestry: objects
# visible from each spot's room view). Memory-ON routes in that infographic are planned navmesh paths (labelled illustrative),
# because relevance x reliability over the 1,193-object tour memory does not pick these
# objects (EXPERIMENT_RESULTS.md Exp 11).
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rpv_mem
source scripts/_driver_lib.sh
unset RPV_NO_RRD
run_batch phase4_batch_spread3_off --max-steps 500 --min-confirming-observations 1 \
    --queries 'a carved fireplace,a tripod,a mirror,a radiator,a tapestry' --run-tag spread3_off
log "SPREAD OFF DONE"
