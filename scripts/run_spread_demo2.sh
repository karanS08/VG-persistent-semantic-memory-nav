#!/bin/bash
# 2026-09-29 spread-object demo, stage 2 (v2, user chose option 2): queries SCREENED
# so that the house-tour memory's CLIP (query text vs object image) top match lies at
# each of the user's five spots -- NE black vase, north-annex faucet, NW sink, SW
# whiteboard, SE painting of the sea. The first query set (shield/tripod/...) failed
# because relevance over 1,193 remembered objects is too flat; kept as a limitation
# (phase4_batch_persist_memory_only_spread_on). Figures must say queries were screened.
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rpv_mem
source scripts/_driver_lib.sh
unset RPV_NO_RRD   # recordings needed for the path plots
Q='a black vase,a faucet,a sink,a whiteboard,a painting of the sea'
run_batch phase4_batch_persist_memory_only_spread2_on --max-steps 500 --min-confirming-observations 1 \
    --queries "$Q" --persist-memory-only \
    --init-memory-snapshot data/runs/phase4_batch_persist_memory_only_house_tour/memory_snapshot_ep0.json --run-tag spread2_on
run_batch phase4_batch_spread2_off --max-steps 500 --min-confirming-observations 1 --queries "$Q" --run-tag spread2_off
log "SPREAD2 DEMO DONE"
