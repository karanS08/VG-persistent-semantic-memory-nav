#!/bin/bash
# 2026-09-29 spread-object demo, stage 1: scripted tour of every room of 00770
# (configs/tours/00770_full_tour.json, 16 waypoints, 360-degree look at each),
# query = absent object so the tour is never cut short by a verification.
# Saves memory_snapshot_ep0.json (with embeddings) for stage 2.
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rpv_mem
source scripts/_driver_lib.sh
run_batch phase4_batch_persist_memory_only_house_tour --max-steps 500 --tour-max-steps 3000 --min-confirming-observations 1 \
    --queries 'a bicycle' --persist-memory-only --tour-waypoints configs/tours/00770_full_tour.json --run-tag house_tour
log "HOUSE TOUR DONE"
