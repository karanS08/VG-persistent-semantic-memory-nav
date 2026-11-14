#!/bin/bash
# 2026-09-29: one memory-only 5-query session on 00770 with the per-episode
# memory snapshot logging, recordings ON -- source data for the teaser figure.
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rpv_mem
source scripts/_driver_lib.sh
run_batch phase4_batch_persist_memory_only_teaser_memonly5 --max-steps 500 --min-confirming-observations 1 \
    --queries 'a large window,a wall-mounted radiator,a cabinet,a wall vent,a bench' --persist-memory-only --run-tag teaser_memonly5
log "TEASER SESSION DONE"
