#!/bin/bash
# 2026-09-29: paired memory-on/off session for the setup figure. Targets exist ONCE in
# 00770 (per the house-tour memory) so both conditions should verify the same object;
# figure uses only a pair whose verified coordinates agree within 1.5 m.
# Q1 "a large window" = shared exploring first search (memory empty in both).
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rpv_mem
source scripts/_driver_lib.sh
unset RPV_NO_RRD
Q='a large window,a suitcase,a teddy bear,a trash can,an oven'
run_batch phase4_batch_persist_memory_only_samecheck_on --max-steps 500 --min-confirming-observations 1 \
    --queries "$Q" --persist-memory-only --run-tag samecheck_on
run_batch phase4_batch_samecheck_off --max-steps 500 --min-confirming-observations 1 --queries "$Q" --run-tag samecheck_off
log "SAMECHECK DONE"
