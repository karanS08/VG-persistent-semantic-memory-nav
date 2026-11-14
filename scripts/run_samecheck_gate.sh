#!/bin/bash
# 2026-09-29: same-object gate for the setup-figure comparison. Runs memory-on /
# memory-off pairs of the same session (unique objects near the start) and stops at
# the first attempt where check_same_object.py finds a search (after the shared first
# one) whose verified object coordinates agree within 1.5 m in both conditions.
# Attempt 1 reuses the already-started samecheck_on/off runs; up to 3 attempts.
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rpv_mem
source scripts/_driver_lib.sh
unset RPV_NO_RRD
Q='a large window,a suitcase,a teddy bear,a trash can,an oven'
while pgrep -f "run_samecheck.sh" >/dev/null; do sleep 30; done   # let attempt 1's driver finish
for n in 1 2 3; do
    sfx=$([ $n -eq 1 ] && echo "" || echo "_rep$n")
    on=phase4_batch_persist_memory_only_samecheck_on$sfx; off=phase4_batch_samecheck_off$sfx
    run_batch $on --max-steps 500 --min-confirming-observations 1 --queries "$Q" --persist-memory-only --run-tag samecheck_on$sfx
    run_batch $off --max-steps 500 --min-confirming-observations 1 --queries "$Q" --run-tag samecheck_off$sfx
    if python scripts/check_same_object.py data/runs/$on data/runs/$off; then
        log "SAMECHECK COMPLETE attempt $n: $on vs $off"; exit 0
    fi
    log "attempt $n not complete, re-running"
done
log "SAMECHECK GATE FAILED after 3 attempts"
