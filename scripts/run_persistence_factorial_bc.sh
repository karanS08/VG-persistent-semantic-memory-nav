#!/bin/bash
# 2026-09-05: completes the 2x2 persistence factorial (critique.md SS4/SS32).
# Conditions A (reset5) and D (continue5) already exist with 3 repeats each
# (reset5/reset5_rep2/reset5_rep3, continue5/continue5_rep2/continue5_rep3).
# This script adds B (--persist-memory-only) and C (--persist-position-only),
# 3 repeats each, same 5-query set, for a real, decoupled comparison --
# the critique's own "key comparison" is B vs A (isolates memory persistence
# cleanly, without the position-carryover confound A-vs-D always had).
#
# Queues behind whatever's currently using the bridge (this session's
# priority-fixes driver, anchor-weight ablation) via the same
# wait_for_bridge_idle pattern -- no manual coordination needed.
#
# Usage: nohup bash scripts/run_persistence_factorial_bc.sh > logs/persistence_factorial_bc_driver.log 2>&1 &
#        disown

set -u
cd "$(dirname "$0")/.." || exit 1

QUERIES5='a large window,a wall-mounted radiator,a cabinet,a wall vent,a bench'
BRIDGE_HEALTH_URL="http://localhost:8765/health"
CONDA_ENV=rpv_mem

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# 2026-09-05: switched to the shared flock-based run_batch after this
# script's own unlocked run_batch raced with a concurrent driver and lost
# rep2 and rep3 (both 409 at start_episode). run_batch skips any rep whose
# batch_metrics.json exists, so restarting is safe and only retries the
# two real failures.
source "$(dirname "$0")/_driver_lib.sh"

log "=== Persistence factorial: condition B (persist-memory-only), 3 reps ==="
for rep in 1 2 3; do
    run_batch "phase4_batch_persist_memory_only_persistmemoryonly5_rep${rep}" \
        --max-steps 500 --queries "$QUERIES5" --persist-memory-only --run-tag "persistmemoryonly5_rep${rep}"
done

log "=== Persistence factorial: condition C (persist-position-only), 3 reps ==="
for rep in 1 2 3; do
    run_batch "phase4_batch_persist_position_only_persistpositiononly5_rep${rep}" \
        --max-steps 500 --queries "$QUERIES5" --persist-position-only --run-tag "persistpositiononly5_rep${rep}"
done

log "=== persistence factorial B/C driver complete ==="
