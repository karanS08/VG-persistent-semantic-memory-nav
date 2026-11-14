#!/bin/bash
# 2026-09-05: repeated full-scene 30-query stress tests on 00770.
# User wants multiple independent large-scale runs (not just one), fully
# detached, willing to wait hours -- same pattern as
# run_coverage_memory_megabatch.sh: sequential (only one episode can run
# against the bridge at a time), resumable (skips any rep whose
# batch_metrics.json already exists), fully independent of any
# interactive session.
#
# Usage: nohup bash scripts/run_scale30_repeats.sh [N_REPS] > logs/scale30_driver.log 2>&1 &
#        disown
# Default N_REPS=10. rep1 was already launched interactively as tag
# "scale30" (no suffix) before this driver existed -- this script starts
# from rep2 to avoid duplicating that work, and its own skip-logic means
# re-running with a higher N_REPS later just continues from wherever it
# left off.

set -u
cd "$(dirname "$0")/.." || exit 1

N_REPS="${1:-10}"
QUERIES='a soft cushion,a large window,a wooden chair,a wall-mounted mirror,a dining table,a wall-mounted radiator,a doorway,a framed painting,a cabinet,a trash bin,a wall vent,a curtain,a fireplace,a lamp,a wall clock,a bench,a chandelier,a vase,a bathtub,a towel,a thermostat,a shelf,a sculpture,a candle,a ceiling light fixture,a drawer,a pillar,a window sill,a rug,a ceiling fan'
BRIDGE_HEALTH_URL="http://localhost:8765/health"
CONDA_ENV=rpv_mem

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_for_bridge_idle() {
    while true; do
        running=$(curl -s "$BRIDGE_HEALTH_URL" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('episode_running', True))" 2>/dev/null)
        if [ "$running" = "False" ]; then
            return 0
        fi
        log "bridge busy or unreachable (episode_running=$running), waiting 30s..."
        sleep 30
    done
}

run_one() {
    local rep="$1"
    local tag="scale30_rep${rep}"
    local out_dir="data/runs/phase4_batch_persistent_${tag}"
    if [ -f "${out_dir}/batch_metrics.json" ]; then
        log "SKIP rep=$rep -- already done (${out_dir}/batch_metrics.json exists)"
        return 0
    fi
    log "START rep=$rep tag=$tag"
    wait_for_bridge_idle
    python -m orchestrator.run_phase4_batch --max-steps 500 \
        --queries "$QUERIES" --persistent-memory --run-tag "$tag" \
        > "logs/phase4_${tag}.log" 2>&1
    local rc=$?
    if [ -f "${out_dir}/batch_metrics.json" ]; then
        sr=$(python3 -c "import json;print(json.load(open('${out_dir}/batch_metrics.json'))['success_rate'])" 2>/dev/null)
        spl=$(python3 -c "import json;print(json.load(open('${out_dir}/batch_metrics.json'))['mean_spl'])" 2>/dev/null)
        log "DONE rep=$rep success_rate=$sr mean_spl=$spl"
    else
        log "FAILED rep=$rep -- no batch_metrics.json produced (exit code $rc), see logs/phase4_${tag}.log"
    fi
}

log "=== scale30 repeats starting: N_REPS=$N_REPS (rep1 already run interactively as tag 'scale30', starting this driver at rep2) ==="
for rep in $(seq 2 "$N_REPS"); do
    run_one "$rep"
done
log "=== scale30 repeats complete: reps 2..$N_REPS ==="
