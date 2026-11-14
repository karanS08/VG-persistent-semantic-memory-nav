#!/bin/bash
# 2026-09-04: properly-powered coverage-memory A/B comparison.
# n=3 per side (interactive, this afternoon) showed a consistent but
# not-yet-significant direction (success_rate ON mean=0.767 vs OFF
# mean=0.633, t~1.4 at n=3 -- nowhere near significant). This script
# gathers many more independent repeats, alternating ON/OFF so any
# slow environmental drift over the run hits both conditions evenly,
# fully detached from any interactive session so it survives a dropped
# SSH connection / Claude session ending.
#
# Usage: nohup bash scripts/run_coverage_memory_megabatch.sh [N_REPS_PER_SIDE] > logs/megabatch_driver.log 2>&1 &
#        disown
# Default N_REPS_PER_SIDE=10 (20 batches total, ~20min each => ~6-7 hours).
#
# Resumable: skips any rep whose batch_metrics.json already exists, so it
# can be killed and relaunched (same N_REPS_PER_SIDE) without repeating
# already-done work or losing already-collected data.

set -u
cd "$(dirname "$0")/.." || exit 1

N_REPS="${1:-10}"
QUERIES='a soft cushion,a large window,a wooden chair,a wall-mounted mirror,a dining table,a wall-mounted radiator,a doorway,a framed painting,a cabinet,a bench'
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
        log "bridge busy or unreachable (episode_running=$running), waiting 15s..."
        sleep 15
    done
}

run_one() {
    local rep="$1" condition="$2" extra_flag="$3"
    local tag="megabatch_${condition}_rep${rep}"
    local out_dir="data/runs/phase4_batch_persistent"
    [ "$condition" = "off" ] && out_dir="${out_dir}_no_coverage_memory"
    out_dir="${out_dir}_${tag}"
    if [ -f "${out_dir}/batch_metrics.json" ]; then
        log "SKIP rep=$rep condition=$condition -- already done (${out_dir}/batch_metrics.json exists)"
        return 0
    fi
    log "START rep=$rep condition=$condition tag=$tag"
    wait_for_bridge_idle
    python -m orchestrator.run_phase4_batch --max-steps 500 \
        --queries "$QUERIES" --persistent-memory $extra_flag --run-tag "$tag" \
        > "logs/phase4_${tag}.log" 2>&1
    local rc=$?
    if [ -f "${out_dir}/batch_metrics.json" ]; then
        sr=$(python3 -c "import json;print(json.load(open('${out_dir}/batch_metrics.json'))['success_rate'])" 2>/dev/null)
        spl=$(python3 -c "import json;print(json.load(open('${out_dir}/batch_metrics.json'))['mean_spl'])" 2>/dev/null)
        log "DONE rep=$rep condition=$condition success_rate=$sr mean_spl=$spl"
    else
        log "FAILED rep=$rep condition=$condition -- no batch_metrics.json produced (exit code $rc), see logs/phase4_${tag}.log"
    fi
}

log "=== megabatch starting: $N_REPS reps per side ==="
for rep in $(seq 1 "$N_REPS"); do
    run_one "$rep" "on" ""
    run_one "$rep" "off" "--no-coverage-memory"
done
log "=== megabatch complete: $N_REPS reps per side ==="
