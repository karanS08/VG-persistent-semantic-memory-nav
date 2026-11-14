#!/bin/bash
# 2026-09-05: verification leave-one-gate-out ablation (critique.md SS5/SS33).
# Baselines (full verifier, both modes) already archived:
#   results/batch_runs/phase4_batch_semantic00770mined_minconf1default_leakfix
#   results/batch_runs/phase4_batch_persistent_semantic00770mined_minconf1default_leakfix
# This script runs only the 4 new single-gate-removed conditions, same real
# 10-query set (8 real categories + 2 adversarial/absent -- staircase,
# bicycle -- needed to compute genuine TP/FP/precision/recall/F1, not just
# success rate). Gate 1 (fresh-evidence guard) only matters in
# persistent/position-carryover mode, since it's gated on `not reset_env`;
# gates 2-4 are tested in the ordinary non-persistent mode matching the
# existing baseline.
#
# Usage: nohup bash scripts/run_verification_leaveoneout.sh > logs/verification_leaveoneout_driver.log 2>&1 &
#        disown

set -u
cd "$(dirname "$0")/.." || exit 1

QUERIES_MINED='a cabinet,a trash bin,a wall vent,a curtain,a fireplace,a lamp,a wall clock,a bench,a staircase,a bicycle'
BRIDGE_HEALTH_URL="http://localhost:8765/health"
CONDA_ENV=rpv_mem

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# 2026-09-05: switched to the shared flock-based run_batch after this
# script's own unlocked run_batch raced with a concurrent driver and lost
# its first run (verifyloo_norelevance, 409 at start_episode). run_batch
# skips any rep whose batch_metrics.json exists, so restarting is safe.
source "$(dirname "$0")/_driver_lib.sh"

log "=== Verification leave-one-out: gate 2 removed (no relevance threshold) ==="
run_batch "phase4_batch_no_relevance_gate_verifyloo_norelevance" \
    --max-steps 500 --queries "$QUERIES_MINED" --no-relevance-gate --run-tag verifyloo_norelevance

log "=== Verification leave-one-out: gate 3 removed (no multi-observation reconfirmation) ==="
run_batch "phase4_batch_minconf0_verifyloo_nomultiobs" \
    --max-steps 500 --queries "$QUERIES_MINED" --min-confirming-observations 0 --run-tag verifyloo_nomultiobs

log "=== Verification leave-one-out: gate 4 removed (no lexical consistency) ==="
run_batch "phase4_batch_no_lexical_gate_verifyloo_nolexical" \
    --max-steps 500 --queries "$QUERIES_MINED" --no-lexical-gate --run-tag verifyloo_nolexical

log "=== Verification leave-one-out: gate 1 removed (no fresh-evidence guard, persistent mode) ==="
run_batch "phase4_batch_persistent_no_fresh_evidence_guard_verifyloo_nofreshguard" \
    --max-steps 500 --queries "$QUERIES_MINED" --persistent-memory --no-fresh-evidence-guard --run-tag verifyloo_nofreshguard

log "=== verification leave-one-out driver complete ==="
