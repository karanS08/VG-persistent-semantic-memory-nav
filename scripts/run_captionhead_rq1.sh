#!/bin/bash
# 2026-09-28: re-run the RQ1 standard battery (paper Fig. ablation) with the
# new caption_head lexical gate (verification.py _caption_head_consistent),
# 3 repeats per condition instead of the paper's single run, to (a) test
# whether removing the audited cushion/doorway/painting false-positive class
# raises AUDITED success, (b) tighten the memory-off vs. memory comparison.
# Interleaved by repeat so a partial run is still balanced across conditions.
# Resumable: run_batch skips any output dir whose batch_metrics.json exists.
#
# 2026-09-28 (later): switched to tag chrq1v2 after the v2 surface-subject
# rule (verification.py _SURFACE_CARRIERS); chrq1_rep1 runs used v1.
# Usage: nohup setsid bash scripts/run_captionhead_rq1.sh > logs/captionhead_rq1_driver.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rpv_mem
source scripts/_driver_lib.sh
N_REPS="${1:-3}"
COMMON=(--max-steps 500 --min-confirming-observations 1 --lexical-mode caption_head)
for rep in $(seq 1 "$N_REPS"); do
    run_batch "phase4_batch_lexcaption_head_chrq1v2_rep${rep}" "${COMMON[@]}" --run-tag "chrq1v2_rep${rep}"
    run_batch "phase4_batch_memory_off_lexcaption_head_chrq1v2_rep${rep}" "${COMMON[@]}" --memory-off --run-tag "chrq1v2_rep${rep}"
    run_batch "phase4_batch_persistent_lexcaption_head_chrq1v2_rep${rep}" "${COMMON[@]}" --persistent-memory --run-tag "chrq1v2_rep${rep}"
done
log "ALL DONE"
