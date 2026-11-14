#!/bin/bash
# 2026-09-05: direct-recall threshold sweep (critique.md SS8). Runs the exact
# 9 theta_direct values the critique specifies, on the standard retuned-default
# 8-query battery (same set Table 2 uses), in both baseline (reset) and
# persistent-across-queries conditions -- baseline because that's this paper's
# primary reported condition, persistent because critique.md SS8 itself cites
# "788 baseline; 1354 persistent" direct recalls at theta=0.1 as the motivating
# evidence that the branch is threshold-sensitive and needs auditing in both
# regimes. One rep per (theta, condition) cell: this is a threshold-selection
# sweep, not a variance study (Section 7.5 already covers repeat-run variance
# separately) -- 9 thetas x 2 conditions = 18 runs.
#
# Requires the 2026-09-05 direct_recall_classification/precision fields in
# run_phase4_batch.py (correctness proxy: a direct-recall target counts as
# "correct" iff the episode succeeds via verification of that exact entity id).
#
# Per critique.md SS9 (no test-set leakage): treat this sweep's results as
# DEVELOPMENT-set threshold selection. Whatever theta_direct is chosen from
# this sweep should not then be re-tuned again against the final held-out
# evaluation once one exists.
#
# Usage: nohup bash scripts/run_direct_recall_sweep.sh > logs/direct_recall_sweep_driver.log 2>&1 &
#        disown

set -u
cd "$(dirname "$0")/.." || exit 1

QUERIES8='a soft cushion,a large window,a wooden chair,a wall-mounted mirror,a dining table,a wall-mounted radiator,a doorway,a framed painting'
THETAS='0.05 0.075 0.10 0.125 0.15 0.175 0.20 0.25 0.30'
CONDA_ENV=rpv_mem

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
source "$(dirname "$0")/_driver_lib.sh"

# 2026-09-05: run_phase4_batch.py builds its real output dir from
# f"theta{args.theta_direct}" using Python's float->str conversion, which
# strips trailing zeros (0.20 -> "0.2", 0.10 -> "0.1", 0.30 -> "0.3"). Our
# shell $theta stays the literal "0.20" etc, so the two would disagree on
# the directory name for 6 of these 18 cells -- a real mismatch that hit
# baseline theta0.20 (it completed fine, real data saved, but this driver
# logged a false FAILED and would have redone it wastefully on any future
# restart, since it never sees the true output dir under the old name).
# Fix: compute the actual float-normalized suffix bash-side to match.
norm_theta() { python3 -c "print(float('$1'))"; }

log "=== Direct-recall threshold sweep: baseline (reset) condition ==="
for theta in $THETAS; do
    tag="drsweep_baseline_theta${theta}"
    nt="$(norm_theta "$theta")"
    run_batch "phase4_batch_theta${nt}_${tag}" \
        --max-steps 500 --queries "$QUERIES8" --theta-direct "$theta" --run-tag "$tag"
done

log "=== Direct-recall threshold sweep: persistent-across-queries condition ==="
for theta in $THETAS; do
    tag="drsweep_persistent_theta${theta}"
    nt="$(norm_theta "$theta")"
    run_batch "phase4_batch_persistent_theta${nt}_${tag}" \
        --max-steps 500 --queries "$QUERIES8" --theta-direct "$theta" --persistent-memory --run-tag "$tag"
done

log "=== direct-recall threshold sweep driver complete ==="
