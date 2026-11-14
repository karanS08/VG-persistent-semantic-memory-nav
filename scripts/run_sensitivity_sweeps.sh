#!/bin/bash
# 2026-09-06: two remaining MEDIUM-priority sensitivity sweeps from critique.md /
# critique_2's priority list (SS42): multi-anchor fusion sigma (previously
# hand-selected at 6m, never swept) and confirmation-threshold extension to 4/5
# (previously only 1/2/3 tested). Both are threshold-selection sweeps on the
# standard 8-query battery, one rep per value, matching the same convention the
# theta_direct sweep used -- not a variance study (Section 7.5 covers that
# separately).
#
# Usage: nohup bash scripts/run_sensitivity_sweeps.sh > logs/sensitivity_sweeps_driver.log 2>&1 &

set -u
cd "$(dirname "$0")/.." || exit 1

QUERIES8='a soft cushion,a large window,a wooden chair,a wall-mounted mirror,a dining table,a wall-mounted radiator,a doorway,a framed painting'
CONDA_ENV=rpv_mem

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
source "$(dirname "$0")/_driver_lib.sh"

# 2026-09-06: same float->str normalization bug as the theta_direct sweep hit
# earlier (see run_direct_recall_sweep.sh's own fix/commit for the full
# writeup) -- run_phase4_batch.py builds its real output dir from
# f"anchorsigma{args.anchor_sigma}" (a Python float), which renders "2" as
# "2.0", not "2". Hit this for real: sigma=2 completed successfully (0.62
# success, 0.1447 SPL) but landed in phase4_batch_anchorsigma2.0_anchorsigma2,
# not the phase4_batch_anchorsigma2.0 this script originally checked for
# (missing the run-tag suffix entirely) -- got a false FAILED and would have
# wastefully redone it on any restart.
norm_sigma() { python3 -c "print(float('$1'))"; }

log "=== Multi-anchor sigma sweep (w=1 fixed, sigma swept) ==="
for sigma in 2 4 6 8 10; do
    tag="anchorsigma${sigma}"
    ns="$(norm_sigma "$sigma")"
    run_batch "phase4_batch_anchorsigma${ns}_${tag}" \
        --max-steps 500 --queries "$QUERIES8" --anchor-sigma "$sigma" --run-tag "$tag"
done

log "=== Confirmation-threshold extension: 4 and 5 (1/2/3 already tested) ==="
for n in 4 5; do
    run_batch "phase4_batch_minconf${n}_confirmext${n}" \
        --max-steps 500 --queries "$QUERIES8" --min-confirming-observations "$n" --run-tag "confirmext${n}"
done

log "=== sensitivity sweeps driver complete ==="
