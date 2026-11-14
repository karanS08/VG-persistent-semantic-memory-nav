#!/bin/bash
# 2026-09-28: Post-Paper-Draft-Test-Plan.md, Priority Group A runs that need
# NEW data (the 2x2 memory x position factorial already exists and is only
# re-analysed, see results/post_paper/). All runs use the paper's frozen
# configuration (lexical_mode=prefix default, minconf=1, 500 steps) and are
# instrumented with the per-decision decisions_ep*.jsonl log (agent target
# vs. live-only counterfactual target + target age) added today, which is
# logging-only and does not change behaviour.
#   pp_full       : Test 0.1 reproduction of "full memory" + Test 2.2 data
#   pp_decay24h   : Test 1.4/3.1, persistent, all memory aged 24 h, decay on
#   pp_nodecay24h : same but lambda_base=0 (no temporal decay)
#   pp_memonly5   : factorial condition B (memory persists, position resets),
#                   instrumented, for helpful/harmful influence in B
# Interleaved by rep; resumable (run_batch skips completed dirs).
# When done, resumes the caption_head RQ1 rerun (scripts/run_captionhead_rq1.sh).
#
# Usage: nohup setsid bash scripts/run_postplan_priorityA.sh > logs/postplan_priorityA_driver.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rpv_mem
source scripts/_driver_lib.sh
Q8='a soft cushion,a large window,a wooden chair,a wall-mounted mirror,a dining table,a wall-mounted radiator,a doorway,a framed painting'
Q5='a large window,a wall-mounted radiator,a cabinet,a wall vent,a bench'
C=(--max-steps 500 --min-confirming-observations 1)
for rep in 1 2; do
    run_batch "phase4_batch_pp_full_rep${rep}" "${C[@]}" --queries "$Q8" --run-tag "pp_full_rep${rep}"
    run_batch "phase4_batch_persistent_pp_decay24h_rep${rep}" "${C[@]}" --queries "$Q8" --persistent-memory \
        --inject-memory-age-s 86400 --run-tag "pp_decay24h_rep${rep}"
    run_batch "phase4_batch_persistent_lambda0.0_pp_nodecay24h_rep${rep}" "${C[@]}" --queries "$Q8" --persistent-memory \
        --inject-memory-age-s 86400 --lambda-base 0 --run-tag "pp_nodecay24h_rep${rep}"
done
run_batch "phase4_batch_persist_memory_only_pp_memonly5_rep1" "${C[@]}" --queries "$Q5" --persist-memory-only --run-tag "pp_memonly5_rep1"
log "PRIORITY-A ALL DONE -- resuming caption_head RQ1 rerun"
bash scripts/run_captionhead_rq1.sh
