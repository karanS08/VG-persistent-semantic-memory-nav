#!/bin/bash
# 2026-09-29: PRIORITY REORDER of run_master_queue.sh (user request): essential
# batches first (P1-P4), then the master queue body unchanged; run_batch skips
# anything already done, so nothing runs twice.
# Original header follows.
# 2026-09-28: ONE unified queue replacing run_postplan_priorityA.sh (and the
# run_captionhead_rq1.sh it chains into) and scripts/postdraft/run_lag_baseline.sh,
# ordered by relevance to the ACRA claims, then by cost. Unattended and
# independent of Claude Code (nohup setsid). Resumable: run_batch skips any
# dir with batch_metrics.json, so relaunch after a crash/reboot just continues.
#
#   Tier A  decides/defends current paper claims        (~6.5 h)
#   Tier B  strengthens the paper / tests caption_head  (~6.5 h)
#   Tier C  low value for ACRA (static scenes) or risky (~5.5 h)
#
# Safety: shared flock run_batch (one episode on the bridge at a time); disk
# guard pauses below MIN_FREE_GB; `touch data/results_postdraft/STOP_MASTER_QUEUE`
# stops it before the next batch. Lag-baseline batches import the
# live_window_s code from the agent worktree (checked at startup); every other
# batch uses the main checkout's code, exactly as the replaced drivers did.
#
# Not included (need code/data/hardware): 1.3 relevance-threshold sweep,
# 1.5 reliability-component ablations, 4.1-4.4 relocation/mobility/forked
# intervention, 5.x. Test 2.2 is analysis (classify_influence_geodesic.py).
#
# Usage: nohup setsid bash scripts/postdraft/run_master_queue.sh > logs/master_queue_driver.log 2>&1 &
set -u
MAIN=/home/ksharma/mem_ws
WT=$MAIN/.claude/worktrees/agent-af3b3b0cbce82a5b9
cd "$MAIN" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rpv_mem
source scripts/_driver_lib.sh

MIN_FREE_GB="${MIN_FREE_GB:-15}"
STOP_FILE="data/results_postdraft/STOP_MASTER_QUEUE"
WT_PATH="$WT/packages/frontier_agent:$WT/packages/orchestrator:$WT/packages/memory_layer:$WT/packages/sim_bridge:$WT/packages/daaam_adapter"

C=(--max-steps 500 --min-confirming-observations 1)
CH=(--max-steps 500 --min-confirming-observations 1 --lexical-mode caption_head)
Q8='a soft cushion,a large window,a wooden chair,a wall-mounted mirror,a dining table,a wall-mounted radiator,a doorway,a framed painting'
Q5='a large window,a wall-mounted radiator,a cabinet,a wall vent,a bench'
QM='a cabinet,a trash bin,a wall vent,a curtain,a fireplace,a lamp,a wall clock,a bench,a staircase,a bicycle'
QPARA='a throw pillow,a big glass pane in the wall,a chair made of wood,a mirror hanging on the wall,a table for eating meals,a heater fixed to the wall,an open door frame,a picture in a frame'
# no GT for 00770: absence must still be confirmed by inspecting any match
QABS='a motorcycle,a surfboard,a horse,a traffic light,a canoe,a fire hydrant,a treadmill,a grand piano'
S337='data/hm3d/example/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb'
Q337='a framed painting,a lamp,a soft cushion,a cabinet,a curtain,a rug,a pillow,a sofa,a bicycle,a laptop'
S861='data/hm3d/example/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb'
# 00861 has semantic GT: first 8 present in semantic.txt, last 2 have 0 matches
Q861='a chair,a mirror,a lamp,a pillow,a toilet,a sink,a bed,a refrigerator,a bicycle,a piano'

guard() {
    [ -f "$STOP_FILE" ] && { log "STOP file present -- exiting"; exit 0; }
    while :; do
        free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
        [ "$free_gb" -ge "$MIN_FREE_GB" ] && return
        log "disk guard: ${free_gb}G free < ${MIN_FREE_GB}G, pausing 10 min"
        sleep 600
        [ -f "$STOP_FILE" ] && { log "STOP file present -- exiting"; exit 0; }
    done
}
rb() { guard; run_batch "$@"; }
rb_lag() { guard; PYTHONPATH="$WT_PATH${PYTHONPATH:+:$PYTHONPATH}" run_batch "$@"; }

log "import check for lag batches (worktree code)"
PYTHONPATH="$WT_PATH" python - <<EOF || { log "ABORT: worktree live_window_s code not importable"; exit 2; }
import inspect, frontier_agent.agent as a, orchestrator.run_phase4_batch as o
for m in (a, o):
    assert m.__file__.startswith("$WT/packages/"), m.__file__
assert "live_window_s" in inspect.signature(a.FrontierAgent.__init__).parameters
print("  OK")
EOF
log "=== priority queue starting (essential first, then the full master queue) ==="

log "--- P1: lag baseline, 30 s window vs same-code memory-off control, 3 reps (RQ1 reviewer check) ---"
for rep in 1 2 3; do
    rb_lag "phase4_batch_memory_off_livewin30_lagbase_rep${rep}" "${C[@]}" --memory-off --live-window-s 30 --run-tag "lagbase_rep${rep}"
    rb_lag "phase4_batch_memory_off_lagbase_rep${rep}" "${C[@]}" --memory-off --run-tag "lagbase_rep${rep}"
done

log "--- P2: all verification removed, repeat 2 (headline 10/10 reported vs 0/10 audited) ---"
rb phase4_batch_persistent_minconf0_no_fresh_evidence_guard_no_relevance_gate_no_lexical_gate_full_no_verification_rep2 \
    --max-steps 500 --queries "$QM" --persistent-memory --min-confirming-observations 0 \
    --no-fresh-evidence-guard --no-relevance-gate --no-lexical-gate --run-tag full_no_verification_rep2

log "--- P3: held-out 00337 memory-off (RQ1 on a second scene) ---"
rb phase4_batch_memory_off_heldout00337_memoff "${C[@]}" --queries "$Q337" --scene-path "$S337" --memory-off --run-tag heldout00337_memoff

log "--- P4: memory-only factorial cell with influence logging ---"
rb phase4_batch_persist_memory_only_pp_memonly5_rep1 "${C[@]}" --queries "$Q5" --persist-memory-only --run-tag pp_memonly5_rep1

log "=== essential set done -- continuing with the full master queue (finished dirs are skipped) ==="

log "--- A1: reproduction of full memory + influence-target logging (Test 0.1/2.2) ---"
rb phase4_batch_pp_full_rep2 "${C[@]}" --queries "$Q8" --run-tag pp_full_rep2

log "--- A2: lag baseline -- 30 s window vs same-code memory-off control, 3 reps ---"
for rep in 1 2 3; do
    rb_lag "phase4_batch_memory_off_livewin30_lagbase_rep${rep}" "${C[@]}" --memory-off --live-window-s 30 --run-tag "lagbase_rep${rep}"
    rb_lag "phase4_batch_memory_off_lagbase_rep${rep}" "${C[@]}" --memory-off --run-tag "lagbase_rep${rep}"
done

log "--- A3: memory-only factorial cell with influence logging (Test 2.2) ---"
rb phase4_batch_persist_memory_only_pp_memonly5_rep1 "${C[@]}" --queries "$Q5" --persist-memory-only --run-tag pp_memonly5_rep1

log "--- A4: verification leave-one-out, second repeat (Test 3.3) ---"
rb phase4_batch_semantic00770mined_minconf1default_leakfix_rep2 "${C[@]}" --queries "$QM" --run-tag semantic00770mined_minconf1default_leakfix_rep2
rb phase4_batch_no_lexical_gate_verifyloo_nolexical_rep2 "${C[@]}" --queries "$QM" --no-lexical-gate --run-tag verifyloo_nolexical_rep2
rb phase4_batch_persistent_no_fresh_evidence_guard_verifyloo_nofreshguard_rep2 "${C[@]}" --queries "$QM" --persistent-memory --no-fresh-evidence-guard --run-tag verifyloo_nofreshguard_rep2
rb phase4_batch_persistent_minconf0_no_fresh_evidence_guard_no_relevance_gate_no_lexical_gate_full_no_verification_rep2 \
    --max-steps 500 --queries "$QM" --persistent-memory --min-confirming-observations 0 \
    --no-fresh-evidence-guard --no-relevance-gate --no-lexical-gate --run-tag full_no_verification_rep2
rb phase4_batch_no_relevance_gate_verifyloo_norelevance_rep2 "${C[@]}" --queries "$QM" --no-relevance-gate --run-tag verifyloo_norelevance_rep2
rb phase4_batch_minconf0_verifyloo_nomultiobs_rep2 --max-steps 500 --queries "$QM" --min-confirming-observations 0 --run-tag verifyloo_nomultiobs_rep2

log "--- A5: held-out scene 00337 memory-off (RQ1 on second scene, Test 3.5) ---"
rb phase4_batch_memory_off_heldout00337_memoff "${C[@]}" --queries "$Q337" --scene-path "$S337" --memory-off --run-tag heldout00337_memoff

log "--- A6: absent-object specificity (Test 2.3) ---"
rb phase4_batch_mq_absent_rep1 "${C[@]}" --queries "$QABS" --run-tag mq_absent_rep1

log "--- B1: caption_head RQ1 rerun (uses main-checkout code, as run_captionhead_rq1.sh) ---"
for rep in 1 2 3; do
    rb "phase4_batch_lexcaption_head_chrq1v2_rep${rep}" "${CH[@]}" --run-tag "chrq1v2_rep${rep}"
    rb "phase4_batch_memory_off_lexcaption_head_chrq1v2_rep${rep}" "${CH[@]}" --memory-off --run-tag "chrq1v2_rep${rep}"
    rb "phase4_batch_persistent_lexcaption_head_chrq1v2_rep${rep}" "${CH[@]}" --persistent-memory --run-tag "chrq1v2_rep${rep}"
done

log "--- B2: held-out 00337 repeats ---"
rb phase4_batch_heldout00337_baseline_rep2 "${C[@]}" --queries "$Q337" --scene-path "$S337" --run-tag heldout00337_baseline_rep2
rb phase4_batch_persistent_heldout00337_persistent_rep2 "${C[@]}" --queries "$Q337" --scene-path "$S337" --persistent-memory --run-tag heldout00337_persistent_rep2

log "--- B3: GT pilot on annotated scene 00861 (previous run there: 0/12) ---"
rb phase4_batch_mq_gt861_rep1 "${C[@]}" --queries "$Q861" --scene-path "$S861" --run-tag mq_gt861_rep1

log "--- B4: absent objects persistent, confirmation sweep, paraphrase (Tests 2.3/1.1/2.4) ---"
rb phase4_batch_persistent_mq_absent_rep1 "${C[@]}" --queries "$QABS" --persistent-memory --run-tag mq_absent_rep1
rb phase4_batch_minconf2_mq_rep1 --max-steps 500 --min-confirming-observations 2 --run-tag mq_rep1
rb phase4_batch_minconf3_mq_rep1 --max-steps 500 --min-confirming-observations 3 --run-tag mq_rep1
rb phase4_batch_mq_paraphrase_rep1 "${C[@]}" --queries "$QPARA" --run-tag mq_paraphrase_rep1
rb phase4_batch_persistent_mq_paraphrase_rep1 "${C[@]}" --queries "$QPARA" --persistent-memory --run-tag mq_paraphrase_rep1

log "--- C1: decay tests (static scenes: validates mechanism only) ---"
rb phase4_batch_persistent_pp_decay24h_rep2 "${C[@]}" --queries "$Q8" --persistent-memory --inject-memory-age-s 86400 --run-tag pp_decay24h_rep2
rb phase4_batch_persistent_lambda0.0_pp_nodecay24h_rep2 "${C[@]}" --queries "$Q8" --persistent-memory --inject-memory-age-s 86400 --lambda-base 0 --run-tag pp_nodecay24h_rep2
rb phase4_batch_lambda0.0_mq_nodecay_rep1 "${C[@]}" --lambda-base 0 --run-tag mq_nodecay_rep1
for lb in 0.00025 0.0005 0.002 0.004; do   # default lambda_base = 0.001
    rb "phase4_batch_persistent_lambda${lb}_mq_decaysens6h" "${C[@]}" --queries "$Q8" --persistent-memory \
        --inject-memory-age-s 21600 --lambda-base "$lb" --run-tag mq_decaysens6h
done

log "--- C2: 15 s lag window (marginal: captions arrive ~13 s old) ---"
for rep in 1 2 3; do
    rb_lag "phase4_batch_memory_off_livewin15_lagbase_rep${rep}" "${C[@]}" --memory-off --live-window-s 15 --run-tag "lagbase_rep${rep}"
done

log "--- C3: 00861 GT pilot, memory-off ---"
rb phase4_batch_memory_off_mq_gt861_rep1 "${C[@]}" --queries "$Q861" --scene-path "$S861" --memory-off --run-tag mq_gt861_rep1

log "=== priority queue ALL DONE ==="
