#!/bin/bash
# 2026-09-28: lag-compensated memory-off baseline (reviewer concern: is the
# RQ1 memory-off gap just DAAAM's ~13s async caption lag?). Runs
# --memory-off --live-window-s W (W in {15,30}) plus a same-code no-window
# memory-off control, 3 reps each, interleaved by rep so a partial run is
# still balanced. Report: data/results_postdraft/lag_baseline/REPORT.md.
#
# CODE: the --live-window-s flag exists ONLY on git branch
# worktree-agent-af3b3b0cbce82a5b9 (worktree below) -- the main checkout's
# packages/ are NOT used. This script puts the worktree packages first on
# PYTHONPATH and aborts if the import check does not resolve there.
# Lexical gate = committed default "prefix" (the paper's rule): the worktree
# branch does not contain the (uncommitted) caption_head mode, so these runs
# are comparable to the paper's minconf1default memory-off (0.125) / full
# (0.75 raw, 0.50 audited), NOT to the chrq1v2 caption_head reruns.
#
# QUEUEING: waits until no run_postplan_priorityA.sh / run_captionhead_rq1.sh
# driver is alive (so it does not interleave with the user's queued work),
# then each batch additionally goes through _driver_lib.sh run_batch (flock on
# /tmp/rpv_mem_bridge.lock + bridge-idle wait). Resumable: finished dirs skip.
#
# Usage (from anywhere):
#   nohup setsid bash /home/ksharma/mem_ws/scripts/postdraft/run_lag_baseline.sh \
#       > /home/ksharma/mem_ws/logs/lag_baseline_driver.log 2>&1 &
set -u
MAIN=/home/ksharma/mem_ws
WT=/home/ksharma/mem_ws/.claude/worktrees/agent-af3b3b0cbce82a5b9
WAIT_FOR_PATTERN="${WAIT_FOR_PATTERN:-scripts/run_(postplan_priorityA|captionhead_rq1)\.sh}"
N_REPS="${1:-3}"

cd "$MAIN" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rpv_mem
source "$MAIN/scripts/_driver_lib.sh"

export PYTHONPATH="$WT/packages/frontier_agent:$WT/packages/orchestrator:$WT/packages/memory_layer:$WT/packages/sim_bridge:$WT/packages/daaam_adapter${PYTHONPATH:+:$PYTHONPATH}"

log "import check (cwd=$(pwd))"
python - <<EOF || { log "ABORT: import check failed -- worktree code not on path"; exit 2; }
import frontier_agent.agent as a, orchestrator.run_phase4_batch as o, memory_layer.store as s
wt = "$WT/packages/"
for m in (a, o, s):
    print("  ", m.__name__, "->", m.__file__)
    assert m.__file__.startswith(wt), m.__file__
import inspect
assert "live_window_s" in inspect.signature(a.FrontierAgent.__init__).parameters
assert "--live-window-s" in inspect.getsource(o)
print("  OK: worktree code with live_window_s is what gets imported")
EOF

while pgrep -f "$WAIT_FOR_PATTERN" > /dev/null; do
    log "waiting 120s: other driver(s) still running: $(pgrep -af "$WAIT_FOR_PATTERN" | head -3 | tr '\n' ';')"
    sleep 120
done
log "no competing driver alive -- starting lag-baseline queue"

COMMON=(--max-steps 500 --min-confirming-observations 1 --memory-off)
for rep in $(seq 1 "$N_REPS"); do
    run_batch "phase4_batch_memory_off_livewin15_lagbase_rep${rep}" "${COMMON[@]}" --live-window-s 15 --run-tag "lagbase_rep${rep}"
    run_batch "phase4_batch_memory_off_livewin30_lagbase_rep${rep}" "${COMMON[@]}" --live-window-s 30 --run-tag "lagbase_rep${rep}"
    run_batch "phase4_batch_memory_off_lagbase_rep${rep}" "${COMMON[@]}" --run-tag "lagbase_rep${rep}"
done
log "LAG BASELINE ALL DONE"
