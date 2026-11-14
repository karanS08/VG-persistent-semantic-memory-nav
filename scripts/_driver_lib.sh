# 2026-09-05: shared helpers for background batch-driver scripts. Source this
# (`source "$(dirname "$0")/_driver_lib.sh"`) instead of redefining
# wait_for_bridge_idle/run_batch in each driver -- also fixes a real race
# condition found today: two driver scripts each independently polling
# episode_running==False, then BOTH calling client.start_episode(), one
# winning and one getting a 409 (see PROGRESS.md 2026-09-05, the
# persist-position-only rep3 failure). A flock-based lock around the whole
# "wait until idle, then start" sequence makes it atomic across drivers.

BRIDGE_HEALTH_URL="${BRIDGE_HEALTH_URL:-http://localhost:8765/health}"
BRIDGE_LOCK_FILE="${BRIDGE_LOCK_FILE:-/tmp/rpv_mem_bridge.lock}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

_bridge_idle() {
    running=$(curl -s "$BRIDGE_HEALTH_URL" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('episode_running', True))" 2>/dev/null)
    [ "$running" = "False" ]
}

# run_batch <out_dir_name> <run_phase4_batch.py args...>
# Acquires BRIDGE_LOCK_FILE for the whole wait+start sequence so no other
# driver holding the same lock can start an episode in between.
#
# 2026-09-05: the skip-if-done check below used to run ONLY here, before
# acquiring the lock -- if a run finished (writing batch_metrics.json)
# while a second driver was still blocked waiting for the lock (e.g. after
# a mid-session setsid relaunch left this call's initial check stale), the
# second driver would still launch a fully redundant duplicate run once it
# got the lock, silently overwriting the just-completed run's .rrd files
# in the same output directory. Hit this for real (verifyloo_norelevance
# got a duplicate run after a driver restart). Fix: re-check inside the
# lock, immediately before launching python, since that's the only check
# that's actually race-free.
run_batch() {
    local out_dir="data/runs/$1"
    shift
    if [ -f "${out_dir}/batch_metrics.json" ]; then
        log "SKIP ${out_dir} -- already done"
        return 0
    fi
    log "START ${out_dir} :: $*"
    (
        flock -x 200
        if [ -f "${out_dir}/batch_metrics.json" ]; then
            log "SKIP ${out_dir} -- already done (completed while waiting for lock)"
            exit 0
        fi
        while ! _bridge_idle; do
            log "bridge busy or unreachable, waiting 30s (holding lock) for ${out_dir}..."
            sleep 30
        done
        if [ -f "${out_dir}/batch_metrics.json" ]; then
            log "SKIP ${out_dir} -- already done (completed while waiting for bridge)"
            exit 0
        fi
        python -m orchestrator.run_phase4_batch "$@" > "logs/phase4_$(basename "$out_dir").log" 2>&1
    ) 200>"$BRIDGE_LOCK_FILE"
    local rc=$?
    if [ -f "${out_dir}/batch_metrics.json" ]; then
        sr=$(python3 -c "import json;print(json.load(open('${out_dir}/batch_metrics.json'))['success_rate'])" 2>/dev/null)
        spl=$(python3 -c "import json;print(json.load(open('${out_dir}/batch_metrics.json'))['mean_spl'])" 2>/dev/null)
        log "DONE ${out_dir} success_rate=$sr mean_spl=$spl"
    else
        log "FAILED ${out_dir} -- no batch_metrics.json (exit $rc), see logs/phase4_$(basename "$out_dir").log"
    fi
}
