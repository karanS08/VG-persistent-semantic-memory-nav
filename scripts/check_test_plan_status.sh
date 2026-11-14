#!/bin/bash
# Claude-independent status check for the background test-plan runs.
# Works whether or not any Claude Code session is alive -- just reads real
# process state and files on disk. Run any time: bash scripts/check_test_plan_status.sh

cd "$(dirname "$0")/.." || exit 1

echo "=== Known driver processes ==="
for pid in 2692798; do
    if ps -p "$pid" > /dev/null 2>&1; then
        etime=$(ps -o etime= -p "$pid" | tr -d ' ')
        cmd=$(ps -o cmd= -p "$pid")
        echo "  pid $pid: ALIVE (running $etime) -- $cmd"
    else
        echo "  pid $pid: not running (finished, or check logs/ for a newer driver)"
    fi
done

echo
echo "=== Latest log activity ==="
for log in logs/phase4_scan60_confirm.log logs/phase4_longsession10_covmem.log; do
    if [ -f "$log" ]; then
        echo "--- $log (last non-engine line) ---"
        grep -v '^\[' "$log" | tail -3
    fi
done

echo
echo "=== TEST_PLAN.md items 1-18: batch_metrics.json present? ==="
declare -A dirs=(
    [1]=phase4_batch [2]=phase4_batch_memory_off [3]=phase4_batch_no_mobility
    [4]=phase4_batch_no_verify_discount [5]=phase4_batch_persistent
    [6]=phase4_batch_theta0.1 [7]=phase4_batch_minconf1 [8]=phase4_batch_minconf3
    [9]=phase4_batch_persistent_theta0.1 [10]=phase4_batch_rep2 [11]=phase4_batch_rep3
    [12]=phase4_batch_persistent_rep2 [13]=phase4_batch_memory_off_persistent
    [14]=phase4_batch_persistent_minconf1
    [15]=phase4_batch_semantic00770mined [16]=phase4_batch_persistent_semantic00770mined
    [17]=phase4_batch_minconf1default [17b]=phase4_batch_memory_off_minconf1default
    [17c]=phase4_batch_no_mobility_minconf1default [17d]=phase4_batch_no_verify_discount_minconf1default
    [17e]=phase4_batch_persistent_minconf1default [18]=phase4_batch_minconf1default_rep2
    [18b]=phase4_batch_persistent_minconf1default_rep2
)
for n in $(echo "${!dirs[@]}" | tr ' ' '\n' | sort -n); do
    d="data/runs/${dirs[$n]}"
    if [ -f "$d/batch_metrics.json" ]; then
        sr=$(python3 -c "import json;print(json.load(open('$d/batch_metrics.json'))['success_rate'])" 2>/dev/null)
        echo "  [$n] ${dirs[$n]}: DONE (success_rate=$sr)"
    else
        echo "  [$n] ${dirs[$n]}: not yet"
    fi
done

echo
echo "=== Reset-vs-Continue benchmark (Tier 5) ==="
for d in phase4_batch_reset5 phase4_batch_persistent_continue5; do
    f="data/runs/$d/batch_metrics.json"
    if [ -f "$f" ]; then
        sr=$(python3 -c "import json;print(json.load(open('$f'))['success_rate'])" 2>/dev/null)
        echo "  $d: DONE (success_rate=$sr)"
    else
        echo "  $d: not yet"
    fi
done

echo
echo "=== Gap-closing batch (2026-09-02: ablation repeats, Tier 4b re-run, long session) ==="
for d in phase4_batch_no_mobility_minconf1default_rep2 phase4_batch_no_verify_discount_minconf1default_rep2 \
         phase4_batch_semantic00770mined_minconf1default phase4_batch_persistent_semantic00770mined_minconf1default \
         phase4_batch_persistent_longsession10; do
    f="data/runs/$d/batch_metrics.json"
    if [ -f "$f" ]; then
        sr=$(python3 -c "import json;print(json.load(open('$f'))['success_rate'])" 2>/dev/null)
        echo "  $d: DONE (success_rate=$sr)"
    else
        echo "  $d: not yet"
    fi
done

echo
echo "=== Leak-fix confirmation (2026-09-02: description-only + prefix-match fix) ==="
for d in phase4_batch_semantic00770mined_minconf1default_leakfix \
         phase4_batch_persistent_semantic00770mined_minconf1default_leakfix; do
    f="data/runs/$d/batch_metrics.json"
    if [ -f "$f" ]; then
        sr=$(python3 -c "import json;print(json.load(open('$f'))['success_rate'])" 2>/dev/null)
        echo "  $d: DONE (success_rate=$sr)"
    else
        echo "  $d: not yet"
    fi
done

echo
echo "=== scan_steps=60 confirmation (2026-09-02: depth-search duration fix) ==="
for d in phase4_batch_scan60 phase4_batch_persistent_scan60 \
         phase4_batch_semantic00770mined_scan60 phase4_batch_persistent_semantic00770mined_scan60; do
    f="data/runs/$d/batch_metrics.json"
    if [ -f "$f" ]; then
        sr=$(python3 -c "import json;print(json.load(open('$f'))['success_rate'])" 2>/dev/null)
        echo "  $d: DONE (success_rate=$sr)"
    else
        echo "  $d: not yet"
    fi
done

echo
echo "=== Coverage memory validation (2026-09-04, vs. pre-feature baseline phase4_batch_persistent_longsession10: success_rate=0.8, mean_spl=0.6236) ==="
for d in phase4_batch_persistent_longsession10_covmem phase4_batch_persistent_freshrestart_smoketest phase4_batch_persistent_longsession10_covmem_v2 phase4_batch_persistent_longsession10_covmem_rep2 phase4_batch_persistent_longsession10_samedaybaseline phase4_batch_persistent_longsession10_samedaybaseline_rep2 phase4_batch_persistent_longsession10_covmem_rep3 phase4_batch_persistent_longsession10_samedaybaseline_rep3; do
    f="data/runs/$d/batch_metrics.json"
    if [ -f "$f" ]; then
        sr=$(python3 -c "import json;d=json.load(open('$f'));print(d['success_rate'], d['mean_spl'])" 2>/dev/null)
        echo "  $d: DONE (success_rate, mean_spl = $sr)"
    else
        echo "  $d: not yet"
    fi
done
