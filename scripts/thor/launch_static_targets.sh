#!/bin/bash
# Static-search check on candidate targets (refrigerator / sofa / television) in ProcTHOR house 1, seeded with the full tour memory.
# memory vs memory-off, 2 reps each. status: logs/static_targets.status ; failures: logs/static_targets.fail
cd "$(dirname "$0")/../.." || exit 1
cat > logs/static_targets.run.sh <<'INNER'
#!/bin/bash
cd __PWD__
PY=$HOME/miniconda3/envs/rpv_thor/bin/python
done_ok() { d=$(ls -d data/runs/phase4_batch_*$1 2>/dev/null | head -1); [ -n "$d" ] && [ -f "$d/batch_metrics.json" ]; }
run() { # tag query extra-args...
  tag=$1; q=$2; shift 2
  done_ok $tag && return 0
  for attempt in 1 2; do
    curl -s -X POST -m 30 localhost:8765/episode/stop > /dev/null 2>&1
    rm -rf data/runs/phase4_batch_*$tag data/runs/thor_gt_$tag
    echo "running $tag attempt $attempt $(date -Is)" > logs/static_targets.status
    THOR_HOUSE=1 THOR_CAM_FIX=1 THOR_GT_DIR=data/runs/thor_gt_$tag timeout 1500 $PY scripts/thor/run_thor_batch.py --no-rrd --run-tag $tag \
      --queries "$q" --max-steps 500 --min-confirming-observations 1 --init-memory-snapshot data/runs/seed_alarm_tour.json "$@" > logs/thor_$tag.log 2>&1
    done_ok $tag && return 0
    echo "FAILED attempt $attempt: $tag" >> logs/static_targets.fail
  done
}
for rep in 1 2; do
  for pair in "fridge:a refrigerator" "sofa:a sofa" "tv:a television"; do
    n=${pair%%:*}; q=${pair#*:}
    run sq_${n}_mem_r$rep "$q"
    run sq_${n}_off_r$rep "$q" --memory-off
  done
done
echo "done_exit_0 $(date -Is)" > logs/static_targets.status
INNER
sed -i "s#__PWD__#$(pwd)#" logs/static_targets.run.sh; chmod +x logs/static_targets.run.sh
echo "queued $(date -Is)" > logs/static_targets.status; rm -f logs/static_targets.fail
setsid nohup flock /tmp/rpv_mem_bridge.lock logs/static_targets.run.sh > /dev/null 2>&1 < /dev/null &
echo launched
