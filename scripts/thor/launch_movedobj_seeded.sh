#!/bin/bash
# Detached moving-object experiment v3 (seeded memory). Memory = a real earlier house tour (seed_alarm_tour.json) in which the robot saw and
# captioned the alarm clock (all entries aged 1 h). Then ONE query "an alarm clock" in ProcTHOR house 1, with the bear either
# moved (bedroom nightstand -> ~(14.0,15.0)) or left in place.  Conditions: full (decay on) | nodecay (--lambda-base 0) | memoff | static (not moved, decay on).
# Hardened: stops any stale DAAAM episode before each run and retries a crashed run once. status: logs/movedobj_seeded.status
cd "$(dirname "$0")/../.." || exit 1
cat > logs/movedobj_seeded.run.sh <<'EOF'
#!/bin/bash
cd __PWD__
PY=$HOME/miniconda3/envs/rpv_thor/bin/python
OID="AlarmClock|surface|6|63"
run() { # tag relocate(0/1) extra-args...
  tag=$1; reloc=$2; shift 2
  d=$(ls -d data/runs/phase4_batch_*_${tag} 2>/dev/null | head -1); [ -n "$d" ] && [ -f "$d/batch_metrics.json" ] && return 0
  for attempt in 1 2; do
    curl -s -X POST -m 30 localhost:8765/episode/stop > /dev/null 2>&1   # clear any stale bridge episode
    rm -rf data/runs/phase4_batch_*_${tag} data/runs/thor_gt_${tag}
    echo "running $tag attempt $attempt $(date -Is)" > logs/movedobj_seeded.status
    if [ "$reloc" = 1 ]; then export THOR_RELOCATE="0:${OID}:14.0:15.0"; else unset THOR_RELOCATE; fi
    THOR_HOUSE=1 THOR_CAM_FIX=1 THOR_GT_DIR=data/runs/thor_gt_${tag} timeout 1500 $PY scripts/thor/run_thor_batch.py --no-rrd --run-tag ${tag} \
      --queries "an alarm clock" --max-steps 500 --min-confirming-observations 1 --init-memory-snapshot data/runs/seed_alarm_tour.json "$@" > logs/thor_${tag}.log 2>&1
    d=$(ls -d data/runs/phase4_batch_*_${tag} 2>/dev/null | head -1); [ -n "$d" ] && [ -f "$d/batch_metrics.json" ] && return 0
    echo "FAILED attempt $attempt: $tag" >> logs/movedobj_seeded.fail
  done
}
for rep in 1 2 3 4; do
  run tbs_static_r$rep  0
  run tbs_full_r$rep    1
  run tbs_nodecay_r$rep 1 --lambda-base 0
  run tbs_memoff_r$rep  1 --memory-off
done
echo "done_exit_0 $(date -Is)" > logs/movedobj_seeded.status
EOF
sed -i "s#__PWD__#$(pwd)#" logs/movedobj_seeded.run.sh; chmod +x logs/movedobj_seeded.run.sh
echo "queued $(date -Is)" > logs/movedobj_seeded.status; rm -f logs/movedobj_seeded.fail
setsid nohup flock /tmp/rpv_mem_bridge.lock logs/movedobj_seeded.run.sh > /dev/null 2>&1 < /dev/null &
echo launched movedobj seeded
