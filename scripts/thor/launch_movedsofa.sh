#!/bin/bash
# Moved-object experiment v4: query "a sofa" in ProcTHOR house 1, memory seeded with the full tour (both sofas at their original spots,
# entries aged 1 h). Conditions: static (nothing moved) | full | nodecay (--lambda-base 0) | memoff. In moved conditions BOTH sofas are
# teleported to other rooms before the agent acts ((14,15) and (6,13) THOR coords) so the remembered spots are empty and no stale sofa remains.
# 'fresh' = clean no-memory baseline (no seed, no persistence); the old memoff arm leaked seed entities into verification and is not used.
# Success must be re-scored post hoc: verified entity must sit on a sofa at its CURRENT GT position. status: logs/movedsofa.status
cd "$(dirname "$0")/../.." || exit 1
cat > logs/movedsofa.run.sh <<'INNER'
#!/bin/bash
cd __PWD__
PY=$HOME/miniconda3/envs/rpv_thor/bin/python
RELOC="0:Sofa|3|3:14.0:15.0;0:Sofa|5|2:6.0:13.0"
done_ok() { d=$(ls -d data/runs/phase4_batch_*$1 2>/dev/null | head -1); [ -n "$d" ] && [ -f "$d/batch_metrics.json" ]; }
run() { # tag reloc(0/1) extra-args...
  tag=$1; reloc=$2; shift 2
  done_ok $tag && return 0
  for attempt in 1 2; do
    curl -s -X POST -m 30 localhost:8765/episode/stop > /dev/null 2>&1
    rm -rf data/runs/phase4_batch_*$tag data/runs/thor_gt_$tag
    echo "running $tag attempt $attempt $(date -Is)" > logs/movedsofa.status
    case $tag in ms_fresh*) SEED="" ;; *) SEED="--persist-memory-only --init-memory-snapshot data/runs/seed_alarm_tour.json" ;; esac
    if [ "$reloc" = 1 ]; then export THOR_RELOCATE="$RELOC"; else unset THOR_RELOCATE; fi
    THOR_HOUSE=1 THOR_CAM_FIX=1 THOR_GT_DIR=data/runs/thor_gt_$tag timeout 1500 $PY scripts/thor/run_thor_batch.py --no-rrd --run-tag $tag \
      --queries "a sofa" --max-steps 500 --min-confirming-observations 1 $SEED "$@" > logs/thor_$tag.log 2>&1
    done_ok $tag && return 0
    echo "FAILED attempt $attempt: $tag" >> logs/movedsofa.fail
  done
}
for rep in 1 2 3 4 5 6 7 8; do
  run ms_fresh_r$rep   1
  run ms_static_r$rep  0
  run ms_full_r$rep    1
  run ms_nodecay_r$rep 1 --lambda-base 0
done
echo "done_exit_0 $(date -Is)" > logs/movedsofa.status
INNER
sed -i "s#__PWD__#$(pwd)#" logs/movedsofa.run.sh; chmod +x logs/movedsofa.run.sh
echo "queued $(date -Is)" > logs/movedsofa.status; rm -f logs/movedsofa.fail
setsid nohup flock /tmp/rpv_mem_bridge.lock logs/movedsofa.run.sh > /dev/null 2>&1 < /dev/null &
echo launched
