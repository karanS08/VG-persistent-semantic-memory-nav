#!/bin/bash
# Moved-object experiment v7 (2026-10-31) = v6 + three setup fixes, ProcTHOR house 1, query "a sofa", 1500-step budget.
# Fixes vs v6: (1) THOR_RELOCATE_CLEAR=1 -- sofas placed in clear floor space (>=0.55 m from every object, house stays
#   connected, agent can stand within 1 m), spots from scripts/thor/find_clear_spots.py; v6 buried Sofa|3|3 in a bed.
#   The reachable grid is rebuilt after the move (v6 planned through the moved sofas).
# (2) agent drops a grid cell it repeatedly collides at (ThorPathfinder.block_point); v6 start s4 livelocked at step 18.
# (3) pre-set exclusion rule, outcome-independent: an attempt with < 0.1 DAAAM captions per step is an infrastructure
#   failure; it is logged to logs/v7.excluded and re-run (max 3 attempts). Mis-captions are NOT excluded.
# Config (all flags off by default in the code; see run_phase4_batch.py / agent.py comments for each):
#   --theta-direct 0.05 --theta-gates-goals   remembered/seen entries below R_i=0.05 are not goals (decayed 1-h memory: R~0.013)
#   --goal-lexical-gate                       goals pass verification's head-noun check
#   --arrival-fix                             re-plan onto current goal = arrival; goal dropped after 3 stuck resets
#   --live-verification --local-verification  only entries observed THIS episode can verify (re-observation); nearest one is checked
# VQA dropped: a second DAM-3B does not fit beside DAAAM's on the 24 GB GPU (starved captioning / OOM, see logs/_v5_*).
# Cells x 6 start poses (paired: start k is the same in every cell): fresh/a5/a60 static, fresh/a5/a60 moved, a60 moved nodecay.
# Moved = Sofa|3|3 -> (12.0,1.0), Sofa|5|2 -> (13.0,12.0) THOR xz; 4.6 / 6.5 m from the nearest remembered sofa entry. Bridge restarted before every episode.
cd "$(dirname "$0")/../.." || exit 1
cat > logs/v7.run.sh <<'INNER'
#!/bin/bash
cd __PWD__
PY=$HOME/miniconda3/envs/rpv_thor/bin/python
FLAGS="--theta-direct 0.05 --theta-gates-goals --goal-lexical-gate --arrival-fix --live-verification --local-verification"
RELOC="0:Sofa|3|3:12.0:1.01;0:Sofa|5|2:13.0:12.01"
STARTS=("15.0,3.0,150" "5.75,1.25,240" "9.75,9.75,30" "5.25,16.5,90" "8.25,14.75,240" "4.25,4.5,270")
done_ok() { d=$(ls -d data/runs/phase4_batch_*$1 2>/dev/null | head -1); [ -n "$d" ] && [ -f "$d/batch_metrics.json" ]; }
run() { # tag age_s(-1 none) reloc start extra...
  tag=$1; age=$2; reloc=$3; start=$4; shift 4
  done_ok $tag && return 0
  for attempt in 1 2 3; do
    echo "running $tag attempt $attempt $(date -Is)" > logs/v7.status
    bash scripts/thor/restart_bridge.sh || { echo "bridge restart failed before $tag" >> logs/v7.fail; sleep 30; continue; }
    rm -rf data/runs/phase4_batch_*$tag data/runs/thor_gt_$tag
    SEED=""
    if [ "$age" != "-1" ]; then
      $PY -c "import json,time,sys; d=json.load(open('data/runs/seed_alarm_tour.json')); t=time.time()-float(sys.argv[1]); [e.update(first_observed=t,last_observed=t) for e in d['entities']]; d['now']=time.time(); json.dump(d,open('data/runs/seed_v7_$tag.json','w'))" $age
      SEED="--persist-memory-only --init-memory-snapshot data/runs/seed_v7_$tag.json"
    fi
    if [ "$reloc" = 1 ]; then export THOR_RELOCATE="$RELOC" THOR_RELOCATE_CLEAR=1; else unset THOR_RELOCATE THOR_RELOCATE_CLEAR; fi
    THOR_START=$start THOR_HOUSE=1 THOR_CAM_FIX=1 THOR_GT_DIR=data/runs/thor_gt_$tag timeout 4500 $PY scripts/thor/run_thor_batch.py --no-rrd --run-tag $tag \
      --queries "a sofa" --max-steps 1500 --min-confirming-observations 1 $FLAGS $SEED "$@" > logs/thor_$tag.log 2>&1
    ncap=$(docker exec daaam_dev bash -lc "grep -c 'Created correction' /tmp/daaam_bridge_server.log")
    nst=$(grep -o 'outcome=[a-z]* reason=[a-z_]* steps=[0-9]*' logs/thor_$tag.log | grep -o '[0-9]*$')
    echo "$tag attempt=$attempt captions=$ncap steps=$nst" >> logs/v7.captions
    if [ -n "$nst" ] && [ "$nst" -gt 0 ] && [ $((ncap * 10)) -lt "$nst" ]; then  # < 0.1 captions/step: infrastructure failure
      echo "$tag attempt=$attempt captions=$ncap steps=$nst $(grep -o 'outcome=[a-z]*' logs/thor_$tag.log)" >> logs/v7.excluded
      mkdir -p logs/_v7_excluded; cp logs/thor_$tag.log logs/_v7_excluded/thor_${tag}_attempt$attempt.log
      rm -rf data/runs/phase4_batch_*$tag; continue
    fi
    done_ok $tag && return 0
    echo "FAILED attempt $attempt: $tag" >> logs/v7.fail
  done
}
for k in 0 1 2 3 4 5; do  # moved cells first (the result the paper needs), then static
  s=${STARTS[$k]}; r=s$k
  run v7_fresh_moved_$r   -1   1 $s
  run v7_a5_moved_$r      300  1 $s
  run v7_a60_moved_$r     3600 1 $s
  run v7_a60_moved_nodecay_$r 3600 1 $s --lambda-base 0
done
for k in 0 1 2 3 4 5; do
  s=${STARTS[$k]}; r=s$k
  run v7_fresh_static_$r  -1   0 $s
  run v7_a5_static_$r     300  0 $s
  run v7_a60_static_$r    3600 0 $s
done
echo "done_exit_0 $(date -Is)" > logs/v7.status
INNER
sed -i "s#__PWD__#$(pwd)#" logs/v7.run.sh; chmod +x logs/v7.run.sh
echo "queued $(date -Is)" > logs/v7.status; rm -f logs/v7.fail logs/v7.captions logs/v7.excluded
setsid nohup flock /tmp/rpv_mem_bridge.lock logs/v7.run.sh > /dev/null 2>&1 < /dev/null &
echo launched
