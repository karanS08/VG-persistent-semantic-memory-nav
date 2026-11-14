#!/bin/bash
# THOR moved-object matrix for a pipeline variant: identical to scripts/thor/launch_v7.sh (same house, sofa placements,
# 6 start poses, 7 cells, 1500-step budget, seed memory, base flags, pre-set infrastructure-exclusion rule, bridge
# restart per episode) plus EXTRA runner flags. Usage:
#   scripts/thor/launch_variant.sh v8  "--fuse-positions --lexical-mode caption_head --revisit-evidence --image-check"
# Logs: logs/<VER>.status, logs/thor_<VER>_*.log, logs/<VER>.excluded; score: python scripts/thor/score_thor.py <VER>
VER=$1; EXTRA=$2
[ -n "$VER" ] || { echo "usage: $0 VER 'extra flags'"; exit 1; }
cd "$(dirname "$0")/../.." || exit 1
cat > logs/$VER.run.sh <<'INNER'
#!/bin/bash
cd __PWD__
PY=$HOME/miniconda3/envs/rpv_thor/bin/python
export PATH=$HOME/miniconda3/envs/rpv_thor/bin:$PATH  # git-lfs for prior.load_dataset
FLAGS="--theta-direct 0.05 --theta-gates-goals --goal-lexical-gate --arrival-fix --live-verification --local-verification __EXTRA__"
RELOC="0:Sofa|3|3:12.0:1.01;0:Sofa|5|2:13.0:12.01"
STARTS=("15.0,3.0,150" "5.75,1.25,240" "9.75,9.75,30" "5.25,16.5,90" "8.25,14.75,240" "4.25,4.5,270")
done_ok() { d=$(ls -d data/runs/phase4_batch_*$1 2>/dev/null | head -1); [ -n "$d" ] && [ -f "$d/batch_metrics.json" ]; }
run() { # tag age_s(-1 none) reloc start extra...
  tag=$1; age=$2; reloc=$3; start=$4; shift 4
  done_ok $tag && return 0
  for attempt in 1 2 3; do
    echo "running $tag attempt $attempt $(date -Is)" > logs/__VER__.status
    bash scripts/thor/restart_bridge.sh || { echo "bridge restart failed before $tag" >> logs/__VER__.fail; sleep 30; continue; }
    rm -rf data/runs/phase4_batch_*$tag data/runs/thor_gt_$tag
    SEED=""
    if [ "$age" != "-1" ]; then
      $PY -c "import json,time,sys; d=json.load(open('data/runs/seed_alarm_tour.json')); t=time.time()-float(sys.argv[1]); [e.update(first_observed=t,last_observed=t) for e in d['entities']]; d['now']=time.time(); json.dump(d,open('data/runs/seed___VER___$tag.json','w'))" $age
      SEED="--persist-memory-only --init-memory-snapshot data/runs/seed___VER___$tag.json"
    fi
    if [ "$reloc" = 1 ]; then export THOR_RELOCATE="$RELOC" THOR_RELOCATE_CLEAR=1; else unset THOR_RELOCATE THOR_RELOCATE_CLEAR; fi
    THOR_START=$start THOR_HOUSE=1 THOR_CAM_FIX=1 THOR_GT_DIR=data/runs/thor_gt_$tag timeout 4500 $PY scripts/thor/run_thor_batch.py --no-rrd --run-tag $tag \
      --queries "a sofa" --max-steps 1500 --min-confirming-observations 1 $FLAGS $SEED "$@" > logs/thor_$tag.log 2>&1
    ncap=$(docker exec daaam_dev bash -lc "grep -c 'Created correction' /tmp/daaam_bridge_server.log")
    nst=$(grep -o 'outcome=[a-z]* reason=[a-z_]* steps=[0-9]*' logs/thor_$tag.log | grep -o '[0-9]*$')
    echo "$tag attempt=$attempt captions=$ncap steps=$nst" >> logs/__VER__.captions
    if [ -n "$nst" ] && [ "$nst" -gt 0 ] && [ $((ncap * 10)) -lt "$nst" ]; then  # < 0.1 captions/step: infrastructure failure
      echo "$tag attempt=$attempt captions=$ncap steps=$nst $(grep -o 'outcome=[a-z]*' logs/thor_$tag.log)" >> logs/__VER__.excluded
      mkdir -p logs/___VER___excluded; cp logs/thor_$tag.log logs/___VER___excluded/thor_${tag}_attempt$attempt.log
      rm -rf data/runs/phase4_batch_*$tag; continue
    fi
    done_ok $tag && return 0
    echo "FAILED attempt $attempt: $tag" >> logs/__VER__.fail
  done
}
for k in 0 1 2 3 4 5; do  # moved cells first (the result the paper needs), then static
  s=${STARTS[$k]}; r=s$k
  run __VER___fresh_moved_$r   -1   1 $s
  run __VER___a5_moved_$r      300  1 $s
  run __VER___a60_moved_$r     3600 1 $s
  run __VER___a60_moved_nodecay_$r 3600 1 $s --lambda-base 0
done
for k in 0 1 2 3 4 5; do
  s=${STARTS[$k]}; r=s$k
  run __VER___fresh_static_$r  -1   0 $s
  run __VER___a5_static_$r     300  0 $s
  run __VER___a60_static_$r    3600 0 $s
done
echo "done_exit_0 $(date -Is)" > logs/__VER__.status
INNER
sed -i "s#__PWD__#$(pwd)#; s#__VER__#$VER#g; s#__EXTRA__#$EXTRA#" logs/$VER.run.sh; chmod +x logs/$VER.run.sh
echo "queued $(date -Is)" > logs/$VER.status; rm -f logs/$VER.fail logs/$VER.captions logs/$VER.excluded
setsid nohup flock /tmp/rpv_mem_bridge.lock logs/$VER.run.sh > /dev/null 2>&1 < /dev/null &
echo "launched $VER"
