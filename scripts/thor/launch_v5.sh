#!/bin/bash
# Moved-object experiment v5 (2026-10-30). Fixes found in v4 diagnosis:
#  (1) seed memory re-stamped to an EXACT age right before each run (v4's absolute timestamps drifted to ~25 h);
#  (2) --theta-direct 0.05: between stale-with-decay R_i (<=0.014 at 1 h) and a fresh sighting (>=0.077), so the robot
#      actually navigates to a target it has seen/remembered (theta=0.2 is never reached: R_i peaks ~0.12);
#  (3) --vqa-verification: DAM-3B confirms on the CURRENT frame, so an empty remembered spot cannot "verify";
#  (4) --verification-feedback: a VQA rejection discounts v_i (x0.5) and holds the entry out until re-observed
#      (smoke 2 showed 33 rejections of one stale entry, then a single VQA false 'yes').
#  (5) --theta-gates-goals: theta_direct otherwise only LABELS the branch; entries below it are now not taken as goals;
#  (6) --goal-lexical-gate: goals pass verification's own head-noun check (ungated goals chased trash cans/cushions);
#  (7) --arrival-fix: re-planning onto the goal the agent already stands at counts as arrival (smoke 3: entity 109
#      re-picked 42 times in place).
#  (8) --local-verification: VQA judges the best candidate within verify_distance, not only the global top entry
#      (smoke 4: robot stood at sofa A's remembered entry 114 but only entry 113, elsewhere, was allowed to verify);
#      arrival_fix also excludes a goal after 3 stuck resets (smoke 4: 320 steps blocked by a THOR collision).
# Query "a sofa", ProcTHOR house 1; moved = both sofas teleported (THOR (14.75,15.5),(15.25,11.5)) >=7.9 m from every remembered
# sofa entry, so VQA (frame-level) cannot see the moved sofa from an old remembered spot (the v5 first smoke had a sofa dropped 2 m from entry 113).
# Cells (6 reps): a5_static a5_moved (memory 5 min old) | a60_static a60_moved a60_moved_nodecay (1 h) | fresh_static fresh_moved.
# Smoke gate first: a5_static_r1 + a60_moved_nodecay_r1 must show [VQA] lines; static must succeed. status: logs/v5.status
cd "$(dirname "$0")/../.." || exit 1
cat > logs/v5.run.sh <<'INNER'
#!/bin/bash
cd __PWD__
PY=$HOME/miniconda3/envs/rpv_thor/bin/python
RELOC="0:Sofa|3|3:14.75:15.5;0:Sofa|5|2:15.25:11.5"
done_ok() { d=$(ls -d data/runs/phase4_batch_*$1 2>/dev/null | head -1); [ -n "$d" ] && [ -f "$d/batch_metrics.json" ]; }
run() { # tag age_s(-1 = no seed) reloc(0/1) extra-args...
  tag=$1; age=$2; reloc=$3; shift 3
  done_ok $tag && return 0
  for attempt in 1 2; do
    curl -s -X POST -m 30 localhost:8765/episode/stop > /dev/null 2>&1
    rm -rf data/runs/phase4_batch_*$tag data/runs/thor_gt_$tag
    echo "running $tag attempt $attempt $(date -Is)" > logs/v5.status
    SEED=""
    if [ "$age" != "-1" ]; then
      $PY -c "import json,time,sys; d=json.load(open('data/runs/seed_alarm_tour.json')); t=time.time()-float(sys.argv[1]); [e.update(first_observed=t,last_observed=t) for e in d['entities']]; d['now']=time.time(); json.dump(d,open('data/runs/seed_v5_$tag.json','w'))" $age
      SEED="--persist-memory-only --init-memory-snapshot data/runs/seed_v5_$tag.json"
    fi
    if [ "$reloc" = 1 ]; then export THOR_RELOCATE="$RELOC"; else unset THOR_RELOCATE; fi
    THOR_HOUSE=1 THOR_CAM_FIX=1 THOR_GT_DIR=data/runs/thor_gt_$tag timeout 2400 $PY scripts/thor/run_thor_batch.py --no-rrd --run-tag $tag \
      --queries "a sofa" --max-steps 500 --min-confirming-observations 1 --theta-direct 0.05 --theta-gates-goals --goal-lexical-gate --arrival-fix --vqa-verification --local-verification --verification-feedback $SEED "$@" > logs/thor_$tag.log 2>&1
    done_ok $tag && return 0
    echo "FAILED attempt $attempt: $tag" >> logs/v5.fail
  done
}
# ---- smoke gate ----
run v5_a5_static_r1 300 0
run v5_a60_moved_nodecay_r1 3600 1 --lambda-base 0
if ! grep -q "\[VQA\]" logs/thor_v5_a5_static_r1.log logs/thor_v5_a60_moved_nodecay_r1.log; then echo "STOPPED smoke: no VQA calls $(date -Is)" > logs/v5.status; exit 1; fi
if ! grep -q "^outcome=success" logs/thor_v5_a5_static_r1.log; then echo "STOPPED smoke: static 5-min run did not succeed $(date -Is)" > logs/v5.status; exit 1; fi
if grep -q "VERIFIED by entity 9000" logs/thor_v5_a60_moved_nodecay_r1.log; then echo "STOPPED smoke: moved run verified an OLD memory entry (confound) $(date -Is)" > logs/v5.status; exit 1; fi
echo "smoke passed $(date -Is)" >> logs/v5.smoke
# ---- matrix ----
for rep in 1 2 3 4 5 6; do
  run v5_a5_static_r$rep        300  0
  run v5_a5_moved_r$rep         300  1
  run v5_a60_static_r$rep       3600 0
  run v5_a60_moved_r$rep        3600 1
  run v5_a60_moved_nodecay_r$rep 3600 1 --lambda-base 0
  run v5_fresh_static_r$rep     -1   0
  run v5_fresh_moved_r$rep      -1   1
done
echo "done_exit_0 $(date -Is)" > logs/v5.status
INNER
sed -i "s#__PWD__#$(pwd)#" logs/v5.run.sh; chmod +x logs/v5.run.sh
echo "queued $(date -Is)" > logs/v5.status; rm -f logs/v5.fail logs/v5.smoke
setsid nohup flock /tmp/rpv_mem_bridge.lock logs/v5.run.sh > /dev/null 2>&1 < /dev/null &
echo launched
