#!/bin/bash
# Detached moving-object experiment v2 (ProcTHOR house 1). Targets are unique in the house AND by caption noun:
#   tb = TeddyBear|surface|8|76 ("a teddy bear", mobile category)  moved from the bedroom bed to ~(14.0,15.0)
#   wm = WashingMachine|11|3   ("a washing machine", static prior 0.02) moved from the bathroom to ~(4.0,5.0)
# Each run = 2 queries (same query twice), --persist-memory-only (memory carries over, env rebuilt per query); the object moves before query 2;
# memory age between queries injected (3600 s). Conditions: full (decay on) | nodecay (--lambda-base 0) | memoff (live-only) | static (no move).
# status: logs/movedobj.status   per-run logs: logs/thor_<tag>.log   score: python3 scripts/thor/score_movedobj.py
cd "$(dirname "$0")/../.." || exit 1
cat > logs/movedobj.run.sh <<'EOF'
#!/bin/bash
cd __PWD__
PY=$HOME/miniconda3/envs/rpv_thor/bin/python
run() { # tag object-id query relocate-xz(or "") extra-args...
  tag=$1; oid=$2; q=$3; xz=$4; shift 4
  ls -d data/runs/phase4_batch_*_${tag} > /dev/null 2>&1 && [ -f "$(ls -d data/runs/phase4_batch_*_${tag} | head -1)/batch_metrics.json" ] && return 0
  echo "running $tag $(date -Is)" > logs/movedobj.status
  if [ -n "$xz" ]; then export THOR_RELOCATE="1:${oid}:${xz}"; else unset THOR_RELOCATE; fi
  THOR_HOUSE=1 THOR_CAM_FIX=1 THOR_GT_DIR=data/runs/thor_gt_${tag} $PY scripts/thor/run_thor_batch.py --no-rrd --run-tag ${tag} \
    --persist-memory-only --queries "${q},${q}" --max-steps 500 --min-confirming-observations 1 --inject-memory-age-s 3600 "$@" > logs/thor_${tag}.log 2>&1
}
for rep in 1 2 3; do
  for tgt in tb wm; do
    if [ $tgt = tb ]; then oid="TeddyBear|surface|8|76"; q="a teddy bear"; xz="14.0:15.0"; else oid="WashingMachine|11|3"; q="a washing machine"; xz="4.0:5.0"; fi
    run ${tgt}_full_r$rep    "$oid" "$q" "$xz"
    run ${tgt}_static_r$rep  "$oid" "$q" ""
    run ${tgt}_nodecay_r$rep "$oid" "$q" "$xz" --lambda-base 0
    run ${tgt}_memoff_r$rep  "$oid" "$q" "$xz" --memory-off
  done
done
echo "done_exit_0 $(date -Is)" > logs/movedobj.status
EOF
sed -i "s#__PWD__#$(pwd)#" logs/movedobj.run.sh; chmod +x logs/movedobj.run.sh
echo "queued $(date -Is)" > logs/movedobj.status
setsid nohup flock /tmp/rpv_mem_bridge.lock logs/movedobj.run.sh > /dev/null 2>&1 < /dev/null &
echo launched movedobj v2
