#!/bin/bash
# Detached Habitat A/B on 00861: as-pushed (HAB_CAM_FIX=0) vs convention-fixed (=1), then entity-level GT scoring.
# status: logs/habab.status  logs: logs/habab_*.log   result: results/post_paper/habab_gt861.json
cd "$(dirname "$0")/../.." || exit 1
cat > logs/habab.run.sh <<'EOF'
#!/bin/bash
cd __PWD__
PY=$HOME/miniconda3/envs/rpv_mem/bin/python
Q="a chair,a mirror,a lamp,a pillow,a toilet,a sink,a bed"
SC=data/hm3d/example/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb
for fix in 0 1; do
  echo "running fix=$fix $(date -Is)" > logs/habab.status
  HAB_CAM_FIX=$fix $PY scripts/hab_cam_ab/run_habitat_ab.py --no-rrd --run-tag habab_fix$fix --scene-path $SC --queries "$Q" --max-steps 300 --min-confirming-observations 1 > logs/habab_fix$fix.log 2>&1
done
echo "scoring $(date -Is)" > logs/habab.status
$PY scripts/hab_cam_ab/score_entities_gt861.py results/post_paper/habab_gt861.json data/runs/phase4_batch_habab_fix0 data/runs/phase4_batch_habab_fix1 > logs/habab_score.log 2>&1
echo "done_exit_$? $(date -Is)" > logs/habab.status
EOF
sed -i "s#__PWD__#$(pwd)#" logs/habab.run.sh; chmod +x logs/habab.run.sh
echo "queued $(date -Is)" > logs/habab.status
setsid nohup flock /tmp/rpv_mem_bridge.lock logs/habab.run.sh > /dev/null 2>&1 < /dev/null &
echo launched habab
