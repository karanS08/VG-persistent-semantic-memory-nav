#!/bin/bash
# Usage: launch_detached.sh <tag> <THOR_HOUSE> <THOR_CAM_FIX 0|1> <run_phase4_batch args...>
# Fully detached (setsid+nohup): survives SSH/Claude-session loss.
#   status: logs/thor_<tag>.status   log: logs/thor_<tag>.log   script: logs/thor_<tag>.run.sh
# Takes the shared bridge lock (flock) so runs queue instead of overlapping on the single DAAAM bridge.
cd "$(dirname "$0")/../.." || exit 1
tag=$1; house=$2; fix=$3; shift 3
args=$(printf '%q ' "$@")
cat > logs/thor_${tag}.run.sh <<EOF
#!/bin/bash
cd $(pwd)
echo "running \$(date -Is)" > logs/thor_${tag}.status
THOR_HOUSE=$house THOR_CAM_FIX=$fix THOR_GT_DIR=data/runs/thor_gt_${tag} \\
  \$HOME/miniconda3/envs/rpv_thor/bin/python scripts/thor/run_thor_batch.py --no-rrd --run-tag ${tag} $args > logs/thor_${tag}.log 2>&1
echo "done_exit_\$? \$(date -Is)" > logs/thor_${tag}.status
EOF
chmod +x logs/thor_${tag}.run.sh
echo "queued $(date -Is)" > logs/thor_${tag}.status
setsid nohup flock /tmp/rpv_mem_bridge.lock logs/thor_${tag}.run.sh > /dev/null 2>&1 < /dev/null &
echo "launched $tag (house $house, cam_fix $fix)"
