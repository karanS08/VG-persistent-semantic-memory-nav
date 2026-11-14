#!/bin/bash
# Detached THOR house-1 tour (builds a memory incl. the teddy bear on the bed). Query is an absent object so the tour is not cut short.
# status: logs/thor_tour.status  log: logs/thor_tour.log  out: data/runs/phase4_batch_*_tour_house1/memory_snapshot_ep0.json
cd "$(dirname "$0")/../.." || exit 1
cat > logs/thor_tour.run.sh <<EOF
#!/bin/bash
cd $(pwd)
curl -s -X POST -m 30 localhost:8765/episode/stop > /dev/null 2>&1
echo "running \$(date -Is)" > logs/thor_tour.status
THOR_HOUSE=1 THOR_CAM_FIX=1 THOR_GT_DIR=data/runs/thor_gt_tour_house1 timeout 3000 \$HOME/miniconda3/envs/rpv_thor/bin/python scripts/thor/run_thor_batch.py --no-rrd --run-tag tour_house1 \\
  --queries "a bicycle" --persist-memory-only --tour-waypoints configs/tours/thor_house1_tour.json --tour-max-steps 2500 --max-steps 100 --min-confirming-observations 1 > logs/thor_tour.log 2>&1
echo "done_exit_\$? \$(date -Is)" > logs/thor_tour.status
EOF
chmod +x logs/thor_tour.run.sh; echo "queued $(date -Is)" > logs/thor_tour.status
setsid nohup flock /tmp/rpv_mem_bridge.lock logs/thor_tour.run.sh > /dev/null 2>&1 < /dev/null &
echo launched tour
