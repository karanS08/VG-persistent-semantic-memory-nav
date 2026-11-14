#!/bin/bash
# Restart the DAAAM HTTP bridge (:8765) and the stand-alone VQA verify server (:8766) inside daaam_dev, killing
# leftover grounding-worker children (each holds a DAM-3B copy, ~9 GB), and wait until both answer /health.
# 2026-10-30 v5 diagnosis: VQA loaded inside the bridge stopped grounding updates reaching /updates, and killed
# bridges leaked their worker children, so VQA runs restart both servers before every episode.
docker exec daaam_dev bash -lc "pkill -f '[d]aaam_bridge_server.py'; pkill -f '[d]aaam_verify_server.py'; pkill -f '[m]ultiprocessing.spawn'; pkill -f '[m]ultiprocessing.resource_tracker'" ; sleep 5
docker exec -d daaam_dev bash -lc "cd /home/daaam/ros2_ws/src/daaam && nohup python3 /tmp/daaam_bridge_server.py > /tmp/daaam_bridge_server.log 2>&1"
docker exec -d daaam_dev bash -lc "cd /home/daaam/ros2_ws/src/daaam && nohup python3 /tmp/daaam_verify_server.py > /tmp/daaam_verify_server.log 2>&1"
for i in $(seq 1 90); do
  curl -s -m 5 localhost:8765/health | grep -q ok && curl -s -m 5 localhost:8766/health | grep -q ok && exit 0; sleep 2
done
echo "bridge/verify server did not come back" >&2; exit 1
