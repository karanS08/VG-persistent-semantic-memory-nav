#!/bin/bash
# Added 2026-11-03 01:25 (mid-run, disclosed): tour jobs were wrongly subject to the search-episode caption-rate
# exclusion rule (tour_house99: 293 captions / 3000 steps, deterministic, bridge healthy). The seed gate for tours is
# validate_seed.py; run it on attempt 3's snapshot as soon as it is written, before the queue reaches the h99 jobs.
cd ~/mem_ws
D=data/runs/phase4_batch_persist_memory_only_tour_house99
for i in $(seq 1 240); do
  if [ -f $D/memory_snapshot_ep0.json ]; then sleep 5
    $HOME/miniconda3/envs/rpv_thor/bin/python scripts/thor/night/validate_seed.py 99 tour_house99 > logs/night/late_validate_house99.txt 2>&1
    echo "rc=$? $(date)" >> logs/night/late_validate_house99.txt; exit 0; fi
  sleep 10
done
echo "timeout $(date)" > logs/night/late_validate_house99.txt
