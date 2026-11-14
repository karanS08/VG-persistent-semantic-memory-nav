#!/bin/bash
# Corrected test 3: waits for the first queue to release the bridge lock, then runs configs/thor_night2/jobs4.json.
cd ~/mem_ws
export PATH=$HOME/miniconda3/envs/rpv_thor/bin:$PATH
flock /tmp/rpv_mem_bridge.lock $HOME/miniconda3/envs/rpv_thor/bin/python scripts/thor/night/run_queue.py configs/thor_night2/jobs4.json >> logs/night/runner4.out 2>&1
echo "$(date)" > logs/night/chain4_done
