#!/bin/bash
# 2026-09-29: relaunch the priority queue with .rrd recording off
# (RPV_NO_RRD=1 -> run_phase4_batch --no-rrd default). Resumable: finished
# runs are skipped. Lag-window batches import worktree code, which ignores
# RPV_NO_RRD, so those 3 batches still record (~17 GB).
cd /home/ksharma/mem_ws || exit 1
RPV_NO_RRD=1 nohup setsid bash scripts/postdraft/run_priority_queue.sh \
    >> logs/priority_queue_driver.log 2>&1 < /dev/null &
