#!/bin/bash
cd ~/mem_ws
until [ -f logs/night/chain3_done ]; do sleep 30; done
scripts/thor/night/night4_chain.sh
