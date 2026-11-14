#!/bin/bash
cd ~/mem_ws
until [ -f logs/night/chain_done ]; do sleep 30; done
scripts/thor/night/night3_chain.sh
