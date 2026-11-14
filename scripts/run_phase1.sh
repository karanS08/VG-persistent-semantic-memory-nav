#!/usr/bin/env bash
# Phase 1: scripted-walk Habitat-Sim episode, streamed to Rerun.
# Restartable, safe to run detached (tmux/nohup) per CLAUDE.md Section 1.
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rpv_mem

export DISPLAY=
python -m orchestrator.run_phase1 "$@"
