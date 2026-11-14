#!/bin/bash
# 2026-10-03 demo (author request): memory-on search for the radiator in the far
# lower-right (south-east) corner of 00770, with memory seeded from the robot's OWN
# earlier observation of it (house-tour memory entry 1170, real DAAAM caption +
# embedding + position), recorded as seen 10 min before the search. Memory-off searches
# from empty memory (1500-step budget). Instance gate: only that radiator counts
# (1.2 m radius; nearest other radiator, entry 1161, is 1.7 m away).
# --scan-steps 30 (demo only, both conditions): look around ~15 s on arrival so the
# ~13 s-late DAAAM caption arrives before the robot moves on (first attempt with the
# default 6 kept in data/runs/_seeded_rad_on_scan6).
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate rpv_mem
source scripts/_driver_lib.sh
unset RPV_NO_RRD
SEED=data/runs/seed_radiator_se.json
python - <<'PY'
import json, time
t = json.load(open("data/runs/phase4_batch_persist_memory_only_house_tour/memory_snapshot_ep0.json"))
e = dict(next(x for x in t["entities"] if x["id"] == 1170))
now = time.time()
e["first_observed"] = e["last_observed"] = now - 600      # seen 10 minutes before the search
json.dump({"_doc": "seed memory: tour entry 1170 (SE-corner radiator), last seen 10 min before the demo search",
           "now": now, "entities": [e]}, open("data/runs/seed_radiator_se.json", "w"))
print("seed written:", e["description"][:60], e["position_world"])
PY
Q='a wall-mounted radiator'
T='{"query": "a wall-mounted radiator", "xyz": [7.98, 1.32, 14.85], "radius": 1.2}'
run_batch phase4_batch_persist_memory_only_seeded_rad_on --max-steps 500 --min-confirming-observations 1 \
    --queries "$Q" --persist-memory-only --init-memory-snapshot $SEED --instance-target "$T" --scan-steps 30 --run-tag seeded_rad_on
run_batch phase4_batch_seeded_rad_off --max-steps 1500 --min-confirming-observations 1 \
    --queries "$Q" --instance-target "$T" --scan-steps 30 --run-tag seeded_rad_off
python scripts/check_same_object.py data/runs/phase4_batch_persist_memory_only_seeded_rad_on data/runs/phase4_batch_seeded_rad_off
log "SEEDED RADIATOR DONE"
