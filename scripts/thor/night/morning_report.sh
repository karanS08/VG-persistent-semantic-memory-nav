#!/bin/bash
# Detached: waits for the night queue to finish (or 08:00 cap), scores every test, writes a mechanical report.
cd ~/mem_ws
PY=$HOME/miniconda3/envs/rpv_thor/bin/python
OUT=results/post_paper/night_2026-11-02_auto_report.md
until [ -f logs/night/chain5_done ] || [ "$(date +%H%M)" -ge 1100 -a "$(date +%H%M)" -lt 1300 ]; do sleep 60; done
sleep 30
{
echo "# Night queue 2026-11-02: automatic report ($(date))"
echo; echo "Queue status: $(cat logs/night/status)"
echo; echo "## Job outcomes"; echo '```'
cut -f2- logs/night/queue.tsv | awk -F'\t' '{print $NF==""?$0:$0}' | grep -oE "OK|EXCLUDED_infrastructure|KILLED_[a-z_]+|FAILED_3_ATTEMPTS|SKIPPED_[a-z_]+|POST_FAILED_rc[0-9]+|NO_RESULT|QUEUE_EXCEPTION|BRIDGE_RESTART_FAILED" | sort | uniq -c
echo; grep -vE "	OK	|POST	OK$|SKIPPED_requires" logs/night/queue.tsv
echo '```'
echo; echo "## Seed checks (test 3)"; echo '```'
for f in logs/night/seed_check_house*.json; do $PY -c "import json;r=json.load(open('$f'));print(r['house'],'entities',r['n_entities_positioned'],'sofa entries',r['n_sofa_entries'],'problems',r['problems']);[print('   ',e['id'],'d_orig',e['dist_to_orig'],'d_new',e['dist_to_new'],'mob',e['mobility_lookup'],e['caption'][:60]) for e in r['sofa_entries']]"; done
ls logs/night/ready_house* 2>/dev/null
echo '```'
echo; echo "## Scores"
for V in t2 t1l0 t1l018 t1l055 h72 h99 h314 h72v8 h99v8; do
  if ls logs/thor_${V}_*_s[0-9].log >/dev/null 2>&1; then echo; echo "### $V"; echo '```'; $PY scripts/thor/score_thor.py $V 2>&1 | tail -40; echo '```'; fi
done
} > $OUT 2>&1
echo done > logs/night/morning_report_done
