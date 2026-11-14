"""Delete Rerun .rrd recordings except for runs cited in the paper.
Keep = scripts/postdraft/audit_registry.GROUPS runs (every paper number; also
what build_blind_audit_package.py reads frames from) + the 2x2 factorial runs.
Metrics (batch_metrics.json) and decision logs are never touched.
Usage: python scripts/prune_rrd.py            # dry run
       python scripts/prune_rrd.py --delete   # delete, log to logs/rrd_deleted_<date>.txt"""
import sys, time
from pathlib import Path
sys.path.insert(0, "scripts/postdraft")
from audit_registry import GROUPS

keep = {f"phase4_batch_{short}" for g in GROUPS.values() for _, short, _ in g}
keep |= {f"phase4_batch_{s}" for s in (
    "reset5_rep2", "reset5_rep3", "persistent_continue5_rep2", "persistent_continue5_rep3",
    *(f"persist_memory_only_persistmemoryonly5_rep{r}" for r in (1, 2, 3)),
    *(f"persist_position_only_persistpositiononly5_rep{r}" for r in (1, 2, 3)))}

drop = [p for p in Path("data/runs").glob("*/*.rrd") if p.parent.name not in keep]
kept = [p for p in Path("data/runs").glob("*/*.rrd") if p.parent.name in keep]
gb = lambda ps: sum(p.stat().st_size for p in ps) / 1e9
print(f"keep {len(kept)} files ({gb(kept):.0f} GB) in {len({p.parent for p in kept})} runs; "
      f"delete {len(drop)} files ({gb(drop):.0f} GB) in {len({p.parent for p in drop})} runs")
missing = sorted(keep - {p.parent.name for p in kept})
if missing:
    print("keep-list runs with no .rrd on disk:", missing)
if "--delete" in sys.argv:
    log = Path(f"logs/rrd_deleted_{time.strftime('%Y-%m-%d')}.txt")
    with log.open("a") as f:
        for p in drop:
            f.write(f"{p}\n")
            p.unlink()
    print(f"deleted; list appended to {log}")
