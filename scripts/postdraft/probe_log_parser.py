"""Parse the per-decision memory-influence probe lines that
packages/orchestrator/orchestrator/run_phase4_batch.py prints to stdout, e.g.

    [step 145] contextual-anchor eval: live=32 remembered=34 value_with=0.0853 value_without=0.0869 influenced=True

These stdout lines are the ONLY per-decision record for runs made before
2026-09-28 (the decisions_ep*.jsonl logger did not exist yet). Offline, read-only.

Log sources (verified by inspection 2026-09-28):
  * one log per batch:  logs/phase4_<out_dir_name>.log       (runs launched via scripts/_driver_lib.sh)
  * multi-batch logs:   logs/phase4_reset_vs_continue.log     (reset5 + persistent_continue5, 2026-09-02)
                        logs/phase4_final_minconf1default.log (RQ3-figure runs, 2026-09-02)
    Multi-batch logs are split on the line "saved metrics to data/runs/<dir>/batch_metrics.json",
    which terminates each batch's section.
"""
import re
from pathlib import Path

EVAL_RE = re.compile(
    r"\[step (\d+)\] contextual-anchor eval: live=(\d+) remembered=(\d+) "
    r"value_with=([0-9.]+) value_without=([0-9.]+) influenced=(True|False)")
EP_RE = re.compile(r"^=== (?:Episode|Query) (\d+)[^:]*: query='([^']*)' ===")
SAVED_RE = re.compile(r"saved metrics to (data/runs/[^/]+)/batch_metrics.json")
PROBE_SUM_RE = re.compile(r"memory probe: influenced (\d+)/(\d+)")

ROOT = Path(__file__).resolve().parents[2]


def _split_sections(text):
    """Yield (run_dir_name or None, lines) sections terminated by 'saved metrics to'."""
    cur = []
    for line in text.splitlines():
        cur.append(line)
        m = SAVED_RE.search(line)
        if m:
            yield Path(m.group(1)).name, cur
            cur = []
    if cur:
        yield None, cur


def parse_section(lines):
    """-> {episode_id: {"query":..., "evals":[...], "probe_summary":(infl, eval)}}"""
    eps = {}
    cur = None
    for line in lines:
        m = EP_RE.search(line.strip())
        if m:
            cur = int(m.group(1))
            eps[cur] = {"query": m.group(2), "evals": [], "probe_summary": None}
            continue
        if cur is None:
            continue
        m = EVAL_RE.search(line)
        if m:
            eps[cur]["evals"].append({
                "step": int(m.group(1)), "live": int(m.group(2)), "remembered": int(m.group(3)),
                "value_with": float(m.group(4)), "value_without": float(m.group(5)),
                "influenced": m.group(6) == "True",
            })
            continue
        m = PROBE_SUM_RE.search(line)
        if m:
            eps[cur]["probe_summary"] = (int(m.group(1)), int(m.group(2)))
    return eps


def load_run_evals(run_dir_name, log_path=None):
    """Return parsed episodes for one batch dir name, locating its log section."""
    if log_path is None:
        log_path = ROOT / "logs" / f"phase4_{run_dir_name}.log"
    text = Path(log_path).read_text(errors="replace")
    sections = list(_split_sections(text))
    named = [s for s in sections if s[0] == run_dir_name]
    if named:
        return parse_section(named[-1][1])
    # single-batch log that ended without the 'saved metrics' line (should not happen)
    if len(sections) == 1:
        return parse_section(sections[0][1])
    raise ValueError(f"no section for {run_dir_name} in {log_path}")


def classify_eval(e):
    """Mutually exclusive probe cases, derived exactly from agent.py semantics:
      memory_influenced = (best_id != live_id), where live_id is None when the
      live-only _select_target returns None, and value_without is then printed as 0.0.
    """
    if not e["influenced"]:
        return "not_influenced"
    if e["value_without"] == 0.0:
        return "influenced_live_empty_live0" if e["live"] == 0 else "influenced_live_empty_liveN_noqualifying"
    return "influenced_target_changed"
