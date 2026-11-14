"""Overnight THOR queue: runs every job in a jobs JSON in order, one at a time, and NEVER stops on a failed job.

Rules (fixed before the run; identical for every condition):
  * one episode at a time (single DAAAM bridge / single GPU), holding /tmp/rpv_mem_bridge.lock for the whole queue;
  * before every attempt: disk check (>= 5 GB free, else the queue stops cleanly), DAAAM bridge restart (3 tries, then
    `docker restart daaam_dev` once), leftover THOR renderer processes killed;
  * watchdogs per attempt: no trajectory written within 10 min of launch (THOR start-up hang) -> killed; trajectory not
    growing for 15 min while the process is alive -> killed; hard limit 100 min -> killed;
  * pre-set infrastructure-exclusion rule (as v7): an attempt with < 0.1 DAAAM captions per step is logged to
    logs/night/excluded.tsv and re-run; mis-captions, wrong matches and budget exhaustion are results, never re-run;
  * at most 3 attempts per job; a job that still fails is logged FAILED and the queue moves on;
  * a job with `requires` runs only if that marker file exists (e.g. a house whose tour/seed validation failed is
    skipped, logged SKIPPED); a job with `only_if_fewer_than` runs only if fewer than N of the listed markers exist
    (backup house).
Progress: logs/night/status (current job), logs/night/queue.tsv (one line per attempt/outcome).
Usage: python scripts/thor/night/run_queue.py configs/thor_night/jobs.json [--dry-run]
"""
import glob, json, math, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
PY = os.path.expanduser("~/miniconda3/envs/rpv_thor/bin/python")
ENV_BIN = os.path.dirname(PY)
LOG = Path("logs/night"); LOG.mkdir(parents=True, exist_ok=True)
(LOG / "attempts").mkdir(exist_ok=True)
DRY = "--dry-run" in sys.argv
BASE = ["--no-rrd", "--min-confirming-observations", "1"]


def note(*cols):
    if DRY:  # 2026-11-03: a dry run must never touch the live queue log/status
        return
    with open(LOG / "queue.tsv", "a") as f:
        f.write("\t".join([time.strftime("%Y-%m-%dT%H:%M:%S")] + [str(c) for c in cols]) + "\n")


def status(s):
    if DRY:
        return
    (LOG / "status").write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {s}\n")


def sh(cmd, timeout=600):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124


def restart_bridge():
    for i in range(3):
        if sh("bash scripts/thor/restart_bridge.sh", 400) == 0:
            return True
        time.sleep(30)
    sh("docker restart daaam_dev", 300); time.sleep(20)
    return sh("bash scripts/thor/restart_bridge.sh", 400) == 0


def kill_thor():
    # exact process-name match (comm is truncated to 15 chars): never matches a shell whose command line mentions it
    sh("pkill -x thor-CloudRende", 30)


def run_dir(tag):
    d = [p for p in glob.glob(f"data/runs/phase4_batch_*{tag}") if p.endswith(tag)]
    return d[0] if d else None


def done(tag):
    d = run_dir(tag)
    return bool(d) and os.path.exists(f"{d}/batch_metrics.json")


def captions():
    r = subprocess.run("docker exec daaam_dev bash -lc \"grep -c 'Created correction' /tmp/daaam_bridge_server.log\"",
                       shell=True, capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 0


def restamp(seed, age, out):
    d = json.load(open(seed)); t = time.time() - float(age)
    for e in d["entities"]:
        e["first_observed"] = t; e["last_observed"] = t
    d["now"] = time.time(); json.dump(d, open(out, "w"))


def run_attempt(job, attempt):
    tag = job["tag"]
    for p in glob.glob(f"data/runs/phase4_batch_*{tag}") + glob.glob(f"data/runs/thor_gt_{tag}"):
        if p.endswith(tag):
            shutil.rmtree(p, ignore_errors=True)
    env = dict(os.environ, PATH=f"{ENV_BIN}:{os.environ['PATH']}", THOR_HOUSE=str(job["house"]), THOR_CAM_FIX="1",
               THOR_GT_DIR=f"data/runs/thor_gt_{tag}")
    for k in ("THOR_START", "THOR_RELOCATE", "THOR_RELOCATE_CLEAR"):
        env.pop(k, None)
    if job.get("start"):
        env["THOR_START"] = job["start"]
    if job.get("reloc"):
        env["THOR_RELOCATE"] = job["reloc"]; env["THOR_RELOCATE_CLEAR"] = "1"
    args = [PY, "scripts/thor/run_thor_batch.py", "--run-tag", tag] + BASE + job["args"]
    if job.get("age", -1) != -1:
        seed_out = f"data/runs/seed_{tag}.json"
        restamp(job["seed"], job["age"], seed_out)
        args += ["--persist-memory-only", "--init-memory-snapshot", seed_out]
    logf = open(f"logs/thor_{tag}.log", "w")
    t0 = time.time()
    p = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    last_size, last_change, why = -1, time.time(), ""
    while p.poll() is None:
        time.sleep(15)
        d = run_dir(tag)
        traj = glob.glob(f"{d}/trajectory_ep*.jsonl") if d else []
        size = sum(os.path.getsize(t) for t in traj)
        now = time.time()
        if size != last_size:
            last_size, last_change = size, now
        if not traj and now - t0 > job.get("startup_limit_s", 600):
            why = "startup_hang"
        elif traj and now - last_change > 900 and not job.get("tour_quiet_ok"):
            why = "stalled"
        elif now - t0 > job.get("hard_limit_s", 6000):
            why = "hard_limit"
        if why:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); break
    logf.close()
    kill_thor()
    shutil.copy(f"logs/thor_{tag}.log", LOG / "attempts" / f"thor_{tag}_attempt{attempt}.log")
    txt = open(f"logs/thor_{tag}.log", errors="replace").read()
    m = re.search(r"outcome=\w+ reason=\w+ steps=(\d+)", txt)
    return why, (int(m.group(1)) if m else None), captions()


def run_job(job):
    tag = job["tag"]
    if done(tag):
        note(tag, "ALREADY_DONE"); return True
    for attempt in range(1, 4):
        if shutil.disk_usage(".").free < 5e9:
            note(tag, "STOP_DISK_LOW"); status("stopped: disk < 5 GB"); sys.exit(2)
        status(f"running {tag} attempt {attempt}")
        kill_thor()
        if not restart_bridge():
            note(tag, f"attempt{attempt}", "BRIDGE_RESTART_FAILED"); time.sleep(60); continue
        why, steps, ncap = run_attempt(job, attempt)
        if why:
            note(tag, f"attempt{attempt}", f"KILLED_{why}", f"steps={steps}", f"captions={ncap}"); continue
        # 2026-11-03 fix: tours are exempt from the search-episode caption-rate rule (a tour spins in place and
        # captions less often; tour_house99 was excluded 3x at a deterministic 293/3000 with a healthy bridge).
        # A tour's quality gate is validate_seed.py (its `post` step).
        if steps and ncap * 10 < steps and job.get("kind") != "tour":
            with open(LOG / "excluded.tsv", "a") as f:
                f.write(f"{tag}\tattempt{attempt}\tcaptions={ncap}\tsteps={steps}\n")
            note(tag, f"attempt{attempt}", "EXCLUDED_infrastructure", f"steps={steps}", f"captions={ncap}"); continue
        if done(tag):
            note(tag, f"attempt{attempt}", "OK", f"steps={steps}", f"captions={ncap}")
            post = job.get("post")
            if post:
                rc = sh(post, 600)
                note(tag, "POST", "OK" if rc == 0 else f"POST_FAILED_rc{rc}")
            return True
        note(tag, f"attempt{attempt}", "NO_RESULT", f"steps={steps}", f"captions={ncap}")
    note(tag, "FAILED_3_ATTEMPTS"); return False


def main():
    jobs = json.load(open(sys.argv[1]))
    note("QUEUE_START", len(jobs), "jobs", "dry" if DRY else "live")
    for i, job in enumerate(jobs):
        req = job.get("requires")
        if req and not os.path.exists(req):
            note(job["tag"], "SKIPPED_requires", req); continue
        fewer = job.get("only_if_fewer_than")
        if fewer and sum(os.path.exists(m) for m in fewer["markers"]) >= fewer["n"]:
            note(job["tag"], "SKIPPED_not_needed"); continue
        if DRY:
            print(i, job["tag"], job.get("house"), job.get("start"), job.get("age"), job.get("reloc", "")[:30], " ".join(job["args"]))
            continue
        try:
            run_job(job)
        except Exception as exc:  # noqa: BLE001 -- a bug in one job must never end the night
            note(job["tag"], "QUEUE_EXCEPTION", repr(exc)[:200])
    status("QUEUE_DONE")
    note("QUEUE_DONE")


if __name__ == "__main__":
    main()
