# Failure Triage Log

Every crash or unexpected result gets one entry: what happened, suspected
cause, whether it's resolved or still open. Per CLAUDE.md §10.

---

## 2026-09-04: `memory_size_at_end=0` across an entire persistent-mode batch
(non-VQA) despite DAAAM genuinely captioning objects

**What happened**: `phase4_batch_persistent_longsession10_covmem` (10
queries, 500 steps, `--persistent-memory`, no `--vqa-verification`) --
0/10 success, `memory_size_at_end=0` on EVERY episode, 100% fallback
branching, every episode ran the full 500-step budget. A complete
regression from the identical-config baseline run the previous day
(`phase4_batch_persistent_longsession10`: success_rate=0.8, memory
growing to 300+ entities by episode end).

**Ruled out first**: the coverage-memory code change (`agent.reset()`
gaining `persist_coverage`) committed just before this run. Confirmed via
`git show` that the diff only touches `FrontierAgent`-internal state
(`_visited_positions`, path state) -- never `MemoryStore` or the DAAAM
client/bridge connection. Not the cause.

**Root cause, localized**: read the raw bridge-server log
(`docker exec daaam_dev ... /tmp/daaam_bridge_server.log`) for this run's
exact time window and confirmed DAM-3B genuinely captioned real objects
and DAAAM's grounding workers genuinely created real corrections
throughout the run (`[GroundingWorker-0] Created correction for
semantic_id 4348: 'A vertical, elongated painting...'`, dozens of them,
real plausible descriptions). So DAAAM's perception pipeline itself was
healthy. But `store.entities` stayed completely empty the whole run --
traced the code path (`create_corrections_from_descriptions` in
`grounding/utils.py`, real DAAAM source) and confirmed these corrections
are created inside a grounding WORKER subprocess and must cross a
`multiprocessing.Queue` into the main orchestrator process before
`_correction_processor_loop` can dequeue them and fire
`semantic_update_callback` (which is what fills `_pending_updates`,
what `/updates` drains, what `run_phase4_batch.py`'s `pull_updates()`
consumes into `MemoryStore`). That hand-off was silently dropping
every correction: workers print their log line successfully, but nothing
ever reaches `/updates`.

**Suspected trigger**: the bridge server (PID alive since 08:17 this
morning) had already run an earlier `--vqa-verification` smoketest
(`phase4_batch_vqa_vqa_smoketest2`) before this batch, in the SAME
long-lived process. `--vqa-verification`'s `_get_verify_agent()` lazily
loads a SEPARATE `DAMAgentPanoptic` instance sharing the GPU with the
async pipeline's own grounding workers, with no cross-coordination
between `_verify_lock` and whatever GPU/CUDA-context locking the async
worker pool relies on internally. Hypothesis: that concurrent GPU usage
destabilized the worker pool's queue hand-off for the rest of that
process's lifetime. NOT YET CONFIRMED -- correlation only so far.

**What WAS confirmed by direct test**: restarted the bridge server
process fresh (killed PID, confirmed dead, relaunched, confirmed
`/health` clean) and ran a short 2-query/150-step smoke test
(`phase4_batch_persistent_freshrestart_smoketest`, no `--vqa-verification`)
against it. Result: episode 0 (first ~60s of the fresh process) STILL
showed `memory_size_at_end=0` -- so a fresh restart alone does not
immediately fix it, there's a genuine cold-start warm-up window (model
loading inside the worker processes takes longer than one short episode).
But episode 1 (same persistent session, immediately after, no restart in
between) showed `memory_size_at_end=232` with 150/150 steps using the
`contextual_anchor` branch -- real memory-driven navigation, working
correctly. This rules out "permanently broken correction hand-off" and
narrows it to a startup-latency effect, consistent with DAAAM's own
measured ~12.9s avg / 21.2s max async grounding latency compounding with
whatever the worker pool's own one-time model-load cost is on top of that.

**Status**: OPEN, partially mitigated. Relaunched the real 10-query
coverage-memory validation batch (`..._covmem_v2`) immediately after the
smoke test confirmed the SAME warmed-up process was producing real
memory, to get a genuine result while the process is in a known-good
state. Real fix (not yet done): either (a) have `run_phase4_batch.py`
verify `/updates` returns non-empty at least once before starting real
metrics collection, per persistent session, and/or (b) avoid mixing
`--vqa-verification` and non-VQA batches on the same long-lived bridge
process without a restart in between, until (a) or the actual GPU-
contention hypothesis is confirmed/ruled out directly.

**Action item**: do not launch a multi-episode batch immediately after a
bridge-server (re)start without first confirming `/updates` is flowing
(e.g. a throwaway 1-2 minute warm-up episode, discarded from metrics).
