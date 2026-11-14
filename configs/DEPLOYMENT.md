# Deployment and calibration guidance

`mobility_priors.json` and `memory.yaml` ship with **cold-start defaults**,
not values validated for any specific real deployment. They were generated
(mobility priors: a single batch LLM prompt; `memory.yaml`'s `alpha`,
`lambda_base`, `theta_direct`, etc.: a documented, stated rationale) for
this project's own simulated evaluation on one HM3D scene -- a static,
prerecorded environment where no object ever actually moves. Real
deployments have real dynamics these defaults were never tuned against, so
before relying on them anywhere else, recalibrate:

- **`lambda_base`** (`memory.yaml`, reliability section) -- the base
  temporal decay rate. Set from how fast *your* environment actually
  changes: a busy shared kitchen warrants a much shorter half-life than a
  quiet private office. The shipped value assumes a fully-mobile object's
  half-life is comparable to a single search episode's rough duration --
  recalibrate against your own typical time-between-queries instead of
  reusing this assumption.
- **Per-category mobility priors** (`mobility_priors.json`) -- an LLM's
  cold-start guess at which object categories move often (a phone) versus
  rarely (a radiator). If verification failures are logged in your
  deployment, `memory_layer/mobility.py`'s documented (not yet
  implemented) update hook is the intended path to replace these with
  priors learned from your own environment's real failure rates, rather
  than trusting the cold-start table indefinitely.
- **Verification-failure discount** (`memory.yaml`, `verification_failure_discount`)
  -- how much a single failed re-verification should discount `v_i`.
  Tune against how often your own perception stack produces one-off,
  low-quality observations versus genuine environmental change.
- **Confirmation threshold** (`min_confirming_observations`) -- this
  project's own sweep (1 through 5, see the paper) found the right value
  highly sensitive to how much per-episode time budget is available to
  accumulate confirmations; recalibrate against your own perception
  stack's grounding/captioning cadence, not this project's specific
  grounding-cycle latency.
- **Observation/update frequency** -- how often your own perception stack
  actually re-observes and re-confirms a given entity governs how quickly
  `n_i` can grow; a slower perception stack needs a more forgiving
  confirmation threshold, a faster one can afford a stricter one.

None of the above is a large new discussion for the paper -- it belongs
here, next to the actual config files a real deployment would edit, not
as prose a reader has to extract before they can act on it.
