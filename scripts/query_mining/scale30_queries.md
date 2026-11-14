# 30-query full-scene stress test (scene 00770), 2026-09-05

Mined from `captions_00770_full.txt` (2733 real distinct DAAAM captions,
same corpus Section 7.7 uses). Goal: stress-test the pipeline across
genuinely more of 00770's 19 real rooms (per HM3D's own metadata.csv),
not just the existing curated ~16-query set.

## Final 30 queries
Existing 16 (8 original + 8 from Section 7.7's mined battery):
cushion, window, chair, mirror, table, radiator, doorway, painting,
cabinet, trash bin, wall vent, curtain, fireplace, lamp, wall clock, bench

New 14 (this session, verified below):
chandelier, vase, bathtub, towel, thermostat, shelf, sculpture, candle,
ceiling light fixture, drawer, pillar, window sill, rug, ceiling fan

## Verification methodology (same rigor as Section 7.7)
For each candidate, counted real substring hits in the caption corpus,
then read the FULL text of every hit (not just a sample) to confirm the
match is a genuine standalone object reference, not an accidental
substring inside a longer word -- exactly the class of bug that produced
the real staircase/air-conditioner false positive earlier in this
project (PROGRESS.md, 2026-08-28).

## Candidates checked and REJECTED (real bugs caught before use)
- "bed": 27 hits, only 1 genuine -- 26/27 were "ribbed" or "embedded".
- "gate": 111 hits, 0 genuine -- all were "elongated" or "corrugated".
- "urn": 15 hits, 0 genuine -- all were "furniture" or "nocturnal".
- "plate": 29 hits, mostly genuine word "plate" but semantically
  ambiguous -- several are a hinge's mounting/back "plate" (hardware),
  not dinnerware. Dropped for semantic risk, not a substring bug.
- "arch"/"archway": 79 hits, but nearly all are "arched" describing a
  WINDOW's shape (e.g. "the top section is a large arched window"), not
  a standalone archway structure. Ambiguous referent, dropped.
- "door": real matches, but "door" is a lexical PREFIX of the
  already-used "doorway" -- would compound the residual cabinet-door/
  doorway ambiguity already documented in paper Section 7 Limitations.
  Dropped to avoid stacking a known ambiguity on itself.
- "fan" (generic): 5 hits, 4 genuine ceiling fans + 1 false positive
  ("the fish has a broad, fan-shaped body..." -- describes a fish, not
  a fan). Kept, but phrased specifically as "a ceiling fan" (matches
  the real captions' actual phrasing) rather than bare "a fan".

## Candidates checked and KEPT (verified genuine)
door-adjacent risk aside, all of: chandelier (30 hits, all genuine),
vase (8, genuine), bathtub (7, genuine), towel (10, genuine),
thermostat (14, genuine), shelf (15, genuine), sculpture (16, genuine),
candle (23, genuine), light fixture (39, genuine -- phrased "a ceiling
light fixture"), drawer (92, genuine), pillar (12, genuine), window
sill (6, genuine), rug (19/23 genuine, 4 false via "struggle"/
"corrugated"/"corrugation" -- kept since the real 19 are solid and the
pipeline's own lexical gate uses prefix-matching, which none of those
false hits satisfy anyway), ceiling fan (4/5 genuine, see above).

Full mining script: run inline in the session, not saved as a
standalone script (one-off analysis) -- methodology fully documented
above for reproducibility if needed again for a future scene.
