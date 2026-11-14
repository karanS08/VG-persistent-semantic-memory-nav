"""Operationalizes "target visually verified" (CLAUDE.md Section 6's success
condition) for a pipeline with no dedicated visual-verification/VQA classifier.

This is a documented proxy, not a real answer: an entity counts as verified if (a)
its stored description is relevant enough to the query (CLIP cosine similarity
above relevance_threshold) and (b) the agent is currently within verify_distance
of its remembered position. Real verification would re-observe the object and
confirm it visually at close range; this instead trusts the DAAAM-grounded
description once the agent is close enough that a real system plausibly could
verify it. Logged as a scoping decision (implementation detail, not a
CLAUDE.md-flagged research-design gap like the contextual-anchor formula), not
silently assumed.

IMPORTANT caveat on threshold calibration: CLIP cosine similarities have a
well-known non-zero baseline -- unrelated text/image pairs routinely score
0.1-0.3. A relevance_threshold in that range is therefore NOT clearly
discriminative, and a "success" it produces may reflect proximity to almost any
grounded object rather than to the actual query target. `VerificationResult`
below exists specifically so that claim can be checked against real runs (what
actually triggered, at what score, versus the best alternative) instead of
assumed either way. See PROGRESS.md's Phase 4 verification-threshold audit.

2026-08-24 fix: the audit's own instrumentation caught this happening for real.
Across an 8-episode batch (data/runs/phase4_batch/batch_metrics.json), 7 of 8
"verified" successes triggered on an entity that was NOT the best-scoring match
in memory -- e.g. query "a wall-mounted mirror" verified against a caption
describing a window, while a caption describing a wall vent scored higher but
was out of range. The old criterion accepted ANY entity clearing the absolute
threshold AND in range, so a semantically-wrong-but-nearby object could win
purely because the true best match hadn't been reached yet. Since an absolute
CLIP-similarity cutoff can't reliably separate signal from baseline noise (see
paragraph above), the fix is rank-based instead: only the entity that is
GLOBALLY the best relevance match across all of memory (ties within
TOP_RANK_EPSILON) is eligible to trigger verification, and only once the agent
is within verify_distance of THAT entity specifically. A worse-scoring nearby
object can no longer stand in for a not-yet-reached better match. This does
not fully solve threshold calibration (the top-ranked match can itself still
be wrong when nothing in the scene truly matches the query -- see
best_relevance_in_memory in the results either way), but it closes the
specific "nearby-but-wrong-object" failure mode this run actually exhibited.

2026-08-27 fix: the 2026-08-24 fix wasn't enough. A persistent-across-queries
8-episode batch (data/runs/phase4_batch_persistent/) hit 4/8 false positives
even with rank-based selection: "a wooden chair" verified against a caption
that actually read "a long, narrow, white marble TABLE"; "a dining table"
verified against "the FLOOR... light beige wooden planks" (not furniture at
all); "a doorway" verified against "a tall, narrow, white CABINET"; "a soft
cushion" against "a rectangular OTTOMAN". Common thread, confirmed by
inspection: these were exactly the kind of single-observation, noise-floor
CLIP coincidence the module docstring above already warned about -- as the
accumulated memory pool grows (up to ~1500 entities in a persistent session),
the odds that SOME entity randomly clears relevance_threshold and ends up
within verify_distance of wherever the agent wanders keeps climbing, because
nothing before this required the match to be more than a single grounding
call's raw CLIP similarity. Fix: require BOTH (a) relevance clears
relevance_threshold as before, AND (b) the entity has at least
`min_confirming_observations` independent observations (`entity.n_i`) -- a
one-off, possibly-hallucinated caption from a single grounding call can no
longer trigger verification on its own; it needs to have been seen again
(record_observation) at least once more, consistent with CLAUDE.md Section 6's
own framing of n_i as diminishing-returns evidence accumulation. AND (c) among
entities passing both gates, rank by R_i = relevance * reliability (the exact
same product CLAUDE.md Section 6 defines for direct recall, see
memory_layer/recall.py), not raw relevance alone -- so a well-corroborated,
still-fresh entity outranks a fresher-but-shakier one at similar relevance,
instead of raw CLIP score alone deciding the winner. This deliberately reuses
the project's own reliability math rather than inventing a new ad hoc
threshold. Entities excluded by the n_i gate are still visible in
`n_excluded_by_confirmation_gate` for auditing whether this gate is doing
real work or just delaying the same coincidences.

2026-08-28 fix: auditing the fixed batches per-episode (not just
success_rate) found a THIRD false-positive class the 08-27 fix doesn't
touch: category confusion between visually/shape-similar object types.
Across every re-run batch, "a wall-mounted mirror" and "a doorway" verified
against genuinely well-confirmed, high-reliability WINDOW entities; "a
framed painting" and "a dining table" verified against genuinely
well-confirmed OVEN/drawer/counter entities -- in one persistent-mode run
the SAME oven entity independently "verified" three different queries
(mirror, dining table, framed painting) in one session. These are not
noise-floor coincidences (n_i>=2, often much higher) -- they're the raw CLIP
relevance score itself failing to discriminate rectangular-framed-glass
things from each other, or boxy-appliance things from each other, at the
similarity magnitudes this pipeline actually produces (~0.11-0.19). Neither
the confirmation gate nor reliability-ranking encodes anything about
WHETHER THE MATCHED ENTITY IS THE RIGHT CATEGORY OF THING, only how
confidently the (possibly wrong) top-ranked entity was observed. Fix: an
optional lexical head-noun consistency gate -- if the caller supplies the
raw `query_text`, a candidate must share a substring-overlapping content
word with the query's head noun (the last non-article word, e.g. "mirror"
in "a wall-mounted mirror") against its `category`/`description` text
before it's eligible at all. This is a blunt heuristic (no synonymy, e.g.
"lamp" vs "light" won't match), logged as such rather than presented as a
real semantic solution -- but it directly targets the observed failure
mode and is deliberately narrow (a genuine word-overlap requirement, not a
new numeric threshold to calibrate). `query_text` defaults to None (gate
skipped) so existing callers/tests are unaffected until they opt in; see
`n_excluded_by_category_gate` for auditing how much work this gate does.

2026-09-02: `min_confirming_observations` default changed 2 -> 1. Evidence
across three independent evaluations (the 08-28 `catgate` 14-batch sweep,
and both Tier 4b real-caption-mined batches on scene 00770) consistently
shows the category gate above, not the confirmation-count gate, is doing
the real discriminative work: with n_i>=2 required, `n_excluded_by_
confirmation_gate` runs into the hundreds-to-thousands per episode while
`n_excluded_by_category_gate` stays single-digit, and inspecting what gets
excluded at n_i=1 shows real, correctly-captioned objects (a genuine
bench, a genuine cabinet) discarded simply for having been seen once. The
`catgate` sweep's minconf1 condition (n_i>=1, category gate active) scored
the single best fully-audited result of any config tried: 5/8 genuine
successes, zero false positives. The n_i>=2 gate remains available via
`--min-confirming-observations 2` for anyone who wants the stricter,
slower-to-trust behavior; see PROGRESS.md for the full before/after data.

2026-09-02 (second fix, same day): with n_i>=1 as the default, MORE
candidates now reach the lexical gate than before, which exposed a real
gap in it: it originally checked the OR of `category` and `description`
text, and DAAAM's own `category` field can be wrong -- specifically wrong
in ways `description` is not -- for categories DAAAM's coarse classifier
handles poorly. Confirmed with a real, reportable failure: on a real-
caption-mined query battery, the adversarial query "a staircase" (an
object confirmed absent from the scene) verified against an entity whose
`category` said "staircase" but whose `description` read "a wall-mounted
air conditioner" -- category alone was enough to pass the old OR'd check
despite the description plainly disagreeing. `_lexically_consistent` now
checks `description` only -- and, since dropping `category` alone turned
out NOT to fix this specific real case (the description itself contains
"air conditioner", and "air" is a coincidental mid-word substring of
"staircase"), also switched from substring-anywhere matching to a prefix
check (see `_lexically_consistent`'s own docstring for the full reasoning
and why this preserves every known-good match, e.g. door/doorway,
vent/ventilation, cushion/cushioned). Every true positive audited across
this project's whole evaluation history matched via `description` text
anyway, so this costs no known recall -- re-verified directly against the
real failing case and re-running the adversarial battery, see PROGRESS.md.

2026-09-07 fix (query interpretation policy, relational queries):
`results/lexical_rule_policy.md` confirmed a real, reproducible bug by
running the query directly: "a lamp near a window" verified against
entity_id=82, described as "A tall, rectangular window with a white
frame..." -- zero mention of a lamp anywhere. Root cause: `_query_head_noun`
took the literal LAST non-article word of the whole query string, so a
query naming two objects ("lamp", the actual target, and "window", a
spatial reference) silently extracted the reference noun instead of the
target -- by construction, on every relational query, not just this one.
Fix: before extracting the head noun, split the query at the first
occurrence of a small, fixed set of spatial-relation markers (_RELATION_
MARKERS below: near/on/above/below/under/behind/etc.) and take only the
portion BEFORE the marker as the target phrase. "a lamp near a window"
now correctly yields "lamp"; "a lamp on a nightstand" yields "lamp", not
"nightstand". This is deliberately narrow -- a fixed marker list, not a
real dependency parse -- and does not attempt conjunctions ("a lamp and a
chair"), appositives, or any relation vocabulary outside the list below.
It also does not change behavior for any query with no marker present:
every single-noun-phrase query this project has ever run or reported
(all 8 in the standard battery, both held-out-scene batteries, every
adversarial query) contains no marker word, so this fix cannot regress
them -- confirmed by re-running the full test suite (unchanged pass
count) plus a new targeted regression test reproducing the exact
entity_id=82 case above.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from memory_layer.mobility import MobilityPriors
from memory_layer.relevance import compute_relevance
from memory_layer.reliability import compute_reliability
from memory_layer.store import MemoryStore


@dataclass
class VerificationResult:
    verified: bool
    entity_id: Optional[int] = None
    description: Optional[str] = None
    relevance: Optional[float] = None
    distance: Optional[float] = None
    reliability: Optional[float] = None
    r_i: Optional[float] = None  # relevance * reliability, the winning entity's
    n_i: Optional[int] = None  # confirming-observation count of the winning entity
    # The highest-relevance entity in memory regardless of distance/threshold/n_i
    # gate, for comparison: if the thing that triggered verification scores no
    # better than the scene's generic best match, the threshold isn't discriminating.
    best_relevance_in_memory: Optional[float] = None
    best_relevance_description: Optional[str] = None
    # How many DISTINCT entities were simultaneously eligible (over threshold,
    # past the n_i gate, AND in range) at the moment of verification. >1 means
    # the criterion had a choice and the "winner" is partly an artifact of
    # dict iteration order.
    n_eligible: int = 0
    # How many entities cleared relevance_threshold but were excluded solely by
    # the n_i (confirming-observation) gate -- see 2026-08-27 module note.
    n_excluded_by_confirmation_gate: int = 0
    # How many entities passed the relevance + n_i gates but were excluded by
    # the lexical head-noun consistency gate -- see 2026-08-28 module note.
    # Always 0 when query_text is not supplied (gate skipped entirely).
    n_excluded_by_category_gate: int = 0


TOP_RANK_EPSILON = 1e-6

_ARTICLES = {"a", "an", "the"}

# 2026-09-07: closed list of spatial-relation markers that introduce a
# REFERENCE object rather than the query's actual target, e.g. "a lamp near
# a window" is a query for a lamp, using "window" only as a spatial hint --
# not a query for a window. See _query_head_noun and the 2026-09-07 module
# note above for the real bug this fixes. Ordered longest-first so a
# multi-word marker is matched before any shorter marker that happens to be
# a substring of it (none currently are, but this keeps the list safe to
# extend).
_RELATION_MARKERS = tuple(sorted((
    "near", "next to", "close to", "beside", "adjacent to",
    "on", "above", "below", "under", "underneath",
    "in front of", "behind", "across from", "opposite",
), key=len, reverse=True))


def _split_target_phrase(query_text: str) -> str:
    """Returns the portion of `query_text` before the first spatial-relation
    marker in `_RELATION_MARKERS`, or the whole (lowercased) string if no
    marker is present. Matches on whole words/phrases only (`\\b` boundaries)
    so a marker never fires on a coincidental substring inside an unrelated
    word (e.g. "on" does not match inside "iron")."""
    lowered = query_text.lower()
    for marker in _RELATION_MARKERS:
        m = re.search(r"\b" + re.escape(marker) + r"\b", lowered)
        if m:
            return lowered[: m.start()]
    return lowered


def _query_head_noun(query_text: str) -> Optional[str]:
    """Head noun of the query's TARGET phrase -- e.g. "lamp" from "a lamp
    near a window" (the portion before the relation marker "near", see
    _split_target_phrase), or "mirror" from "a wall-mounted mirror"
    (unchanged: no relation marker present, so the whole string is the
    target phrase). Still a blunt heuristic within that target phrase
    itself -- the last non-article word, no real POS tagging, no synonymy --
    see the module docstring for the known limits that remain even after
    this fix (compound nouns, conjunctions, synonyms)."""
    target_phrase = _split_target_phrase(query_text)
    words = [w for w in re.findall(r"[a-zA-Z]+", target_phrase) if w not in _ARTICLES]
    return words[-1] if words else None


def _lexically_consistent(query_text: str, entity) -> bool:
    """2026-09-02 fix: checks `entity.description` ONLY -- `category` was
    dropped from this check. See module docstring for why: real-world
    testing (the Tier 4b real-caption-mined battery, at the retuned
    min_confirming_observations=1 default) found DAAAM's own `category`
    field can disagree with an accurate `description` for categories
    DAAAM's coarse classifier handles poorly, and when it does, the OR'd
    check let category alone wave through a genuine false positive --
    concretely, an adversarial "a staircase" query verified against an
    entity whose `category` said "staircase" but whose `description`
    plainly read "a wall-mounted air conditioner". Every genuine true
    positive audited across this project matched via `description` text
    anyway, so dropping `category` from this check costs no known recall.

    Second bug found writing the regression test for the first: dropping
    `category` alone did NOT fix the real staircase/air-conditioner case,
    because the description itself contains "air conditioner" -- and the
    OLD matching rule (`head in w or w in head`, substring anywhere) let
    "air" match inside "st-AIR-case" purely by coincidental mid-word letter
    overlap, not any real relationship between the words. Every case this
    gate is actually meant to catch (door/doorway, vent/ventilation,
    cushion/cushioned, window/windows) is a PREFIX relationship -- one word
    is what the other starts with, i.e. real morphological
    inflection/pluralization. "air" is not a prefix of "staircase" (which
    starts with "st"), so switching to a prefix check (`w.startswith(head)
    or head.startswith(w)`) fixes the real bug while preserving every
    known-good match audited across this project."""
    head = _query_head_noun(query_text)
    if head is None:
        return True
    target_text = (entity.description or "").lower()
    target_words = [w for w in re.findall(r"[a-zA-Z]+", target_text) if len(w) > 2]
    return any(w.startswith(head) or head.startswith(w) for w in target_words)


# 2026-09-28: "caption_head" lexical mode. The legacy prefix rule above
# checks the query head against EVERY word of the caption, so any object whose
# caption merely mentions the query word passes: "a soft cushion" -> "a bench
# with a cushioned seat", "a doorway" -> "an oven ... with a glass door",
# "a framed painting" -> "a window ... with peeling paint". Offline replay of
# every logged verified (query, caption) pair (276 pairs / 377 episodes, see
# PROGRESS.md 2026-09-28) showed nearly all legacy-only accepts are exactly
# these audited false positives. caption_head mode instead compares against
# the caption's own SUBJECT noun phrase -- DAM-3B captions reliably open with
# "A <modifiers> <noun> with/featuring/...", or "The <noun> is/features ...".
# Still no POS tagger or synonym table; closed stop-word list only.
_CAPTION_NP_STOP = frozenset({
    "with", "featuring", "features", "feature", "that", "which", "is", "are",
    "has", "have", "of", "on", "in", "at", "mounted", "attached", "set",
    "placed", "positioned", "hanging", "consisting", "consists", "made", "and",
    "showing", "shows", "displaying", "depicting", "situated", "located",
    "against", "to", "from", "by", "for", "near", "under", "above", "below",
    "over", "beside", "leaning", "resting", "containing", "framed", "adorned",
    "covered", "bordered", "surrounded", "supported", "appears", "extends",
    "runs", "sits", "stands",
})
# Scene-surface subjects: DAAAM sometimes segments a wall/ceiling/floor/panel
# region around an object and captions it "The wall is adorned with a large
# rectangular mirror" / "A white wall panel with a central mirror". For these
# subjects only, the noun phrase after EACH carrier marker
# (_SURFACE_CARRIERS) is also treated as a subject. v2 (2026-09-28, after the
# first live caption_head batch lost a genuine "wall panel with a central
# mirror"): added "panel" and the with/adorned/covered carriers; replay over
# all 1061 logged (query, caption) pairs changed 5 decisions, all genuine
# gains (4 mirrors, 1 rug), 0 losses.
_SURFACE_SUBJECTS = frozenset({"wall", "ceiling", "floor", "panel"})
_SURFACE_CARRIERS = r"\b(features|featuring|with|adorned|covered)\b"


def _caption_noun_phrase(text: str) -> List[str]:
    """Content tokens of the leading noun phrase of `text` (first sentence,
    articles dropped, cut at the first `_CAPTION_NP_STOP` token). Hyphenated
    tokens ("wall-mounted", "built-in") are kept whole so their parts never
    count as the subject noun."""
    first = re.split(r"[.;:()]", text.lower())[0]
    out = []
    for tok in re.findall(r"[a-z]+(?:-[a-z]+)*", first):
        if tok in _CAPTION_NP_STOP:
            break
        if tok not in _ARTICLES:
            out.append(tok)
    return out


def _caption_subject_nouns(description: str) -> List[List[str]]:
    """One or more subject noun phrases, each reduced to its last two content
    tokens (so "drawer front" / "air conditioner unit" keep their real noun)."""
    np_ = _caption_noun_phrase(description)
    phrases = [np_[-2:]] if np_ else []
    if np_ and np_[-1] in _SURFACE_SUBJECTS:
        first = description.lower().split(".")[0]
        for m in re.finditer(_SURFACE_CARRIERS, first):
            obj = _caption_noun_phrase(first[m.end():])
            if obj:
                phrases.append(obj[-2:])
    return phrases


def _singular(word: str) -> str:
    for suf in ("es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _caption_head_consistent(query_text: str, entity) -> bool:
    """Query head noun must match the caption's subject noun: exactly (modulo
    plural) against either of the phrase's last two tokens, or by prefix
    against the FINAL token only (door->doorway, lamp->lampshade). Modifiers
    such as "cushioned" or "built-in" are never the final token of a
    well-formed subject phrase, so the prefix allowance cannot reach them."""
    head = _query_head_noun(query_text)
    if head is None:
        return True
    head = _singular(head)
    for phrase in _caption_subject_nouns(entity.description or ""):
        tokens = [_singular(t.split("-")[-1]) for t in phrase]
        if head in tokens:
            return True
        last = tokens[-1]
        if last.startswith(head) or head.startswith(last):
            return True
    return False


LEXICAL_MODES = ("prefix", "caption_head")


def check_target_verified_detailed(
    agent_position: List[float],
    memory_store: MemoryStore,
    query_embedding: List[float],
    relevance_threshold: float,
    verify_distance: float,
    now: float,
    mobility_priors: MobilityPriors,
    alpha: float,
    lambda_base: float,
    mobility_scaling_enabled: bool = True,
    min_confirming_observations: int = 1,
    query_text: Optional[str] = None,
    lexical_mode: str = "prefix",
) -> VerificationResult:
    agent_xz = (agent_position[0], agent_position[2])

    all_scored = []  # (relevance, entity) -- every entity with position+embedding
    for entity in memory_store.entities.values():
        if entity.position_world is None or entity.embedding is None:
            continue
        relevance = compute_relevance(query_embedding, entity.embedding)
        all_scored.append((relevance, entity))

    if not all_scored:
        return VerificationResult(verified=False)

    all_scored.sort(key=lambda t: t[0], reverse=True)
    best_relevance, best_entity = all_scored[0]
    best_relevance_desc = best_entity.description

    if best_relevance < relevance_threshold:
        return VerificationResult(
            verified=False,
            best_relevance_in_memory=best_relevance,
            best_relevance_description=best_relevance_desc,
        )

    # Gate 1: clears relevance_threshold. Gate 2 (2026-08-27): at least
    # min_confirming_observations independent sightings -- a single grounding
    # call's caption isn't enough to trust on its own, see module docstring.
    above_threshold = [(rel, ent) for rel, ent in all_scored if rel >= relevance_threshold]
    n_excluded_by_confirmation_gate = sum(1 for _, ent in above_threshold if ent.n_i < min_confirming_observations)
    after_ni_gate = [(rel, ent) for rel, ent in above_threshold if ent.n_i >= min_confirming_observations]

    # Gate 3 (2026-08-28): lexical head-noun consistency -- see module note.
    # Skipped entirely (candidates = after_ni_gate) if the caller didn't pass
    # query_text, so old callers/tests keep their pre-08-28 behavior exactly.
    if query_text is not None:
        if lexical_mode not in LEXICAL_MODES:
            raise ValueError(f"unknown lexical_mode {lexical_mode!r}")
        consistent = _caption_head_consistent if lexical_mode == "caption_head" else _lexically_consistent
        candidates = [(rel, ent) for rel, ent in after_ni_gate if consistent(query_text, ent)]
        n_excluded_by_category_gate = len(after_ni_gate) - len(candidates)
    else:
        candidates = after_ni_gate
        n_excluded_by_category_gate = 0

    if not candidates:
        return VerificationResult(
            verified=False,
            best_relevance_in_memory=best_relevance,
            best_relevance_description=best_relevance_desc,
            n_excluded_by_confirmation_gate=n_excluded_by_confirmation_gate,
            n_excluded_by_category_gate=n_excluded_by_category_gate,
        )

    # Rank by R_i = relevance * reliability -- the same product CLAUDE.md
    # Section 6 defines for direct recall -- not raw relevance alone, so a
    # well-corroborated entity outranks a shakier one at similar relevance.
    scored = []  # (r_i, relevance, reliability, entity)
    for relevance, entity in candidates:
        mobility = mobility_priors.mobility_for(entity.category) if mobility_scaling_enabled else 1.0
        reliability = compute_reliability(entity, now, mobility, alpha, lambda_base)
        scored.append((relevance * reliability, relevance, reliability, entity))
    scored.sort(key=lambda t: t[0], reverse=True)
    best_r_i = scored[0][0]

    top_candidates = [t for t in scored if best_r_i - t[0] <= TOP_RANK_EPSILON]

    eligible = []  # (r_i, relevance, reliability, distance, entity)
    for r_i, relevance, reliability, entity in top_candidates:
        entity_xz = (entity.position_world[0], entity.position_world[2])
        distance = math.sqrt((agent_xz[0] - entity_xz[0]) ** 2 + (agent_xz[1] - entity_xz[1]) ** 2)
        if distance <= verify_distance:
            eligible.append((r_i, relevance, reliability, distance, entity))

    if not eligible:
        return VerificationResult(
            verified=False,
            best_relevance_in_memory=best_relevance,
            best_relevance_description=best_relevance_desc,
            n_excluded_by_confirmation_gate=n_excluded_by_confirmation_gate,
            n_excluded_by_category_gate=n_excluded_by_category_gate,
        )

    # Among tied top candidates, prefer the nearest.
    eligible.sort(key=lambda t: t[3])
    r_i, relevance, reliability, distance, entity = eligible[0]
    return VerificationResult(
        verified=True,
        entity_id=entity.entity_id,
        description=entity.description,
        relevance=relevance,
        distance=distance,
        reliability=reliability,
        r_i=r_i,
        n_i=entity.n_i,
        best_relevance_in_memory=best_relevance,
        best_relevance_description=best_relevance_desc,
        n_eligible=len(eligible),
        n_excluded_by_confirmation_gate=n_excluded_by_confirmation_gate,
        n_excluded_by_category_gate=n_excluded_by_category_gate,
    )


def check_target_verified(
    agent_position: List[float],
    memory_store: MemoryStore,
    query_embedding: List[float],
    relevance_threshold: float,
    verify_distance: float,
    now: float,
    mobility_priors: MobilityPriors,
    alpha: float,
    lambda_base: float,
    mobility_scaling_enabled: bool = True,
    min_confirming_observations: int = 1,
    query_text: Optional[str] = None,
) -> Tuple[bool, Optional[int]]:
    """Backwards-compatible wrapper over check_target_verified_detailed."""
    result = check_target_verified_detailed(
        agent_position, memory_store, query_embedding, relevance_threshold, verify_distance,
        now, mobility_priors, alpha, lambda_base, mobility_scaling_enabled, min_confirming_observations,
        query_text,
    )
    return result.verified, result.entity_id
