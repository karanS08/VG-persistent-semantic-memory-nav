"""Score a returned blind-audit answer sheet against the key (Test 3.4).

Run AFTER the second rater returns answers_template.csv (filled). Computes:
  * confusion matrix + Cohen's kappa: rater vs system verdict (verified / not verified)
  * confusion matrix + Cohen's kappa: rater vs the author's existing audit, on reported
    successes, under the published (lenient) and strict readings
  * per-condition verification precision according to the rater, with Wilson 95% CIs
Q2 (caption) is the primary rating, because that is what the author's audit used; Q1 (image) is
reported separately where images exist. "partial" counts as NOT a match (primary) and a
sensitivity analysis counts it as a match. "unsure"/"not_visible"/"na"/blank are excluded and counted.

Usage:
  ~/miniconda3/envs/rpv_mem/bin/python scripts/postdraft/score_blind_audit.py <answers.csv> [--out DIR]
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEY = ROOT / "data/results_postdraft/gt_audit_precision/audit_key_DO_NOT_SHARE/key.csv"
Z = 1.959963984540054


def wilson(k, n):
    if not n:
        return None
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def kappa(pairs):
    """Cohen's kappa for binary pairs [(a, b)] with a, b in {0, 1}."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def confusion(pairs):
    c = Counter(pairs)
    return {"both_yes": c[(1, 1)], "a_yes_b_no": c[(1, 0)], "a_no_b_yes": c[(0, 1)], "both_no": c[(0, 0)]}


def to_bin(ans, partial_is_match):
    a = (ans or "").strip().lower()
    if a == "yes":
        return 1
    if a == "no":
        return 0
    if a == "partial":
        return 1 if partial_is_match else 0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--out", default=str(ROOT / "data/results_postdraft/gt_audit_precision/second_rater_scoring"))
    args = ap.parse_args()
    key = {r["item_id"]: r for r in csv.DictReader(open(KEY))}
    ans = {r["item_id"]: r for r in csv.DictReader(open(args.answers))}
    missing = sorted(set(key) - set(ans))
    res = {"n_key": len(key), "n_answered_rows": len(ans), "missing_items": missing}
    for q in ("q2_caption", "q1_image"):
        for partial in (False, True):
            tag = f"{q}__partial_{'match' if partial else 'nonmatch'}"
            sys_pairs, auth_len, auth_str = [], [], []
            excluded = Counter()
            per_cond = defaultdict(lambda: [0, 0])
            for iid, k in key.items():
                r = to_bin(ans.get(iid, {}).get(q), partial)
                if r is None:
                    excluded[(ans.get(iid, {}).get(q) or "blank").strip().lower() or "blank"] += 1
                    continue
                sys_pairs.append((r, 1 if k["system_verdict"] == "verified" else 0))
                if k["kind"] == "reported_success":
                    auth_len.append((r, 1 if k["author_verdict"] in ("correct", "lenient") else 0))
                    auth_str.append((r, 1 if k["author_verdict"] == "correct" else 0))
                    pc = per_cond[f"{k['group']} | {k['condition']}"]
                    pc[0] += r
                    pc[1] += 1
            res[tag] = {
                "excluded_answers": dict(excluded),
                "rater_vs_system": {"n": len(sys_pairs), "confusion(rater,system)": confusion(sys_pairs), "kappa": kappa(sys_pairs)},
                "rater_vs_author_published": {"n": len(auth_len), "confusion(rater,author)": confusion(auth_len), "kappa": kappa(auth_len)},
                "rater_vs_author_strict": {"n": len(auth_str), "confusion(rater,author)": confusion(auth_str), "kappa": kappa(auth_str)},
                "rater_precision_by_condition": {c: {"correct": v[0], "reported_rated": v[1],
                                                     "precision": round(v[0] / v[1], 4) if v[1] else None,
                                                     "wilson95": wilson(v[0], v[1])} for c, v in sorted(per_cond.items())},
            }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "second_rater_scores.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "rater_precision_by_condition"})
                      for k, v in res.items()}, indent=1))
    print(f"wrote {out / 'second_rater_scores.json'}")


if __name__ == "__main__":
    main()
