"""Two-sided permutation tests (episode level, 100k shuffles) on the
factorial summary, queries 1-4 only (query 0 is identical by construction)."""
import json, random, statistics as st
S = json.load(open("results/post_paper/factorial_summary.json"))
def eps(c): return [e for e in S["conditions"][c]["episodes"] if not e["key"].endswith(".0")]
def perm(x, y, n=100000, seed=0):
    rng = random.Random(seed); obs = st.mean(x) - st.mean(y); pool = x + y; k = len(x); hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:k]) - st.mean(pool[k:])) >= abs(obs) - 1e-12: hits += 1
    return round(obs, 3), round((hits + 1) / (n + 1), 4)
res = {}
for a, b in [("B", "A"), ("C", "A"), ("D", "A"), ("B", "D")]:
    for m, f in [("strict_success", lambda e: float(e["strict"])), ("audited_spl", lambda e: e["spl"] * e["strict"]),
                 ("steps", lambda e: float(e["steps"]))]:
        d, p = perm([f(e) for e in eps(a)], [f(e) for e in eps(b)])
        res[f"{a}_vs_{b}_{m}"] = {"diff": d, "p": p}
        print(f"{a} vs {b} {m:15s} diff={d:8.3f} p={p}")
json.dump(res, open("results/post_paper/factorial_permtests.json", "w"), indent=1)
