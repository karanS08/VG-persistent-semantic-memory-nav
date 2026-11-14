"""Pick the image-check top-k from the calibration crops (ProcTHOR houses 2-40, never the evaluation house).
Rule fixed before looking: smallest k whose own-type pass rate is >= 0.90. Reports the wrong-query pass rate at each k
and the seating confusions (chair/armchair/stool queried as sofa, and the reverse) that caused v7's false successes."""
import collections, json

rows = json.load(open("results/post_paper/v8_image_check_calib.json"))
own = [r for r in rows if r["own"]]
wrong = [r for r in rows if not r["own"]]
SEAT = {"Chair", "ArmChair", "Stool", "Sofa"}
seat_conf = [r for r in wrong if r["type"] in SEAT and r["query"] in ("a sofa", "a chair", "an armchair", "a stool")]
print(f"crops: {len(own)} objects ({len({r['house'] for r in own})} houses); wrong-query rows {len(wrong)}; "
      f"seating confusion rows {len(seat_conf)}")
print(" k  own-pass  wrong-pass  seating-confusion-pass")
chosen = None
for k in range(1, 11):
    a = sum(r["rank"] <= k for r in own) / len(own)
    b = sum(r["rank"] <= k for r in wrong) / len(wrong)
    c = sum(r["rank"] <= k for r in seat_conf) / max(len(seat_conf), 1)
    mark = ""
    if chosen is None and a >= 0.90:
        chosen, mark = k, "  <- chosen (smallest k with own-pass >= 0.90)"
    print(f"{k:2d}   {a:.3f}     {b:.3f}       {c:.3f}{mark}")
print("\nown-pass by type at chosen k:")
by = collections.defaultdict(list)
for r in own:
    by[r["type"]].append(r["rank"] <= (chosen or 1))
for t, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    print(f"  {t:13s} n={len(v):3d} pass={sum(v)/len(v):.2f}")
json.dump({"chosen_top_k": chosen}, open("results/post_paper/v8_image_check_topk.json", "w"))
print("chosen top_k:", chosen)
