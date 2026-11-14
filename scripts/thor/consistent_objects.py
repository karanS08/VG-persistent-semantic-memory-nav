"""Which THOR objects does DAAAM label correctly in EVERY independent mapping run of ProcTHOR house 1?

For each run (memory_snapshot_ep0.json + matching THOR ground-truth gt_*.json) an object o counts as 'recalled' when some
memory entity (a) lies within TOL m of o's xz footprint and (b) names o's type (synonym list below) in the first HEAD characters
of its caption. Objects are then ranked by the fraction of runs recalling them. Also reports per-type 'confusers' (entities
naming the type but >1.5 m from every instance) so a target's retrieval ambiguity is visible.
Excluded runs: camera-unfixed (camfix0), seeded (tbs_*), smoke tests, and runs whose type-matched median error is > 1.5 m.
"""
import glob, json, math, os, re, sys, collections

TOL, HEAD = 0.5, 70
RUNS = "data/runs"
SYN = {"Fridge": ["refrigerator", "fridge"], "Television": ["television", "tv ", "flat-screen", "flat screen", "monitor"],
       "Sofa": ["sofa", "couch"], "ArmChair": ["armchair", "arm chair", "chair"], "Chair": ["chair"],
       "Pillow": ["pillow", "cushion"], "GarbageCan": ["trash can", "garbage can", "waste", "bin"], "CellPhone": ["phone"],
       "DiningTable": ["table"], "TVStand": ["tv stand", "stand", "console"], "HousePlant": ["plant"], "FloorLamp": ["lamp"],
       "DeskLamp": ["lamp"], "SinkBasin": ["sink", "basin"], "Sink": ["sink"], "Bed": ["bed"], "DogBed": ["dog bed", "pet bed"],
       "Toilet": ["toilet"], "ToiletPaper": ["toilet paper"], "Dresser": ["dresser", "chest of drawers"], "Drawer": ["drawer"],
       "Shelf": ["shelf", "shelving", "bookcase"], "Statue": ["statue", "sculpture", "figurine"], "Painting": ["painting"],
       "Laptop": ["laptop"], "Candle": ["candle"], "Bowl": ["bowl"], "Book": ["book"], "Plate": ["plate"], "Vase": ["vase"],
       "Box": ["box"], "Mug": ["mug", "cup"], "Kettle": ["kettle"], "Pan": ["pan"], "WashingMachine": ["washing machine", "washer"],
       "CounterTop": ["countertop", "counter"], "RemoteControl": ["remote"], "Watch": ["watch"], "AlarmClock": ["alarm clock", "clock"],
       "TeddyBear": ["teddy", "stuffed"], "Faucet": ["faucet", "tap"], "Pen": ["pen "], "Pencil": ["pencil"],
       "TissueBox": ["tissue"], "SprayBottle": ["spray"], "SoapBottle": ["soap"], "SoapBar": ["soap"], "WineBottle": ["wine"],
       "Newspaper": ["newspaper"], "BaseballBat": ["bat"], "TennisRacket": ["racket", "racquet"], "Plunger": ["plunger"],
       "PaperTowelRoll": ["paper towel"], "GarbageBag": ["bag"], "Apple": ["apple"], "Tomato": ["tomato"], "Potato": ["potato"],
       "Egg": ["egg"], "Ladle": ["ladle"], "Spoon": ["spoon"], "ButterKnife": ["knife"], "SaltShaker": ["salt"], "KeyChain": ["key"],
       "CreditCard": ["card"], "CD": ["cd", "disc"], "DishSponge": ["sponge"]}
SKIP = {"Wall", "Floor", "Doorway", "Doorframe", "Window", "Door", "Ceiling"}


def dist_to_box(p, o):
    cx, cz = o["pos_h"][0], o["pos_h"][2]
    return math.hypot(max(abs(p[0] - cx) - o["size"]["x"] / 2, 0), max(abs(p[2] - cz) - o["size"]["z"] / 2, 0))


def names(desc, typ):
    return any(k in (desc[:HEAD].lower() + " ") for k in SYN.get(typ, []))


def runs():
    out = []
    for gdir in sorted(glob.glob(f"{RUNS}/thor_gt_*") + glob.glob(f"{RUNS}/_*/thor_gt_*")):
        tag = os.path.basename(gdir)[len("thor_gt_"):]
        if tag.startswith(("camfix0", "tbs_", "smoke")): continue
        base = os.path.dirname(gdir)
        snaps = glob.glob(f"{base}/phase4_batch_*{tag}/memory_snapshot_ep0.json")
        gts = sorted(glob.glob(f"{gdir}/gt_*.json"))
        if tag == "tour_house1": snaps = glob.glob(f"{RUNS}/phase4_batch_persist_memory_only_tour_house1/memory_snapshot_ep0.json")
        if snaps and gts: out.append((tag, snaps[0], gts[0]))
    return out


def main():
    per_obj, accs, used = collections.defaultdict(dict), {}, []
    for tag, sp, gp in runs():
        gt = json.load(open(gp)); snap = json.load(open(sp))
        ents = [e for e in (snap["entities"] if isinstance(snap, dict) else snap) if e.get("position_world")]
        # sanity: type-matched median error (rejects mirrored / broken-frame runs)
        errs = []
        for e in ents:
            for o in gt:
                if o["type"] not in SKIP and max(o["size"].values()) < 1.5 and names(e.get("description", ""), o["type"]):
                    errs.append(min(dist_to_box(e["position_world"], q) for q in gt if q["type"] == o["type"])); break
        med = sorted(errs)[len(errs) // 2] if errs else 99
        accs[tag] = (len(ents), len(errs), round(med, 2))
        if med > 1.5: continue
        used.append(tag)
        moved = set(re.findall(r"(ArmChair|TeddyBear|WashingMachine)", tag)) if ("mv_" in tag or "tb_" in tag or "wm_" in tag) else set()
        for o in gt:
            if o["type"] in SKIP or o["type"] in moved: continue
            hit = any(dist_to_box(e["position_world"], o) <= TOL and names(e.get("description", ""), o["type"]) for e in ents)
            per_obj[o["id"]][tag] = hit
    print("runs (entities, type-matched, median err m):"); [print(f"  {t:28s} {v}") for t, v in accs.items()]
    print(f"\nusable independent runs: {len(used)} -> {used}\n")
    gt0 = {o["id"]: o for o in json.load(open(runs()[0][2]))}
    rows = []
    for oid, d in per_obj.items():
        n = len(d); k = sum(d.values()); o = gt0.get(oid)
        if not o or n < 2: continue
        rows.append((k / n, n, oid, o["type"], max(o["size"].values()), o["pos_h"]))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    types = collections.Counter(o["type"] for o in gt0.values())
    print("fraction of runs recalling each object (n runs where scored):")
    for f, n, oid, t, sz, pos in rows:
        if f >= 0.5: print(f"  {f:4.2f} ({int(round(f*n))}/{n}) {oid:28s} {t:14s} size {sz:.2f} m  instances_of_type={types[t]}  pos=({pos[0]:.1f},{pos[2]:.1f})")
    json.dump([dict(frac=r[0], n=r[1], id=r[2], type=r[3], size=r[4], pos_h=r[5]) for r in rows], open("results/post_paper/thor_consistent_objects.json", "w"), indent=1)


if __name__ == "__main__":
    main()
