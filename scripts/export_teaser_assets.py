"""Export standalone, high-resolution teaser assets + their data into
acra_paper/teaser_assets/ for a design agent (2026-09-29). Every asset is real
simulation output; see acra_paper/teaser_assets/README.md for provenance."""
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, "scripts")
from make_teaser import to_px, col  # noqa: E402
import rerun.dataframe as rd  # noqa: E402

OUT = Path("acra_paper/teaser_assets"); OUT.mkdir(exist_ok=True)
T = Path("acra_paper/figures/teaser"); LG = Path("acra_paper/figures/live_grounding")
QCOL = ["#8c8c8c", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
OFF = Path("data/runs/phase4_batch_reset5_rep2")
ON = Path("data/runs/phase4_batch_persist_memory_only_persistmemoryonly5_rep2")
TEASER = Path("data/runs/phase4_batch_persist_memory_only_teaser_memonly5")
DPI = 300


def load(run):
    eps = json.load(open(run / "batch_metrics.json"))["episodes"]
    paths = []
    for e in eps:
        rec = rd.load_recording(str(run / f"episode_{e['episode_id']}.rrd"))
        paths.append(np.array([p[0] for p in col(rec, "/world/agent/position", "Position3D") if p]))
    return eps, paths


m = json.load(open(T / "topdown_00770.json"))
top = np.array(Image.open(T / "topdown_00770.png"))
fl = top.sum(2) > 30
rr, cc = np.where(fl)
r0, r1, c0, c1 = rr.min() - 10, rr.max() + 10, cc.min() - 10, cc.max() + 10
crop = top[r0:r1, c0:c1]
Image.fromarray(top).save(OUT / "map_topdown_full.png")
Image.fromarray(crop).save(OUT / "map_topdown_house.png")
cp = lambda xyz: ((lambda p: (p[0] - c0, p[1] - r0))(to_px(xyz, m)))  # world -> house-crop pixels


def map_axes(w_in=6.0):
    h_in = w_in * crop.shape[0] / crop.shape[1]
    fig = plt.figure(figsize=(w_in, h_in)); ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(crop); ax.set_axis_off(); ax.set_xlim(0, crop.shape[1]); ax.set_ylim(crop.shape[0], 0)
    return fig, ax


# ---- memory off / on paths ----
onoff = {}
for key, run in (("off", OFF), ("on", ON)):
    eps, paths = load(run)
    fig, ax = map_axes()
    rec = []
    for i, (p, e) in enumerate(zip(paths, eps)):
        px, pz = cp(p)
        ok = e["outcome"] == "success"
        ax.plot(px, pz, color=QCOL[i], lw=0.9 if i == 0 else 2.2, alpha=0.6 if i == 0 else 1,
                ls="-" if i == 0 else (0, (3, 1.2)))
        ax.scatter(px[-1], pz[-1], marker="*" if ok else "X", s=160 if ok else 70, color=QCOL[i], ec="black", lw=0.6, zorder=6)
        rec.append({"query_index": i + 1, "query": e["query"], "steps": e["steps_taken"], "found": ok,
                    "path_px": np.round(np.c_[px, pz], 1).tolist()})
    sx, sz = cp(paths[1][0])
    ax.scatter(sx, sz, s=80, color="#1f3b73", ec="white", lw=1.5, zorder=7)
    fig.savefig(OUT / f"paths_memory_{key}.png", dpi=DPI); plt.close(fig)
    onoff[key] = {"run": str(run), "start_px": [round(sx, 1), round(sz, 1)], "queries": rec,
                  "steps_queries_2_to_5": sum(e["steps_taken"] for e in eps[1:])}
json.dump(onoff, open(OUT / "paths_memory_onoff.json", "w"), indent=1)

# ---- teaser session: objects coloured by time since last seen ----
eps, paths = load(TEASER)
snap = json.load(open(TEASER / "memory_snapshot_ep3.json"))  # before the object-number collision (README)
ents = [e for e in snap["entities"] if e["position_world"]]
age = np.array([(snap["now"] - e["last_observed"]) / 60 for e in ents])
fig, ax = map_axes()
ox, oz = cp(np.array([e["position_world"] for e in ents]))
ax.scatter(ox, oz, c=age, cmap="turbo", s=22, ec="white", lw=0.4, zorder=4)
for i, p in enumerate(paths[:4]):
    px, pz = cp(p)
    ax.plot(px, pz, color=QCOL[i], lw=0.9 if i == 0 else 2.0, alpha=0.6 if i == 0 else 1, ls="-" if i == 0 else (0, (3, 1.2)))
fig.savefig(OUT / "map_memory_objects_by_age.png", dpi=DPI); plt.close(fig)
ver = [e["verification_detail"] for e in eps]
json.dump({"run": str(TEASER), "snapshot": "memory_snapshot_ep3.json (after query 4)",
           "age_colormap": "turbo, 0 = just seen", "age_range_min": [round(float(age.min()), 2), round(float(age.max()), 2)],
           "objects": [{"caption": e["description"], "px": [round(float(x), 1), round(float(z), 1)],
                        "minutes_since_seen": round(float(a), 2), "n_obs": e["n_i"]} for e, x, z, a in zip(ents, ox, oz, age)],
           "verified_targets": [{"query": e["query"], "steps": e["steps_taken"], "caption": v["description"],
                                 "px": [round(float(q), 1) for q in cp(v["entity_position_world"])] if v.get("entity_position_world") else None}
                                for e, v in zip(eps, ver)]},
          open(OUT / "map_memory_objects.json", "w"), indent=1)

# ---- camera view + detections ----
frame = Image.open(LG / "raw_frame_matched.png").convert("RGB")
frame.save(OUT / "camera_view_raw.png")
tracks = {t["id"]: t["bbox"] for t in json.load(open(LG / "tracks_matched.json"))}
det = frame.copy(); d = ImageDraw.Draw(det)
for tid, bb in tracks.items():
    d.rectangle(bb, outline=(255, 215, 0) if tid in (3, 4) else (124, 252, 0), width=3 if tid in (3, 4) else 1)
det.save(OUT / "camera_view_detections.png")
caps = {q: v["description"] for q, v in ((e["query"], e["verification_detail"]) for e in json.load(open(ON / "batch_metrics.json"))["episodes"])}
json.dump({"image": "camera_view_raw.png (640x480)", "boxes_xyxy": tracks,
           "key_objects": {"4": {"label": "window", "daaam_caption": caps["a large window"]},
                           "3": {"label": "radiator", "daaam_caption": caps["a wall-mounted radiator"]}}},
          open(OUT / "camera_view_detections.json", "w"), indent=1)

# ---- object crops ----
fa = np.array(frame); ent_img = np.array(Image.open("acra_paper/figures/room_survey_final/room1_entrance.png").convert("RGB"))
b3, b4 = tracks[3], tracks[4]
for name, im in (("radiator", fa[int(b3[1]) - 14:int(b3[3]) + 6, int(b3[0]) - 12:int(b3[2]) + 12]),
                 ("window", fa[int(b4[1]):int(b4[3]), int(b4[0]):int(b4[2])]),
                 ("bag", ent_img[355:455, 275:335]), ("cushion", ent_img[378:440, 130:285])):
    Image.fromarray(im).resize((im.shape[1] * 3, im.shape[0] * 3), Image.LANCZOS).save(OUT / f"object_{name}.png")

# ---- robot ----
shutil.copy("acra_paper/figures/spot_third_person.png", OUT / "robot_third_person.png")

# ---- decay curves (Eq. 1) ----
mob = json.load(open("configs/mobility_priors.json"))["categories"]
rel = yaml.safe_load(open("configs/memory.yaml"))["reliability"]
t = np.logspace(np.log10(60), np.log10(86400 * 3), 120)
base = min(1.0, 1 - math.exp(-rel["alpha"]))
with open(OUT / "decay_curves.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seconds_since_seen", "bag_0.80", "cushion_0.55", "radiator_0.03"])
    for s in t:
        w.writerow([round(float(s), 1)] + [round(base * math.exp(-rel["lambda_base"] * mob[n] * s), 6) for n in ("bag", "cushion", "radiator")])
fig, ax = plt.subplots(figsize=(3.2, 2.2))
for n, c in (("bag", "#c0392b"), ("cushion", "#2f6db3"), ("radiator", "#2e8b57")):
    ax.plot(t / 3600, base * np.exp(-rel["lambda_base"] * mob[n] * t), color=c, lw=2, label=f"{n} (mobility {mob[n]:.2f})")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 0.6)
ax.set_xticks([1 / 60, 1, 24]); ax.set_xticklabels(["1 min", "1 h", "1 day"])
ax.set_xlabel("time since last seen"); ax.set_ylabel("reliability $C_i$"); ax.legend(frameon=False, fontsize=7)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT / "decay_curves.png", dpi=DPI); plt.close(fig)

# ---- 3D memory scatter ----
P = np.array([e["position_world"] for e in ents])
fig = plt.figure(figsize=(3, 3)); ax = fig.add_axes([0, 0, 1, 1], projection="3d")
ax.scatter(P[:, 0], P[:, 2], P[:, 1], c=age, cmap="turbo", s=14, depthshade=False)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(28, -60)
fig.savefig(OUT / "memory_3d_scatter.png", dpi=DPI, transparent=True); plt.close(fig)

# ---- memory entry card values ----
snap1 = json.load(open(TEASER / "memory_snapshot_ep1.json"))
v1 = ver[1]; e1 = next(e for e in snap1["entities"] if e["id"] == v1["entity_id"])
json.dump({"object": "radiator", "caption": v1["description"], "position_world_m": e1["position_world"],
           "observations_n_i": e1["n_i"], "mobility_prior": mob["radiator"], "reliability_C_i_at_verification": v1["reliability"],
           "relevance_to_query": v1["relevance"], "minutes_since_seen_at_query_2": round((snap1["now"] - e1["last_observed"]) / 60, 1),
           "source": f"{TEASER}/memory_snapshot_ep1.json + batch_metrics.json"}, open(OUT / "memory_entry_radiator.json", "w"), indent=1)

# ---- references ----
for src, dst in (("acra_paper/figures/fig_teaser.pdf", "reference_draft_teaser_3band.pdf"),
                 ("acra_paper/figures/teaser/fig_teaser_v1.pdf", "reference_draft_teaser_v1.pdf"),
                 ("/home/ksharma/.claude/uploads/228ff7f4-4e94-4d56-9832-54bf4b393403/43c11c8b-image.png", "reference_concept_mockup.png")):
    if Path(src).exists():
        shutil.copy(src, OUT / dst)
print(sorted(p.name for p in OUT.iterdir()))
print(json.dumps({k: onoff[k]["steps_queries_2_to_5"] for k in onoff}), "house crop px", crop.shape)
