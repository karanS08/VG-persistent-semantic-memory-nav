"""Test 3.4 / Task A.3: build a BLIND second-rater audit package.

HM3D semantic ground truth does not exist for scenes 00770/00337 (see
gt_semantic_availability.py), so an independent check has to be a second human
rater. This script builds the package for that rater. It does NOT rate anything.

Items
  * every reported success (system said "verified") in the runs behind every
    audited rate in acra_paper/main.tex (registry: audit_registry.GROUPS);
  * decoys: a seeded random sample of FAILED episodes' best-relevance memory entry
    (system said "not verified"), so the rater cannot assume every item is a
    system "yes", and so the system's verdict has a real negative class.

Per item the rater sees: the query, and (where a .rrd recording exists)
  - the RGB frame at the verification step (last logged step of the episode),
  - the frame where the entity's 3D position projects best into the camera
    (depth-checked for occlusion), with a marker + crop,
and, hidden behind a click, the DAAAM caption. Condition / run / system verdict /
author verdict are NOT shown; they go to a separate key file.

Rerun reading API used (rerun-sdk 0.19.1, verified against the installed package):
  rerun.dataframe.load_recording(path).view(index="step", contents={...})
      .filter_index_values([...]).select().read_all() -> pyarrow.Table
Entity paths (verified from the .rrd schema, logged by sim_bridge/rerun_stream.py and
orchestrator/run_phase4_batch.py): /world/agent {Translation3D, RotationQuat},
/world/agent/rgb, /world/agent/depth {ImageBuffer, ImageFormat},
/world/daaam/objects {Position3D, Text=description[:60]}.

Coordinate-frame handling is checked, not assumed: for every located entity the
projected pixel's rendered depth is compared with the entity's camera-frame
depth, for two candidate sensor offsets (0 m and configs/habitat.yaml sensor.height),
and the agreement rates are written to the key directory.

Usage: ~/miniconda3/envs/rpv_mem/bin/python scripts/postdraft/build_blind_audit_package.py
"""
import csv
import html
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_registry import ADVERSARIAL, AUDIT_SOURCES, GROUPS, ROOT, author_verdict, run_path  # noqa: E402

import rerun.dataframe as rrd  # noqa: E402

OUT = ROOT / "data/results_postdraft/gt_audit_precision"
PKG = OUT / "blind_audit_package"
KEY = OUT / "audit_key_DO_NOT_SHARE"
SEED = 20260928
N_DECOYS = 50

HAB = yaml.safe_load(open(ROOT / "configs/habitat.yaml"))
H, W = HAB["sensor"]["rgb_resolution"]
HFOV = float(HAB["sensor"]["hfov"])
SENSOR_H = float(HAB["sensor"]["height"])
FX = (W / 2.0) / np.tan(np.deg2rad(HFOV) / 2.0)
CX, CY = W / 2.0, H / 2.0

# rerun ChannelDatatype enum values (rerun-sdk 0.19 datatypes/channel_datatype.py)
CHANNEL_DT = {6: np.uint8, 7: np.int8, 8: np.uint16, 9: np.int16, 10: np.uint32, 11: np.int32,
              12: np.uint64, 13: np.int64, 33: np.float16, 34: np.float32, 35: np.float64}


def _cell(v):
    """rerun dataframe cells are object arrays holding one array per row."""
    if v is None:
        return None
    if isinstance(v, np.ndarray) and v.dtype == object:
        return v[0] if len(v) else None
    return v


def decode_image(buf, fmt):
    fmt = _cell(fmt) if not isinstance(fmt, dict) else fmt
    raw = np.asarray(_cell(buf), dtype=np.uint8).tobytes()
    dt = CHANNEL_DT[fmt["channel_datatype"]]
    arr = np.frombuffer(raw, dtype=dt)
    h, w = fmt["height"], fmt["width"]
    ch = arr.size // (h * w)
    return arr.reshape(h, w, ch) if ch > 1 else arr.reshape(h, w)


class Episode:
    def __init__(self, rrd_path):
        self.rec = rrd.load_recording(str(rrd_path))
        t = self.rec.view(index="step", contents={"/world/agent": ["Translation3D", "RotationQuat"]}).select().read_all()
        df = t.to_pandas()
        self.steps = df["step"].to_numpy().astype(int)
        self.trans = np.stack([np.asarray(_cell(x), dtype=float) for x in df["/world/agent:Translation3D"]])
        self.quat = np.stack([np.asarray(_cell(x), dtype=float) for x in df["/world/agent:RotationQuat"]])  # xyzw
        self.final_step = int(self.steps.max())
        o = self.rec.view(index="step", contents={"/world/daaam/objects": ["Position3D", "Text"]}).select().read_all().to_pandas()
        o = o[o["step"] <= self.final_step]
        last = o.iloc[-1] if len(o) else None
        self.obj_labels = list(last["/world/daaam/objects:Text"]) if last is not None else []
        self.obj_pos = [np.asarray(p, dtype=float) for p in last["/world/daaam/objects:Position3D"]] if last is not None else []

    def agent_at(self, step):
        i = int(np.where(self.steps == step)[0][0])
        return self.trans[i], self.quat[i]

    def locate(self, description, target_dist=None):
        """Return (position, n_label_matches, chosen_by)."""
        pref = (description or "")[:60]
        cands = [p for l, p in zip(self.obj_labels, self.obj_pos) if l == pref]
        if not cands:
            return None, 0, "no_label_match"
        fa, _ = self.agent_at(self.final_step)
        dxz = [float(np.hypot(p[0] - fa[0], p[2] - fa[2])) for p in cands]
        if len(cands) == 1:
            return cands[0], 1, "unique_label"
        if target_dist is not None:
            k = int(np.argmin([abs(d - target_dist) for d in dxz]))
            return cands[k], len(cands), f"label+verification_distance(|err|={abs(dxz[k] - target_dist):.3f}m)"
        k = int(np.argmin(dxz))
        return cands[k], len(cands), "label+nearest_to_final_agent_pose"

    def project(self, p_world, step, sensor_h):
        t, q = self.agent_at(step)
        cam = t + np.array([0.0, sensor_h, 0.0])
        R = Rotation.from_quat(q).as_matrix()
        pc = R.T @ (p_world - cam)  # habitat camera: x right, y up, looks along -z
        z = -pc[2]
        if z <= 1e-3:
            return None
        u = CX + FX * pc[0] / z
        v = CY - FX * pc[1] / z
        return u, v, z

    def images(self, step):
        v = self.rec.view(index="step", contents={"/world/agent/rgb": ["ImageBuffer", "ImageFormat"],
                                                  "/world/agent/depth": ["ImageBuffer", "ImageFormat"]})
        df = v.filter_index_values([int(step)]).select().read_all().to_pandas()
        r = df.iloc[0]
        rgb = decode_image(r["/world/agent/rgb:ImageBuffer"], r["/world/agent/rgb:ImageFormat"])
        dep = decode_image(r["/world/agent/depth:ImageBuffer"], r["/world/agent/depth:ImageFormat"])
        return rgb[..., :3], dep


def depth_status(dep, u, v, z):
    """Compare rendered depth at the projected pixel with the entity's camera-frame depth.
    consistent: |d - z| <= tol ; see_through: d > z + tol (nothing in front of the point, the
    ray passes the recorded centroid, typical for thin/low objects) ; occluded: d < z - tol."""
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < dep.shape[1] and 0 <= vi < dep.shape[0]):
        return "off_image", None
    patch = dep[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3].astype(float)
    patch = patch[patch > 0]
    if patch.size == 0:
        return "no_depth", None
    d = float(np.median(patch))
    tol = max(0.5, 0.25 * z)
    if abs(d - z) <= tol:
        return "consistent", d
    return ("see_through", d) if d > z else ("occluded", d)


def depth_ok(dep, u, v, z):
    s, d = depth_status(dep, u, v, z)
    return s == "consistent", d


def yaw_null(ep, pos, step, sh, deg=30.0):
    """Null model for the frame check: the same point rotated by `deg` about the camera's vertical axis."""
    t, q = ep.agent_at(step)
    cam = t + np.array([0.0, sh, 0.0])
    R = Rotation.from_quat(q).as_matrix()
    pc = Rotation.from_euler("y", deg, degrees=True).as_matrix() @ (R.T @ (pos - cam))
    z = -pc[2]
    if z <= 1e-3:
        return None
    return CX + FX * pc[0] / z, CY - FX * pc[1] / z, z


def best_view(ep, pos):
    """Pick the step whose camera sees `pos` best; returns dict with frame choice + frame checks."""
    out = {"frame_check": {}}
    for sh in (0.0, SENSOR_H):
        cands = []
        for s in ep.steps:
            pr = ep.project(pos, int(s), sh)
            if pr is None:
                continue
            u, v, z = pr
            if 40 <= u <= W - 40 and 40 <= v <= H - 40 and 0.5 <= z <= 5.0:
                cands.append((abs(z - 2.0) + abs(u - CX) / W, int(s), u, v, z))
        cands.sort()
        out["frame_check"][f"sensor_h={sh}"] = {"n_steps_in_view": len(cands)}
        out[f"cands_{sh}"] = cands
    return out


def select_view(ep, pos):
    """Choose the best unoccluded view of `pos` in episode `ep`; also returns frame-check counts
    (depth status at the top-8 candidate views, true pose vs a 30-degree yaw-perturbed null on the
    same images). Sensor offset 1.5 m comes from sim_bridge/habitat_env.py (rgb_spec.position =
    [0, height, 0]); 0 m is reported for contrast."""
    bv = best_view(ep, pos)
    chosen, fallback, checks = None, None, {}
    for sh in (SENSOR_H, 0.0):
        top = bv[f"cands_{sh}"][:8]
        cnt = {"consistent": 0, "see_through": 0, "occluded": 0, "other": 0}
        nul = {"consistent": 0, "see_through": 0, "occluded": 0, "other": 0}
        for _, s, u, v, z in top:
            rgb, dep = ep.images(s)
            st, dmeas = depth_status(dep, u, v, z)
            cnt[st if st in cnt else "other"] += 1
            pn = yaw_null(ep, pos, s, sh)
            sn = depth_status(dep, *pn)[0] if pn else "off_image"
            nul[sn if sn in nul else "other"] += 1
            if sh == SENSOR_H:
                if st == "consistent" and chosen is None:
                    chosen = (s, u, v, z, dmeas, rgb, "depth_consistent")
                elif st == "see_through" and fallback is None:
                    fallback = (s, u, v, z, dmeas, rgb, "unoccluded_see_through")
        checks[f"frame_check_sensor_h={sh}"] = {"n": len(top), "true_pose": cnt, "yaw30_null": nul}
    chosen = chosen or fallback
    if chosen is None and bv[f"cands_{SENSOR_H}"]:
        _, s, u, v, z = bv[f"cands_{SENSOR_H}"][0]
        rgb, dep = ep.images(s)
        chosen = (s, u, v, z, depth_status(dep, u, v, z)[1] or -1.0, rgb, "possibly_occluded")
    return chosen, checks


def draw(rgb, u, v, color=(255, 0, 255)):
    im = Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8)))
    d = ImageDraw.Draw(im)
    for r in (14, 16):
        d.ellipse([u - r, v - r, u + r, v + r], outline=color, width=2)
    d.line([u - 26, v, u - 18, v], fill=color, width=2)
    d.line([u + 18, v, u + 26, v], fill=color, width=2)
    return im


def crop(im, u, v, half=120):
    x0 = int(min(max(0, u - half), W - 2 * half))
    y0 = int(min(max(0, v - half), H - 2 * half))
    return im.crop((x0, y0, x0 + 2 * half, y0 + 2 * half))


def collect_items():
    items, decoy_pool = [], []
    for group, entries in GROUPS.items():
        for cond, short, role in entries:
            d = json.load(open(run_path(short)))
            for e in d["episodes"]:
                vd = e.get("verification_detail") or {}
                rrd_path = ROOT / e["rrd_path"] if e.get("rrd_path") else None
                base = dict(group=group, condition=cond, run=f"phase4_batch_{short}", run_short=short, role=role,
                            episode_id=e["episode_id"], query=e["query"], steps_taken=e["steps_taken"],
                            spl=e["spl"], persistent=bool(d.get("persistent_across_queries")),
                            adversarial_query=e["query"] in ADVERSARIAL,
                            rrd=str(rrd_path.relative_to(ROOT)) if rrd_path and rrd_path.exists() else None)
                if e["outcome"] == "success":
                    av, note = author_verdict(short, e["episode_id"])
                    items.append(dict(base, kind="reported_success", system_verdict="verified",
                                      author_verdict=av, author_note=note, author_source=AUDIT_SOURCES[short],
                                      description=vd.get("description"), entity_id=vd.get("entity_id"),
                                      relevance=vd.get("relevance"), verify_distance_m=vd.get("distance")))
                elif vd.get("best_relevance_description"):
                    decoy_pool.append(dict(base, kind="decoy_best_unverified", system_verdict="not_verified",
                                           author_verdict="not_audited", author_note="", author_source="",
                                           description=vd["best_relevance_description"], entity_id=None,
                                           relevance=vd.get("best_relevance_in_memory"), verify_distance_m=None))
    rng = random.Random(SEED)
    decoys = rng.sample(decoy_pool, min(N_DECOYS, len(decoy_pool)))
    return items, decoys, len(decoy_pool)


def main():
    (PKG / "images").mkdir(parents=True, exist_ok=True)
    KEY.mkdir(parents=True, exist_ok=True)
    items, decoys, n_pool = collect_items()
    allitems = items + decoys
    rng = random.Random(SEED + 1)
    rng.shuffle(allitems)
    for i, it in enumerate(allitems):
        it["item_id"] = f"R{i + 1:03d}"

    frame_log = []
    cache = {}
    for it in allitems:
        it["images"] = []
        if not it["rrd"]:
            continue
        try:
            if it["rrd"] not in cache:
                cache.clear()
                cache[it["rrd"]] = Episode(ROOT / it["rrd"])
            ep = cache[it["rrd"]]
            pos, nmatch, how = ep.locate(it["description"], it["verify_distance_m"])
            rec = {"item_id": it["item_id"], "rrd": it["rrd"], "final_step": ep.final_step,
                   "label_matches": nmatch, "located_by": how,
                   "entity_position_world": None if pos is None else [round(float(x), 3) for x in pos]}
            if True:  # last frame for every item (successes AND decoys) so image presence does not leak the system verdict
                rgb, dep = ep.images(ep.final_step)
                im = Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8)))
                if pos is not None:
                    pr = ep.project(pos, ep.final_step, SENSOR_H)
                    if pr and 0 <= pr[0] < W and 0 <= pr[1] < H:
                        im = draw(rgb, pr[0], pr[1])
                        rec["final_frame_marker_in_view"] = True
                    else:
                        rec["final_frame_marker_in_view"] = False
                fn = f"{it['item_id']}_last_frame.jpg"
                im.save(PKG / "images" / fn, quality=90)
                it["images"].append(("Last frame of the search (marker = recorded object position, if in view)", fn))
            if pos is not None:
                chosen, checks = select_view(ep, pos)
                rec.update(checks)
                if chosen is None:
                    # static scene, one world frame per scene: look for a view in the run's other episodes
                    run_dir = (ROOT / it["rrd"]).parent
                    others = sorted(run_dir.glob("episode_*.rrd"), key=lambda q: abs(int(q.stem.split("_")[1]) - it["episode_id"]))
                    for q in others:
                        if q.name == Path(it["rrd"]).name:
                            continue
                        chosen, _ = select_view(Episode(q), pos)
                        if chosen:
                            rec["view_from_other_episode"] = str(q.relative_to(ROOT))
                            break
                if chosen:
                    s, u, v, z, dmeas, rgb, status = chosen
                    im = draw(rgb, u, v)
                    fn = f"{it['item_id']}_best_view.jpg"
                    im.save(PKG / "images" / fn, quality=90)
                    fc = f"{it['item_id']}_best_view_crop.jpg"
                    crop(Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8))), u, v).resize((360, 360)).save(PKG / "images" / fc, quality=92)
                    it["images"].append(("Best view of the recorded object position (magenta marker)", fn))
                    it["images"].append(("Crop around the marker (unmarked)", fc))
                    rec.update(best_view_step=int(s), best_view_px=[round(u, 1), round(v, 1)],
                               best_view_expected_depth_m=round(z, 2), best_view_measured_depth_m=round(dmeas, 2),
                               best_view_status=status)
                else:
                    rec["best_view_status"] = "never_in_view"
            frame_log.append(rec)
            print(it["item_id"], it["rrd"], rec.get("best_view_status"),
                  json.dumps(rec.get(f"frame_check_sensor_h={SENSOR_H}")), flush=True)
        except Exception as ex:  # keep going; record the failure
            frame_log.append({"item_id": it["item_id"], "rrd": it["rrd"], "error": repr(ex)})
            print("ERROR", it["item_id"], repr(ex), flush=True)

    # ---- key (not for the rater) ----
    key_fields = ["item_id", "kind", "group", "condition", "run", "run_short", "role", "episode_id", "query",
                  "adversarial_query", "persistent", "steps_taken", "spl", "system_verdict", "author_verdict",
                  "author_note", "author_source", "entity_id", "relevance", "verify_distance_m", "description", "rrd"]
    with open(KEY / "key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=key_fields, extrasaction="ignore")
        w.writeheader()
        for it in sorted(allitems, key=lambda x: x["item_id"]):
            w.writerow(it)
    (KEY / "frame_extraction_log.json").write_text(json.dumps(
        {"camera": {"W": W, "H": H, "hfov_deg": HFOV, "fx": FX, "sensor_height_m": SENSOR_H},
         "seed": SEED, "n_decoy_pool": n_pool, "n_decoys": len(decoys), "items": frame_log}, indent=1))

    # ---- rater package ----
    with open(PKG / "answers_template.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "q1_image", "q2_caption", "comment", "rater_id"])
        for it in sorted(allitems, key=lambda x: x["item_id"]):
            w.writerow([it["item_id"], "" if it["images"] else "na", "", "", ""])
    rows = []
    for it in sorted(allitems, key=lambda x: x["item_id"]):
        imgs = "".join(f'<figure><img src="images/{fn}" loading="lazy"><figcaption>{html.escape(c)}</figcaption></figure>'
                       for c, fn in it["images"]) or "<p class='noimg'>No recording saved for this item: answer Q1 = na, rate Q2 only.</p>"
        rows.append(f"""<section id="{it['item_id']}"><h2>{it['item_id']} &mdash; query: <q>{html.escape(it['query'])}</q></h2>
<div class="imgs">{imgs}</div>
<p><b>Q1 (image, answer BEFORE opening the caption):</b> is the object at the marker a <q>{html.escape(it['query'])}</q>?</p>
<details><summary>Show the robot's caption for this object (then answer Q2)</summary><blockquote>{html.escape(it['description'] or '')}</blockquote>
<p><b>Q2 (caption):</b> is the object this caption is <i>about</i> a <q>{html.escape(it['query'])}</q>?</p></details></section>""")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind object audit</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161616;--fg:#eee;--mut:#aaa;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:0 16px 60px}}
section{{border-top:1px solid var(--line);padding:12px 0}} h2{{font-size:17px}}
.imgs{{display:flex;flex-wrap:wrap;gap:8px}} figure{{margin:0;max-width:100%}} img{{max-width:100%;width:420px;height:auto;display:block}}
figcaption,.noimg{{color:var(--mut);font-size:13px}} blockquote{{margin:6px 0;padding:6px 10px;border-left:3px solid var(--line)}}
</style></head><body><h1>Blind object audit ({len(allitems)} items)</h1>
<p>Instructions: see <code>RATER_INSTRUCTIONS.md</code>. The magenta ring is the robot\'s own approximate position estimate (can be ~1 m off). Record answers in <code>answers_template.csv</code>.
Q1 values: <b>yes</b> / <b>no</b> / <b>partial</b> (query object is present but is not the main object marked) / <b>unsure</b> / <b>not_visible</b> / <b>na</b>.
Q2 values: <b>yes</b> / <b>no</b> / <b>partial</b> / <b>unsure</b>.</p>
{''.join(rows)}</body></html>"""
    (PKG / "rater_sheet.html").write_text(page)
    (PKG / "RATER_INSTRUCTIONS.md").write_text(f"""# Blind object audit: instructions for the second rater

You will judge {len(allitems)} items. Each item is one object that a robot's memory proposed for a text query
(for example "a doorway") in a simulated house. Some items were accepted by the robot and some were rejected.
The mix and the order are randomised. You are not told which is which, and you should not try to work it out.

For each item, open `rater_sheet.html` in a browser and:

1. **Q1 (image).** Look at the images first. The magenta ring marks the 3D position the robot recorded for
   the object, projected into a camera frame. Answer whether the object at the ring is an instance of the queried
   category:
   `yes`, `no`, `partial` (the queried object is visible but is only part of or next to what is marked, e.g. a
   cushion lying on a bench when the ring marks the bench), `unsure`, or `not_visible` (the ring is on a wall,
   the view is blocked, or the object cannot be made out). If the item has no images, write `na`.
   **The ring is approximate.** It is the robot's own stored 3D estimate projected into the image, and spot checks show it can be about 1 m away from the object it refers to (for example on the wall next to a radiator). Judge the object nearest the ring that the ring plausibly refers to. If no object near the ring is identifiable, answer `not_visible`.
2. **Q2 (caption).** Only after answering Q1, open "Show the robot's caption" and answer whether the object the
   caption is about is an instance of the queried category: `yes` / `no` / `partial` / `unsure`.
   Judge the caption's main subject. Treat a word that only appears as a detail (for example "a bed ... with a pillow"
   for the query "a pillow") as `partial`.
3. Adjectives in the query ("wooden", "soft", "wall-mounted", "large", "framed") are secondary. Answer `yes` if
   the category is right and the adjective is not clearly contradicted. Mention adjective problems in `comment`.
4. Do not change an answer after you open the caption. If the caption changes your view, say so in `comment`.
5. Fill `answers_template.csv` (keep `item_id`; put your initials in `rater_id`) and return only that file.

Please do not open anything outside this folder. The answer key is kept separately, and scoring is done by
`scripts/postdraft/score_blind_audit.py`.
""")
    print(f"items: {len(items)} reported successes + {len(decoys)} decoys (pool {n_pool}); package: {PKG}")


if __name__ == "__main__":
    main()
