"""Real-path versions of the teaser maps (2026-10-03), in the style of the author's
mem_on.png / mem_off.png on the same base render (highres/map_topdown_house_black.png,
scaled to 1811x925). Only recorded robot movement is drawn: the path (green = memory on,
red = memory off), the start dot with its label, and one end marker where the robot stood
when it verified the object (filled dot) or when its step budget ran out (X). No star and
no leader line: DAAAM's estimate of the object's position is 1-2 m off, so marking it would
show a point in empty floor.
Usage: make_real_onoff_maps.py [on_run_dir off_run_dir query_index tag]
Default pair (query 2 = the radiator search):
  on  = phase4_batch_persist_memory_only_instance_on
  off = phase4_batch_instance_off
Both must have verified the SAME object (check_same_object.py), or memory-off must have
cleanly failed; refuses otherwise.
Out: acra_paper/figures/mem_on_<tag>.png, mem_off_<tag>.png"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "scripts")
from make_memory_onoff_figure import load  # noqa: E402
from make_teaser import to_px  # noqa: E402

A = Path("acra_paper/teaser_assets")
RUNS = {"on": Path(sys.argv[1] if len(sys.argv) > 1 else "data/runs/phase4_batch_persist_memory_only_instance_on"),
        "off": Path(sys.argv[2] if len(sys.argv) > 2 else "data/runs/phase4_batch_instance_off")}
Q = int(sys.argv[3]) if len(sys.argv) > 3 else 1      # 0-based query index (1 = radiator pair)
TAG = sys.argv[4] if len(sys.argv) > 4 else "real"
COL = {"on": (30, 132, 73), "off": (192, 57, 43)}      # green / red, as in the teaser headers
S = 2                                    # render at 2x (3622 x 1850)
W, H = 1811 * S, 925 * S


def main():
    chk = {r["episode"]: r for r in json.load(open(RUNS["on"] / "same_object_check.json"))}
    if not chk[Q]["same_object"] and not (chk[Q]["outcome_on"] == "success" and chk[Q]["outcome_off"] == "failure"):
        sys.exit(f"neither same object nor clean not-found: {chk[Q]}")
    m = json.load(open("acra_paper/figures/teaser/topdown_00770.json"))
    hr = json.load(open(A / "highres/map_topdown_house_highres.json"))
    base = Image.open(A / "highres/map_topdown_house_black.png").convert("RGB")
    k = 1811 / base.size[0] * S
    base = base.resize((W, H), Image.LANCZOS)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 46 * S)

    def px(P):
        u, v = to_px(np.asarray(P, float), m)
        return (u * hr["scale_vs_1600_render"] - hr["crop_origin_px"][0]) * k, \
               (v * hr["scale_vs_1600_render"] - hr["crop_origin_px"][1]) * k

    def tag(d, x, y, text, anchor):
        tw = d.textlength(text, font=font)
        x0 = x - tw - 30 * S if anchor == "left" else x + 30 * S
        d.rectangle([x0 - 10 * S, y - 34 * S, x0 + tw + 10 * S, y + 34 * S], fill="black")
        d.text((x0, y - 28 * S), text, fill="white", font=font)

    for key, run in RUNS.items():
        eps, paths = load(run)
        e = eps[Q]
        u, v = px(paths[Q])
        pts = list(zip(u, v))
        im = base.copy(); d = ImageDraw.Draw(im)
        d.line(pts, fill="white", width=19 * S, joint="curve")                 # halo for contrast on the floor
        d.line(pts, fill=COL[key], width=12 * S, joint="curve")
        sx, sy = u[0], v[0]
        d.ellipse([sx - 18 * S, sy - 18 * S, sx + 18 * S, sy + 18 * S], fill="black", outline="white", width=6 * S)
        tag(d, sx, sy, "start", "left")
        rx, ry = u[-1], v[-1]
        if e["outcome"] == "success":
            d.ellipse([rx - 22 * S, ry - 22 * S, rx + 22 * S, ry + 22 * S], fill=COL[key], outline="white", width=7 * S)
            tag(d, rx, ry, e["query"].split()[-1], "right")
        else:
            for a, b in (((-18 * S, -18 * S), (18 * S, 18 * S)), ((-18 * S, 18 * S), (18 * S, -18 * S))):
                d.line([(rx + a[0], ry + a[1]), (rx + b[0], ry + b[1])], fill="white", width=14 * S)
                d.line([(rx + a[0], ry + a[1]), (rx + b[0], ry + b[1])], fill=COL[key], width=8 * S)
            tag(d, rx, ry, "not found", "right")
        out = Path(f"acra_paper/figures/mem_{key}_{TAG}.png")
        im.save(out)
        print(key, e["query"], e["steps_taken"], "steps", e["outcome"], "->", out)


if __name__ == "__main__":
    main()
