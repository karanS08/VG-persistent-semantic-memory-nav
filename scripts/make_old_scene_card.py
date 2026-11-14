"""Observe-card image from the ORIGINAL teaser scene (window + radiator room, same pose as the Remember/Navigate cards)
(2026-10-03): 3840x2880 render (render_hires_sources.py) with DAAAM's boxes for the window (track 4) and radiator (track 3)
from acra_paper/figures/live_grounding/tracks_matched.json (640x480 -> x6), large labels for a ~1.1 in card.
Out: acra_paper/figures/old_scene_card.png"""
import json
from PIL import Image, ImageDraw, ImageFont
LG = "acra_paper/figures/live_grounding"
im = Image.open(f"{LG}/raw_frame_matched_6x.png").convert("RGB"); T = {t["id"]: t["bbox"] for t in json.load(open(f"{LG}/tracks_matched.json"))}
d = ImageDraw.Draw(im, "RGBA"); f = ImageFont.truetype("DejaVuSans-Bold.ttf", 190)
for tid, name, col, side in ((4, "Window", "#17becf", "above"), (3, "Radiator", "#f28e2b", "right")):
    x0, y0, x1, y1 = [v * 6 for v in T[tid]]
    d.rectangle([x0, y0, x1, y1], outline=col, width=30)
    tw = d.textlength(name, font=f); cx = (x0 + x1) / 2
    if side == "right":
        ty = (y0 + y1) / 2 - 112; lx = x1 + 70
    else:
        ty = y0 - 250; lx = max(30, min(cx - tw / 2, im.width - tw - 70))
    d.rectangle([lx - 35, ty, lx + tw + 35, ty + 225], fill=col); d.text((lx, ty + 8), name, fill="white", font=f)
im.save("acra_paper/figures/old_scene_card.png"); print("saved", im.size)
