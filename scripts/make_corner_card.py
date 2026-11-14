"""Corner-view card image for teaser box 1 (2026-10-03): crop of acra_paper/figures/corner_grounding/raw_6x.png to the two
windows and three radiators, with DAAAM's boxes (acra_paper/figures/corner_grounding/tracks.json) and large labels sized for
a ~1.1 in wide card. Only boxes whose caption was checked against the pixels (see match_corner_boxes.py) are labelled.
Out: acra_paper/figures/corner_view_card.png"""
import json
from PIL import Image, ImageDraw, ImageFont
D = "acra_paper/figures/corner_grounding"
im = Image.open(f"{D}/raw_6x.png").convert("RGB"); T = json.load(open(f"{D}/tracks.json"))
KEEP = {"13": ("Radiator", "#f28e2b", "below"), "11": ("Radiator", "#f28e2b", "below"), "3": ("Radiator", "#f28e2b", "below"),
        "12": ("Window", "#17becf", "above"), "5": ("Window", "#17becf", "above")}
CROP = (220, 760, 2660, 2540)          # x0, y0, x1, y1 on the 1920 x 1440 render (aspect 1.37 = the card)
d = ImageDraw.Draw(im, "RGBA"); f = ImageFont.truetype("DejaVuSans-Bold.ttf", 140)
for k, (name, col, side) in KEEP.items():
    x0, y0, x1, y1 = [v * 6 for v in T[k]]
    d.rectangle([x0, y0, x1, y1], outline=col, width=24)
    tw = d.textlength(name, font=f); cx = (x0 + x1) / 2
    ty = y0 - 184 if side == "above" else y1 + 24
    d.rectangle([cx - tw / 2 - 28, ty, cx + tw / 2 + 28, ty + 160], fill=col)
    d.text((cx - tw / 2, ty + 4), name, fill="white", font=f)
im.crop(CROP).save("acra_paper/figures/corner_view_card.png")
print("saved", im.crop(CROP).size)
