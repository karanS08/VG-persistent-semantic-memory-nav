"""Icon-only Verify card (2026-10-03): dashed amber ring + map pin + clock (remembered, unconfirmed) --eye--> solid green ring with
the radiator photo (acra_paper/figures/live_grounding/raw_frame_matched_6x.png, track 3 = the same radiator as the Remember card) and a check badge (confirmed by a fresh view).
Out: acra_paper/figures/verify_card_icons.png (transparent)"""
import json, math
from PIL import Image, ImageDraw
D = "acra_paper/figures/live_grounding"
W, H = 2400, 2000
AMBER, GREEN, GREY, INK = (230, 168, 23, 255), (46, 139, 87, 255), (130, 130, 130, 255), (31, 42, 68, 255)
im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)


def dashed_circle(cx, cy, r, col, w=26, n=22):
    for i in range(n):
        a0 = 360 / n * i; d.arc([cx - r, cy - r, cx + r, cy + r], a0, a0 + 360 / n * 0.58, fill=col, width=w)


# radiator photo crop (DAAAM box 13)
T = {str(t["id"]): t["bbox"] for t in json.load(open(f"{D}/tracks_matched.json"))}; ph = Image.open(f"{D}/raw_frame_matched_6x.png").convert("RGB")
x0, y0, x1, y1 = [v * 6 for v in T["3"]]; cx_, cy_ = (x0 + x1) / 2, (y0 + y1) / 2; s = max(x1 - x0, y1 - y0) * 0.75
crop = ph.crop((int(cx_ - s), int(cy_ - s), int(cx_ + s), int(cy_ + s))).resize((760, 760))
mask = Image.new("L", (760, 760), 0); ImageDraw.Draw(mask).ellipse([0, 0, 759, 759], fill=255)
R = 380
# right: confirmed
rx, ry = 1760, 820
im.paste(crop, (rx - R, ry - R), mask); d.ellipse([rx - R - 14, ry - R - 14, rx + R + 14, ry + R + 14], outline=GREEN, width=36)
bx, by = rx + 290, ry + 290
d.ellipse([bx - 130, by - 130, bx + 130, by + 130], fill=GREEN, outline=(255, 255, 255, 255), width=14)
d.line([(bx - 62, by + 4), (bx - 18, by + 52), (bx + 70, by - 52)], fill="white", width=34, joint="curve")
# left: remembered (hypothesis)
lx, ly = 640, 820
d.ellipse([lx - R, ly - R, lx + R, ly + R], fill=(255, 248, 225, 255)); dashed_circle(lx, ly, R, AMBER)
d.ellipse([lx - 110, ly - 250, lx + 110, ly - 30], fill=AMBER)                      # pin head
d.polygon([(lx - 100, ly - 90), (lx + 100, ly - 90), (lx, ly + 190)], fill=AMBER)    # pin tail
d.ellipse([lx - 42, ly - 182, lx + 42, ly - 98], fill=(255, 248, 225, 255))
cx, cy = lx + 255, ly + 255                                                       # clock badge
d.ellipse([cx - 120, cy - 120, cx + 120, cy + 120], fill=(255, 255, 255, 255), outline=GREY, width=14)
d.line([(cx, cy), (cx, cy - 74)], fill=GREY, width=16); d.line([(cx, cy), (cx + 56, cy + 22)], fill=GREY, width=16)
# middle: eye on an arrow
my = 820
d.line([(1110, my), (1290, my)], fill=INK, width=24); d.polygon([(1290, my - 56), (1290, my + 56), (1370, my)], fill=INK)
ex, ey = 1240, 600
d.ellipse([ex - 150, ey - 80, ex + 150, ey + 80], outline=INK, width=22); d.ellipse([ex - 44, ey - 44, ex + 44, ey + 44], fill=INK)
im.save("acra_paper/figures/verify_card_icons.png"); print("saved", im.size)
