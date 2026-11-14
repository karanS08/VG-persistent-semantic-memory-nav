"""Second attempt at the teaser figure (2026-09-06), rebuilt around DAAAM's
actual composition rather than a row of bordered boxes: one large "hero" scene
with real detections and floating caption callouts directly on it, fed by a
thumbnail strip on the left, with a compact query/outcome flow on the right --
not four side-by-side panels.

All real data: the hero photo + bboxes are the actual live-pipeline capture
(paper/figures/live_grounding/), the two floating captions are real DAAAM
descriptions from this project (the radiator's own verified caption, and a
real window caption from the same batch run/scene), and the query/outcome
numbers are the same real episode used throughout this session's work.
"""
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FIG_DIR = Path("paper/figures")
ROOMS_DIR = FIG_DIR / "room_survey_final"
OUT = FIG_DIR / "fig_hero_teaser.png"

SCALE = 2
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"

PALETTE = {
    "bg": (252, 253, 254, 255),
    "text": (25, 25, 30, 255),
    "arrow": (55, 90, 130, 255),
    "callout_fill": (255, 249, 214, 255),
    "callout_border": (160, 140, 60, 255),
    "highlight": (46, 160, 90, 255),
    "muted_box": (140, 150, 165, 200),
    "query_fill": (198, 224, 240, 255),
    "outcome_fill": (198, 232, 200, 255),
    "card_border": (70, 70, 80, 255),
}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(FONT_DIR + name, size * SCALE)


def soft_shadow(size, radius, blur, opacity=80):
    pad = blur * 3
    layer = Image.new("RGBA", (size[0] + pad * 2, size[1] + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle([pad, pad, pad + size[0], pad + size[1]],
                                             radius=radius, fill=(15, 20, 35, opacity))
    return layer.filter(ImageFilter.GaussianBlur(blur)), pad


def rounded_paste(canvas, img, xy, size, radius, shadow_blur=8, shadow_opacity=80, offset=(0, 8)):
    x, y = xy
    shadow, pad = soft_shadow(size, radius, shadow_blur, shadow_opacity)
    canvas.alpha_composite(shadow, (x - pad + offset[0], y - pad + offset[1]))
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, *size], radius=radius, fill=255)
    resized = img.convert("RGBA").resize(size, Image.LANCZOS)
    canvas.paste(resized, (x, y), mask)
    ImageDraw.Draw(canvas).rounded_rectangle([x, y, x + size[0], y + size[1]], radius=radius,
                                              outline=PALETTE["card_border"], width=SCALE)


def wrapped_text(draw, xy, text, f, fill, max_width, line_gap=5):
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textbbox((0, 0), trial, font=f)[2] > max_width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=f, fill=fill)
        y += f.size + line_gap * SCALE
    return y


def rounded_bubble(canvas, xy, size, text, fill, border, f, radius=16, pad=14, tail_to=None):
    x, y = xy
    w, h = size
    shadow, spad = soft_shadow((w, h), radius * SCALE, 6 * SCALE, 60)
    canvas.alpha_composite(shadow, (x - spad, y - spad + 6 * SCALE))
    d = ImageDraw.Draw(canvas)
    if tail_to is not None:
        tx, ty = tail_to
        # small triangle tail from the nearest edge toward tail_to
        cx, cy = x + w / 2, y + h / 2
        if abs(tx - cx) > abs(ty - cy):
            ex = x + w if tx > cx else x
            ey = cy
            tri = [(ex, ey - 10 * SCALE), (ex, ey + 10 * SCALE), (tx, ty)]
        else:
            ey = y + h if ty > cy else y
            ex = cx
            tri = [(ex - 10 * SCALE, ey), (ex + 10 * SCALE, ey), (tx, ty)]
        d.polygon(tri, fill=fill)
    d.rounded_rectangle([x, y, x + w, y + h], radius=radius * SCALE, fill=fill,
                         outline=border, width=SCALE)
    wrapped_text(d, (x + pad * SCALE, y + pad * SCALE), text, f, PALETTE["text"], w - 2 * pad * SCALE)


def draw_arrow(canvas, p1, p2, width=5, color=None, label=None, f=None):
    d = ImageDraw.Draw(canvas)
    color = color or PALETTE["arrow"]
    x1, y1 = p1
    x2, y2 = p2
    ang = math.atan2(y2 - y1, x2 - x1)
    head = 14 * SCALE
    xe = x2 - head * 0.85 * math.cos(ang)
    ye = y2 - head * 0.85 * math.sin(ang)
    d.line([x1, y1, xe, ye], fill=color, width=width * SCALE)
    left = (x2 - head * math.cos(ang - 0.5), y2 - head * math.sin(ang - 0.5))
    right = (x2 - head * math.cos(ang + 0.5), y2 - head * math.sin(ang + 0.5))
    d.polygon([p2, left, right], fill=color)
    if label and f:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 34 * SCALE
        bbox = d.textbbox((0, 0), label, font=f)
        lw = bbox[2] - bbox[0]
        d.text((mx - lw / 2, my), label, font=f, fill=PALETTE["text"])


# ---------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 2400 * SCALE, 1050 * SCALE
canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), PALETTE["bg"])
draw = ImageDraw.Draw(canvas)

f_title = font(28, bold=True)
f_label = font(17)
f_bubble = font(16)
f_bubble_b = font(17, bold=True)

MARGIN = 46 * SCALE
title = "DejaVu-Nav: real scenes, real grounding, one verified recall"
bbox = draw.textbbox((0, 0), title, font=f_title)
draw.text(((CANVAS_W - (bbox[2] - bbox[0])) / 2, 20 * SCALE), title, font=f_title, fill=PALETTE["text"])

content_top = 90 * SCALE
content_h = CANVAS_H - content_top - MARGIN

# ---- thumbnail strip (no panel box, just floating photos with shadow) ----
strip_w = 190 * SCALE
room_files = ["room1_entrance.png", "room2_window_radiator.png",
              "room3_hallway_wood.png", "room4_doorway.png"]
n = len(room_files)
gap_v = 16 * SCALE
thumb_h = (content_h - gap_v * (n - 1)) // n
y = content_top
for fname in room_files:
    img = Image.open(ROOMS_DIR / fname)
    rounded_paste(canvas, img, (MARGIN, y), (strip_w, thumb_h), radius=12 * SCALE,
                  shadow_blur=6 * SCALE, shadow_opacity=70)
    y += thumb_h + gap_v

strip_right = MARGIN + strip_w
hero_x = strip_right + 170 * SCALE
mid_y = content_top + content_h // 2

draw_arrow(canvas, (strip_right + 14 * SCALE, mid_y), (hero_x - 14 * SCALE, mid_y))
d2 = ImageDraw.Draw(canvas)
f_arrow_label = font(15)
label_lines = ["DAAAM", "grounding"]
ly = mid_y - 62 * SCALE
for line in label_lines:
    b = d2.textbbox((0, 0), line, font=f_arrow_label)
    lw = b[2] - b[0]
    cx = (strip_right + hero_x) / 2
    d2.text((cx - lw / 2, ly), line, font=f_arrow_label, fill=PALETTE["text"])
    ly += f_arrow_label.size + 4 * SCALE

# ---- hero image: real scene + real bboxes, styled ----
hero_w = 1150 * SCALE
hero_h = content_h
tracks = json.load(open(FIG_DIR / "live_grounding" / "tracks_matched.json"))
base = Image.open(FIG_DIR / "live_grounding" / "raw_frame_matched.png").convert("RGB")
base = base.resize((hero_w - 20 * SCALE, hero_h - 20 * SCALE), Image.LANCZOS)
sx = base.width / 640
sy = base.height / 480

hero_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
hd = ImageDraw.Draw(hero_layer)
highlight_ids = {3, 4}  # radiator, window -- the two we caption
radiator_center = window_center = None
for t in tracks:
    if t["id"] not in highlight_ids:
        continue  # only show the 2 detections we caption -- the other 5 are
        # large background/wall/floor regions that just clutter the hero image
    x1, y1, x2, y2 = [v * (sx if i % 2 == 0 else sy) for i, v in enumerate(t["bbox"])]
    hd.rectangle([x1, y1, x2, y2], outline=PALETTE["highlight"], width=4 * SCALE)
    if t["id"] == 3:
        radiator_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    if t["id"] == 4:
        window_center = ((x1 + x2) / 2, (y1 + y2) / 2)

rounded_paste(canvas, base, (hero_x, content_top + 10 * SCALE),
              (base.width, base.height), radius=18 * SCALE, shadow_blur=10 * SCALE, shadow_opacity=90)
canvas.alpha_composite(hero_layer, (hero_x + 10 * SCALE, content_top + 20 * SCALE))

hero_origin = (hero_x + 10 * SCALE, content_top + 20 * SCALE)


def to_canvas(pt):
    return (hero_origin[0] + pt[0], hero_origin[1] + pt[1])


radiator_caption = ("“A black, horizontal, column-style radiator with a series "
                     "of vertical fins and a top grille...”")
window_caption = ("“A rectangular window with a white frame, divided into two "
                   "sections by a horizontal bar...”")

bubble_w = 300 * SCALE
bubble_h = 130 * SCALE
rc = to_canvas(radiator_center)
wc = to_canvas(window_center)

rounded_bubble(canvas, (hero_x - 40 * SCALE, content_top + hero_h - bubble_h + 10 * SCALE),
               (bubble_w, bubble_h), radiator_caption, PALETTE["callout_fill"],
               PALETTE["callout_border"], f_bubble, tail_to=rc)
rounded_bubble(canvas, (hero_x + hero_w - bubble_w - 30 * SCALE, content_top - 6 * SCALE),
               (bubble_w, bubble_h), window_caption, PALETTE["callout_fill"],
               PALETTE["callout_border"], f_bubble, tail_to=wc)

hero_right = hero_x + hero_w

# ---- compact query/outcome flow on the right ----
flow_x = hero_right + 90 * SCALE
flow_w = CANVAS_W - flow_x - MARGIN
bub_h = 120 * SCALE
bub_gap = 40 * SCALE
flow_top = content_top + 20 * SCALE

items = [
    ("Query", "“a wall-mounted radiator”", PALETTE["query_fill"]),
    ("Direct recall", "s(q,dᵢ)=0.217  Cᵢ=0.393  Rᵢ=0.085", (250, 200, 170, 255)),
    ("Outcome: verified", "198 steps, SPL=0.096 — 1 of 9 correct\nfirings for this query (Sec. 7.4)", PALETTE["outcome_fill"]),
]
y = flow_top
draw_arrow(canvas, (hero_right + 14 * SCALE, mid_y), (flow_x - 14 * SCALE, mid_y))
for i, (label, text, color) in enumerate(items):
    rounded_bubble(canvas, (flow_x, y), (flow_w, bub_h), text, color, PALETTE["card_border"], f_bubble)
    d3 = ImageDraw.Draw(canvas)
    d3.text((flow_x + 14 * SCALE, y - 26 * SCALE), label, font=f_bubble_b, fill=PALETTE["text"])
    if i < len(items) - 1:
        draw_arrow(canvas, (flow_x + flow_w // 2, y + bub_h), (flow_x + flow_w // 2, y + bub_h + bub_gap - 10 * SCALE))
    y += bub_h + bub_gap

canvas = canvas.resize((CANVAS_W // SCALE, CANVAS_H // SCALE), Image.LANCZOS)
canvas.convert("RGB").save(OUT, dpi=(300, 300))
print(f"wrote {OUT}, size {canvas.size}")
