"""Polished assembly of the 4-stage infographic (2026-09-06, v2). Same real data
and structure as the v1 skeleton the user approved, now with: rounded-corner
photo cards, soft drop shadows everywhere, a cohesive palette, supersampled
rendering for crisp edges/text, smoother connectors, and a title bar -- taking
visual inspiration from DAAAM's own Figure 1 (rounded card callouts, soft
shadows, clean sans-serif type) without copying its specific graphics.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FIG_DIR = Path("paper/figures")
ROOMS_DIR = FIG_DIR / "room_survey_final"
OUT = FIG_DIR / "fig_infographic.png"

SCALE = 2  # supersample factor
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"

PALETTE = {
    "panel_bg": (234, 242, 248, 255),      # soft blue-gray panel background
    "panel_border": (190, 205, 218, 255),
    "card_border": (70, 70, 80, 255),
    "text": (25, 25, 30, 255),
    "text_light": (90, 95, 105, 255),
    "arrow": (55, 90, 130, 255),
    "accent": (33, 102, 172, 255),
    "bg": (252, 253, 254, 255),
}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(FONT_DIR + name, size * SCALE)


def soft_shadow(size, radius, blur, opacity=80, offset=(0, 10)):
    pad = blur * 3
    layer = Image.new("RGBA", (size[0] + pad * 2, size[1] + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([pad, pad, pad + size[0], pad + size[1]], radius=radius,
                         fill=(15, 20, 35, opacity))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    return layer, pad


def paste_with_shadow(canvas, img_or_none, xy, size, radius, blur=10, fill=None, offset=(0, 10)):
    """Pastes a rounded-corner image (or a flat-fill rounded rect if img is None)
    at xy with a soft shadow behind it. Returns nothing, mutates canvas."""
    x, y = xy
    shadow, pad = soft_shadow(size, radius, blur, offset=offset)
    canvas.alpha_composite(shadow, (x - pad + offset[0], y - pad + offset[1]))

    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size[0], size[1]], radius=radius, fill=255)

    if img_or_none is not None:
        content = img_or_none.convert("RGBA").resize(size, Image.LANCZOS)
    else:
        content = Image.new("RGBA", size, fill)
    canvas.paste(content, (x, y), mask)
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle([x, y, x + size[0], y + size[1]], radius=radius,
                         outline=PALETTE["card_border"], width=SCALE)


def draw_arrow(canvas, p1, p2, width=5, color=None):
    d = ImageDraw.Draw(canvas)
    color = color or PALETTE["arrow"]
    x1, y1 = p1
    x2, y2 = p2
    # shorten for arrowhead
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    head = 16 * SCALE
    xe = x2 - head * 0.85 * math.cos(ang)
    ye = y2 - head * 0.85 * math.sin(ang)
    d.line([x1, y1, xe, ye], fill=color, width=width * SCALE, joint="curve")
    left = (x2 - head * math.cos(ang - 0.5), y2 - head * math.sin(ang - 0.5))
    right = (x2 - head * math.cos(ang + 0.5), y2 - head * math.sin(ang + 0.5))
    d.polygon([p2, left, right], fill=color)


def centered_text(canvas, xy, w, text, f, color, line_gap=6):
    d = ImageDraw.Draw(canvas)
    x, y = xy
    for line in text.split("\n"):
        bbox = d.textbbox((0, 0), line, font=f)
        lw = bbox[2] - bbox[0]
        d.text((x + (w - lw) / 2, y), line, font=f, fill=color)
        y += f.size + line_gap * SCALE
    return y


# ---------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 2500 * SCALE, 1150 * SCALE
canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), PALETTE["bg"])
draw = ImageDraw.Draw(canvas)

MARGIN = 50 * SCALE
TITLE_H = 100 * SCALE

f_stage_label = font(21, bold=True)
f_small = font(16)
f_title = font(30, bold=True)

# title bar
title_text = "DejaVu-Nav: from real scenes to a verified recall"
bbox = draw.textbbox((0, 0), title_text, font=f_title)
tw = bbox[2] - bbox[0]
draw.text(((CANVAS_W - tw) / 2, 24 * SCALE), title_text, font=f_title, fill=PALETTE["text"])

content_top = TITLE_H + MARGIN
content_h = CANVAS_H - content_top - MARGIN

# ---- Stage 1: room strip, inside one soft panel ----
stage1_w = 340 * SCALE
panel1_pad = 18 * SCALE
draw.rounded_rectangle([MARGIN, content_top, MARGIN + stage1_w, content_top + content_h],
                        radius=24 * SCALE, fill=PALETTE["panel_bg"], outline=PALETTE["panel_border"], width=SCALE)
centered_text(canvas, (MARGIN, content_top + 14 * SCALE), stage1_w, "Real rooms\nscene 00770", f_stage_label, PALETTE["text"])

room_files = ["room1_entrance.png", "room2_window_radiator.png",
              "room3_hallway_wood.png", "room4_doorway.png"]
n_rooms = len(room_files)
label_h = 70 * SCALE
avail_h = content_h - label_h - panel1_pad * (n_rooms + 1)
thumb_h = avail_h // n_rooms
thumb_w = stage1_w - panel1_pad * 2
ry = content_top + label_h
for fname in room_files:
    img = Image.open(ROOMS_DIR / fname)
    paste_with_shadow(canvas, img, (MARGIN + panel1_pad, ry), (thumb_w, thumb_h),
                       radius=14 * SCALE, blur=6 * SCALE, offset=(0, 5 * SCALE))
    ry += thumb_h + panel1_pad

stage1_right = MARGIN + stage1_w
mid_y = content_top + content_h // 2

# ---- arrow 1 ----
gap = 70 * SCALE
draw_arrow(canvas, (stage1_right + 12 * SCALE, mid_y), (stage1_right + gap - 12 * SCALE, mid_y))

# ---- Stage 2: two stacked cards ----
stage2_x = stage1_right + gap
stage2_w = 460 * SCALE
box_gap = 26 * SCALE
box_h = (content_h - box_gap) // 2

# top: the one clear image, in a card with a header strip
top_y = content_top
draw.rounded_rectangle([stage2_x, top_y, stage2_x + stage2_w, top_y + box_h],
                        radius=22 * SCALE, fill=PALETTE["panel_bg"], outline=PALETTE["panel_border"], width=SCALE)
centered_text(canvas, (stage2_x, top_y + 12 * SCALE), stage2_w, "1 clear image, picked", f_stage_label, PALETTE["text"])
img_pad = 16 * SCALE
img_top = top_y + 52 * SCALE
clear_img = Image.open(ROOMS_DIR / "room2_window_radiator.png")
paste_with_shadow(canvas, clear_img, (stage2_x + img_pad, img_top),
                   (stage2_w - img_pad * 2, box_h - 52 * SCALE - img_pad),
                   radius=14 * SCALE, blur=6 * SCALE, offset=(0, 5 * SCALE))

# bottom: scatter dots on a soft gradient card, no raw photo yet
bottom_y = top_y + box_h + box_gap
draw.rounded_rectangle([stage2_x, bottom_y, stage2_x + stage2_w, bottom_y + box_h],
                        radius=22 * SCALE, fill=PALETTE["panel_bg"], outline=PALETTE["panel_border"], width=SCALE)
centered_text(canvas, (stage2_x, bottom_y + 12 * SCALE), stage2_w,
              "Real DAAAM grounding\n(detected-object scatter)", f_stage_label, PALETTE["text"])

dots_area = (stage2_w - img_pad * 2, box_h - 70 * SCALE)
dots_img = Image.open(FIG_DIR / "live_grounding" / "scatter_dots_only_matched.png").convert("RGBA")
dots_img = dots_img.resize(dots_area, Image.LANCZOS)
canvas.alpha_composite(dots_img, (stage2_x + img_pad, bottom_y + 62 * SCALE))

stage2_right = stage2_x + stage2_w

# ---- arrow 2 ----
draw_arrow(canvas, (stage2_right + 12 * SCALE, mid_y), (stage2_right + gap - 12 * SCALE, mid_y))

# ---- Stage 3: peel-apart overlay, softened and slightly rotated for realism ----
stage3_x = stage2_right + gap
stage3_w = 620 * SCALE

centered_text(canvas, (stage3_x, content_top), stage3_w, "Overlay on the real scene",
              f_stage_label, PALETTE["text"])

back = Image.open(FIG_DIR / "live_grounding" / "raw_frame_matched.png").convert("RGBA")
front = Image.open(FIG_DIR / "live_grounding" / "live_grounding_scatter_matched.png").convert("RGBA")
card_size = (int(stage3_w * 0.78), int(stage3_w * 0.78 * 0.75))

peel_h = content_h - 60 * SCALE
peel_area = Image.new("RGBA", (stage3_w, peel_h), (0, 0, 0, 0))

offset = 34 * SCALE
back_xy = ((stage3_w - card_size[0]) // 2 - offset // 2, (peel_h - card_size[1]) // 2 - offset // 2)
front_xy = (back_xy[0] + offset, back_xy[1] + offset)

back_shadow, pad = soft_shadow(card_size, 16 * SCALE, 10 * SCALE, opacity=70)
peel_area.alpha_composite(back_shadow, (back_xy[0] - pad, back_xy[1] - pad + 10 * SCALE))
back_mask = Image.new("L", card_size, 0)
ImageDraw.Draw(back_mask).rounded_rectangle([0, 0, *card_size], radius=16 * SCALE, fill=255)
back_resized = back.resize(card_size, Image.LANCZOS)
peel_area.paste(back_resized, back_xy, back_mask)
ImageDraw.Draw(peel_area).rounded_rectangle(
    [back_xy[0], back_xy[1], back_xy[0] + card_size[0], back_xy[1] + card_size[1]],
    radius=16 * SCALE, outline=PALETTE["card_border"], width=SCALE)

front_shadow, pad = soft_shadow(card_size, 16 * SCALE, 12 * SCALE, opacity=90)
peel_area.alpha_composite(front_shadow, (front_xy[0] - pad, front_xy[1] - pad + 12 * SCALE))
front_mask = Image.new("L", card_size, 0)
ImageDraw.Draw(front_mask).rounded_rectangle([0, 0, *card_size], radius=16 * SCALE, fill=255)
front_resized = front.resize(card_size, Image.LANCZOS)
peel_area.paste(front_resized, front_xy, front_mask)
ImageDraw.Draw(peel_area).rounded_rectangle(
    [front_xy[0], front_xy[1], front_xy[0] + card_size[0], front_xy[1] + card_size[1]],
    radius=16 * SCALE, outline=PALETTE["card_border"], width=SCALE)

canvas.alpha_composite(peel_area, (stage3_x, content_top + 60 * SCALE))
stage3_right = stage3_x + stage3_w

# ---- arrow 3 ----
draw_arrow(canvas, (stage3_right + 12 * SCALE, mid_y), (stage3_right + gap - 12 * SCALE, mid_y))

# ---- Stage 4: pipeline panel ----
stage4_x = stage3_right + gap
pipeline = Image.open(FIG_DIR / "pipeline_panel.png").convert("RGBA")
pipeline_h = content_h
pipeline_w = int(pipeline.width * pipeline_h / pipeline.height)
pipeline_resized = pipeline.resize((pipeline_w, pipeline_h), Image.LANCZOS)
canvas.alpha_composite(pipeline_resized, (stage4_x, content_top))

final_w = stage4_x + pipeline_w + MARGIN
canvas = canvas.crop((0, 0, final_w, CANVAS_H))
canvas = canvas.resize((final_w // SCALE, CANVAS_H // SCALE), Image.LANCZOS)
canvas.convert("RGB").save(OUT, dpi=(300, 300))
print(f"wrote {OUT}, size {canvas.size}")
