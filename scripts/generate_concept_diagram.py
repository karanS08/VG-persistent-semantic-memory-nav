"""Concept diagram v2 (2026-09-06): PAST -> MEMORY -> PRESENT schematic, per the
user's hand-drawn sketch, rebuilt with real photos and simple vector-drawn
shapes instead of emoji (emoji read as unprofessional for a paper figure).

Real elements used:
  - The robot icon is a real Boston Dynamics Spot photo (Wikimedia Commons,
    CC BY-SA 4.0, credit: Jonatan Holmgren) -- NOT a claim that this project
    used a physical Spot; the paper's own Limitations section already states
    a Spot deployment is unimplemented future work. Labeled as such here.
  - The "radiator" memory card uses a real crop of the actual detected
    radiator from this session's live grounding capture.
  - The room photo is the same real matched frame used throughout.
  - C=0.631 at "1 min elapsed" is a real row from
    results/controlled_temporal_decay.json (radiator, n_i=2, relevance=0.15,
    age=60s) -- not an illustrative placeholder.
  - The verify checkmark and location pin are simple vector shapes drawn
    directly, not emoji.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FIG_DIR = Path("paper/figures")
OUT = FIG_DIR / "fig_concept_diagram.png"
SCALE = 2
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"

PALETTE = {
    "bg": (252, 253, 254, 255),
    "text": (25, 25, 30, 255),
    "text_muted": (100, 105, 115, 255),
    "arrow": (55, 90, 130, 255),
    "card_fill": (255, 249, 214, 255),
    "card_border": (170, 145, 60, 255),
    "banner_fill": (33, 102, 172, 255),
    "banner_text": (255, 255, 255, 255),
    "divider": (210, 215, 222, 255),
    "tag_fill": (222, 235, 245, 255),
    "tag_border": (120, 150, 180, 255),
    "photo_border": (70, 70, 80, 255),
    "check": (46, 160, 90, 255),
    "pin": (192, 57, 43, 255),
}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(FONT_DIR + name, size * SCALE)


def soft_shadow(size, radius, blur, opacity=70):
    pad = blur * 3
    layer = Image.new("RGBA", (size[0] + pad * 2, size[1] + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle([pad, pad, pad + size[0], pad + size[1]],
                                             radius=radius, fill=(15, 20, 35, opacity))
    return layer.filter(ImageFilter.GaussianBlur(blur)), pad


def rounded_photo(canvas, img_path, xy, size, radius=12 * SCALE, blur=6 * SCALE):
    x, y = xy
    shadow, pad = soft_shadow(size, radius, blur)
    canvas.alpha_composite(shadow, (x - pad, y - pad + 6 * SCALE))
    img = Image.open(img_path).convert("RGB").resize(size, Image.LANCZOS)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, *size], radius=radius, fill=255)
    canvas.paste(img, (x, y), mask)
    ImageDraw.Draw(canvas).rounded_rectangle([x, y, x + size[0], y + size[1]], radius=radius,
                                              outline=PALETTE["photo_border"], width=SCALE)


def centered(draw, cx, y, text, f, fill):
    bbox = draw.textbbox((0, 0), text, font=f)
    w = bbox[2] - bbox[0]
    draw.text((cx - w / 2, y), text, font=f, fill=fill)
    return bbox[3] - bbox[1]


def draw_arrow(canvas, p1, p2, width=5, color=None):
    import math
    d = ImageDraw.Draw(canvas)
    color = color or PALETTE["arrow"]
    x1, y1 = p1
    x2, y2 = p2
    ang = math.atan2(y2 - y1, x2 - x1)
    head = 13 * SCALE
    xe = x2 - head * 0.85 * math.cos(ang)
    ye = y2 - head * 0.85 * math.sin(ang)
    d.line([x1, y1, xe, ye], fill=color, width=width * SCALE)
    left = (x2 - head * math.cos(ang - 0.5), y2 - head * math.sin(ang - 0.5))
    right = (x2 - head * math.cos(ang + 0.5), y2 - head * math.sin(ang + 0.5))
    d.polygon([p2, left, right], fill=color)


def draw_check(canvas, cx, cy, r, color):
    d = ImageDraw.Draw(canvas)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=int(3 * SCALE))
    d.line([cx - r * 0.5, cy, cx - r * 0.1, cy + r * 0.4, cx + r * 0.55, cy - r * 0.4],
           fill=color, width=int(4 * SCALE), joint="curve")


def draw_pin(canvas, cx, top_y, h, color):
    d = ImageDraw.Draw(canvas)
    r = h * 0.32
    cy = top_y + r
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    d.polygon([(cx - r * 0.7, cy + r * 0.5), (cx + r * 0.7, cy + r * 0.5), (cx, top_y + h)], fill=color)
    d.ellipse([cx - r * 0.35, cy - r * 0.35, cx + r * 0.35, cy + r * 0.35], fill=(255, 255, 255, 255))


def rounded_tag(canvas, cx, y, text, f, fill, border):
    d = ImageDraw.Draw(canvas)
    bbox = d.textbbox((0, 0), text, font=f)
    w = bbox[2] - bbox[0] + 40 * SCALE
    h = f.size + 24 * SCALE
    x = cx - w / 2
    d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=fill, outline=border, width=SCALE)
    centered(d, cx, y + (h - f.size) / 2 - 4 * SCALE, text, f, PALETTE["text"])
    return h


# ---------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 2100 * SCALE, 1000 * SCALE
canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), PALETTE["bg"])
draw = ImageDraw.Draw(canvas)

f_col_header = font(24, bold=True)
f_body = font(16)
f_body_b = font(16, bold=True)
f_small = font(13)
f_banner = font(20, bold=True)
f_credit = font(11)

MARGIN = 60 * SCALE
col_w = (CANVAS_W - 2 * MARGIN) // 3
past_cx = MARGIN + col_w // 2
mem_cx = MARGIN + col_w + col_w // 2
present_cx = MARGIN + 2 * col_w + col_w // 2

header_y = 36 * SCALE
centered(draw, past_cx, header_y, "PAST", f_col_header, PALETTE["text"])
centered(draw, mem_cx, header_y, "MEMORY", f_col_header, PALETTE["text"])
centered(draw, present_cx, header_y, "PRESENT", f_col_header, PALETTE["text"])

row_y = header_y + 80 * SCALE
photo_size = (120 * SCALE, 120 * SCALE)

# ---- PAST column: Spot photo -> real room photo ----
rounded_photo(canvas, FIG_DIR / "spot_robot_ccbysa4_holmgren.jpg", (past_cx - 150 * SCALE, row_y), photo_size)
draw_arrow(canvas, (past_cx - 20 * SCALE, row_y + 60 * SCALE), (past_cx + 20 * SCALE, row_y + 60 * SCALE))
rounded_photo(canvas, FIG_DIR / "room_survey_final/room2_window_radiator.png", (past_cx + 30 * SCALE, row_y), photo_size)
centered(draw, past_cx, row_y + 140 * SCALE, "RGB-D observation", f_body, PALETTE["text_muted"])
centered(draw, past_cx, row_y + 166 * SCALE, "(real Habitat-Sim capture)", f_small, PALETTE["text_muted"])
credit_y = row_y + 190 * SCALE
centered(draw, past_cx - 150 * SCALE + 60 * SCALE, credit_y, "Spot: Holmgren,", f_credit, PALETTE["text_muted"])
centered(draw, past_cx - 150 * SCALE + 60 * SCALE, credit_y + 15 * SCALE, "CC BY-SA 4.0", f_credit, PALETTE["text_muted"])
centered(draw, past_cx - 150 * SCALE + 60 * SCALE, credit_y + 34 * SCALE, "(future deployment", f_credit, PALETTE["text_muted"])
centered(draw, past_cx - 150 * SCALE + 60 * SCALE, credit_y + 49 * SCALE, "target, not used here)", f_credit, PALETTE["text_muted"])

# ---- MEMORY column: the memory card ----
card_w, card_h = 380 * SCALE, 280 * SCALE
card_x = mem_cx - card_w // 2
card_y = row_y - 10 * SCALE
shadow, pad = soft_shadow((card_w, card_h), 20 * SCALE, 8 * SCALE)
canvas.alpha_composite(shadow, (card_x - pad, card_y - pad + 8 * SCALE))
draw.rounded_rectangle([card_x, card_y, card_x + card_w, card_y + card_h], radius=20 * SCALE,
                        fill=PALETTE["card_fill"], outline=PALETTE["card_border"], width=2 * SCALE)

rad_thumb = (90 * SCALE, 76 * SCALE)
rounded_photo(canvas, FIG_DIR / "radiator_crop.png", (card_x + 24 * SCALE, card_y + 22 * SCALE), rad_thumb, radius=8 * SCALE, blur=4 * SCALE)
draw.text((card_x + 130 * SCALE, card_y + 26 * SCALE), "radiator", font=f_body_b, fill=PALETTE["text"])
draw.text((card_x + 130 * SCALE, card_y + 56 * SCALE), "entity #49", font=f_body, fill=PALETTE["text_muted"])

info_y = card_y + 118 * SCALE
draw_pin(canvas, card_x + 40 * SCALE, info_y, 26 * SCALE, PALETTE["pin"])
draw.text((card_x + 66 * SCALE, info_y + 4 * SCALE), "real 3D location (Sec. 3)", font=f_body, fill=PALETTE["text"])

info_y += 46 * SCALE
draw.text((card_x + 24 * SCALE, info_y), "C = 0.631", font=f_body_b, fill=PALETTE["text"])
info_y += 34 * SCALE
draw.text((card_x + 24 * SCALE, info_y), "1 min elapsed", font=f_body, fill=PALETTE["text_muted"])
info_y += 40 * SCALE
draw.text((card_x + 24 * SCALE, info_y), "↓ decay (Eq. 1)", font=f_body, fill=PALETTE["text_muted"])

tm_y = card_y + card_h + 26 * SCALE
centered(draw, mem_cx, tm_y, "TIME  /  MOBILITY", f_body_b, PALETTE["text_muted"])
draw_arrow(canvas, (mem_cx, tm_y + 34 * SCALE), (mem_cx, tm_y + 72 * SCALE))
centered(draw, mem_cx, tm_y + 80 * SCALE, "confidence ↓", f_body_b, PALETTE["text"])

# ---- PRESENT column ----
centered(draw, present_cx, row_y - 4 * SCALE, "“Find radiator”", f_body_b, PALETTE["text"])
draw_arrow(canvas, (present_cx, row_y + 26 * SCALE), (present_cx, row_y + 58 * SCALE))
centered(draw, present_cx, row_y + 66 * SCALE, "memory retrieval", f_body, PALETTE["text_muted"])
draw_arrow(canvas, (present_cx, row_y + 94 * SCALE), (present_cx, row_y + 126 * SCALE))

icon_row_y = row_y + 134 * SCALE
rounded_photo(canvas, FIG_DIR / "spot_robot_ccbysa4_holmgren.jpg", (present_cx - 150 * SCALE, icon_row_y), photo_size)
draw_arrow(canvas, (present_cx - 20 * SCALE, icon_row_y + 60 * SCALE), (present_cx + 20 * SCALE, icon_row_y + 60 * SCALE))
rounded_photo(canvas, FIG_DIR / "radiator_crop.png", (present_cx + 30 * SCALE, icon_row_y), photo_size, radius=8 * SCALE, blur=4 * SCALE)

verify_y = icon_row_y + 150 * SCALE
draw_check(canvas, present_cx - 60 * SCALE, verify_y + 14 * SCALE, 16 * SCALE, PALETTE["check"])
draw.text((present_cx - 24 * SCALE, verify_y), "VERIFIED", font=f_body_b, fill=PALETTE["text"])

# ---- connecting arrows between columns ----
mid_row_y = row_y + 60 * SCALE
draw_arrow(canvas, (MARGIN + col_w - 30 * SCALE, mid_row_y), (MARGIN + col_w + 10 * SCALE, mid_row_y))
draw_arrow(canvas, (MARGIN + 2 * col_w - 30 * SCALE, mid_row_y), (MARGIN + 2 * col_w + 10 * SCALE, mid_row_y))

for cx_line in (MARGIN + col_w, MARGIN + 2 * col_w):
    draw.line([cx_line, header_y - 10 * SCALE, cx_line, verify_y + 60 * SCALE],
              fill=PALETTE["divider"], width=SCALE)

# ---- bottom banner ----
banner_y = verify_y + 90 * SCALE
banner_h = 58 * SCALE
draw.rectangle([MARGIN, banner_y, CANVAS_W - MARGIN, banner_y + banner_h], fill=PALETTE["banner_fill"])
centered(draw, CANVAS_W // 2, banner_y + (banner_h - f_banner.size) // 2 - 2 * SCALE,
         "MEMORY-INFLUENCED NAVIGATION", f_banner, PALETTE["banner_text"])

# ---- bottom property tags ----
tags_y = banner_y + banner_h + 28 * SCALE
tag_texts = ["persistent", "decaying", "verified"]
tag_gap = 40 * SCALE
tag_widths = []
for t in tag_texts:
    bbox = draw.textbbox((0, 0), t, font=f_body_b)
    tag_widths.append(bbox[2] - bbox[0] + 40 * SCALE)
total_w = sum(tag_widths) + tag_gap * (len(tag_texts) - 1)
x = (CANVAS_W - total_w) // 2
for t, w in zip(tag_texts, tag_widths):
    rounded_tag(canvas, x + w / 2, tags_y, t, f_body_b, PALETTE["tag_fill"], PALETTE["tag_border"])
    x += w + tag_gap

final_h = tags_y + 84 * SCALE
canvas = canvas.crop((0, 0, CANVAS_W, final_h))
canvas = canvas.resize((CANVAS_W // SCALE, final_h // SCALE), Image.LANCZOS)
canvas.convert("RGB").save(OUT, dpi=(300, 300))
print(f"wrote {OUT}, size {canvas.size}")
