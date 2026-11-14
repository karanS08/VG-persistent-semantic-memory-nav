"""Standalone 'PIPELINE' flowchart panel (Query -> Direct Recall -> Matched Entity
-> Outcome), polished pass (2026-09-06): rounded cards, soft drop shadows, a
cohesive palette, smoother connectors. Same real numbers throughout: the
wall-mounted-radiator query, entity_id=49, real relevance/reliability/outcome
from a real episode (data/runs/.../drsweep_baseline_theta0.30)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path("paper/figures/pipeline_panel.png")
SCALE = 3  # supersample factor for crisp rounded corners/text, downsampled at the end

W, H = 620, 1180
canvas = Image.new("RGBA", (W * SCALE, H * SCALE), (0, 0, 0, 0))
draw = ImageDraw.Draw(canvas)

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
f_title = ImageFont.truetype(FONT_DIR + "DejaVuSans-Bold.ttf", 30 * SCALE)
f_body = ImageFont.truetype(FONT_DIR + "DejaVuSans.ttf", 21 * SCALE)
f_body_b = ImageFont.truetype(FONT_DIR + "DejaVuSans-Bold.ttf", 21 * SCALE)

PALETTE = {
    "query": (198, 224, 240, 255),
    "recall": (250, 200, 170, 255),
    "match": (238, 238, 240, 255),
    "outcome": (198, 232, 200, 255),
    "border": (70, 70, 80, 255),
    "text": (25, 25, 30, 255),
    "arrow": (60, 60, 70, 255),
}


def rounded_shadow(size, radius, blur, opacity=90):
    """A soft drop shadow layer the size of `size`, for a rounded rect."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pad = blur * 2
    d.rounded_rectangle([pad, pad, size[0] - pad, size[1] - pad], radius=radius,
                         fill=(20, 20, 30, opacity))
    return layer.filter(ImageFilter.GaussianBlur(blur))


def draw_card(xy, size, text, fill, header=None, radius=22 * SCALE):
    x, y = xy
    w, h = size
    shadow = rounded_shadow((w + 24 * SCALE, h + 24 * SCALE), radius, blur=7 * SCALE, opacity=70)
    canvas.paste(shadow, (x - 12 * SCALE, y - 6 * SCALE + 8 * SCALE), shadow)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill,
                            outline=PALETTE["border"], width=max(1, SCALE))
    # text, vertically centered, word-wrapped by caller (text has its own \n)
    lines = text.split("\n")
    line_h = f_body.size + 6 * SCALE
    total_h = line_h * len(lines)
    ty = y + (h - total_h) / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=f_body)
        lw = bbox[2] - bbox[0]
        draw.text((x + (w - lw) / 2, ty), line, font=f_body, fill=PALETTE["text"])
        ty += line_h


def draw_connector(x, y1, y2):
    """Smooth vertical connector with a small circular joint and arrowhead."""
    draw.line([x, y1, x, y2 - 14 * SCALE], fill=PALETTE["arrow"], width=int(3.5 * SCALE))
    ah = 11 * SCALE
    draw.polygon([(x, y2), (x - ah, y2 - ah * 1.6), (x + ah, y2 - ah * 1.6)], fill=PALETTE["arrow"])


# title
title = "Frontier agent: memory-weighted\nverification, one real episode"
ty = 6 * SCALE
for line in title.split("\n"):
    bbox = draw.textbbox((0, 0), line, font=f_title)
    lw = bbox[2] - bbox[0]
    draw.text(((W * SCALE - lw) / 2, ty), line, font=f_title, fill=PALETTE["text"])
    ty += f_title.size + 8 * SCALE

card_w = 560 * SCALE
card_h = 190 * SCALE
card_x = (W * SCALE - card_w) // 2
gap = 46 * SCALE
start_y = 130 * SCALE

cards = [
    ("Query:\n“a wall-mounted radiator”", PALETTE["query"]),
    ("Direct recall:\ns(q, dᵢ) = 0.217\nCᵢ = 0.393\nRᵢ = 0.085", PALETTE["recall"]),
    ("Matched entity:\n“A black, horizontal, column-\nstyle radiator with a series\nof vertical fins...”", PALETTE["match"]),
    ("Outcome: verified\n198 steps, SPL = 0.096\n(one of 9 correct firings\nfor this query, Section 7.4)", PALETTE["outcome"]),
]

y = start_y
positions = []
for text, color in cards:
    draw_card((card_x, y), (card_w, card_h), text, color)
    positions.append(y)
    y += card_h + gap

for y_top, y_next in zip(positions[:-1], positions[1:]):
    draw_connector(W * SCALE // 2, y_top + card_h, y_next)

canvas = canvas.resize((W, H), Image.LANCZOS)
canvas.save(OUT)
print(f"wrote {OUT}")
