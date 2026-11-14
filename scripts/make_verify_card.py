"""Verify-card content for the teaser (2026-10-03): two REAL DAAAM captions from our own memory entries, for the query
"radiator" -- a window-region caption that mentions a radiator, and a caption whose subject is a radiator. Excerpts are
verbatim fragments (checked against the stored caption; "[...]" marks omissions), with the query word highlighted.
Sources (printed): data/runs/_superseded_instance_off_postgate/memory_snapshot_ep1.json entity 436;
                   data/runs/phase4_batch_instance_off/memory_snapshot_ep1.json entity 49.
Out: acra_paper/figures/verify_card_captions.png (transparent background; drawn at 2400 x 2200 px for a ~1.1 in wide card)"""
import json
from PIL import Image, ImageDraw, ImageFont

SRC = [("data/runs/_superseded_instance_off_postgate/memory_snapshot_ep1.json", 436,
        ["a large window with a white frame", "a white radiator beneath it."], "#e6a817", "subject: window", "look closer"),
       ("data/runs/phase4_batch_instance_off/memory_snapshot_ep1.json", 49,
        ["A black, vertical, column-style radiator with a series of closely spaced, thin, elongated panels"], "#2e8b57", "subject: radiator", "accept")]
W, H = 2400, 2200
F = lambda sz, b=True: ImageFont.truetype("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf", sz)
im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
INK = (31, 42, 68)


def words_layout(parts, font, maxw, x0):
    """parts: list of (text, colour) -> lines of (x, text, colour)"""
    lines, cur, x = [], [], x0
    for text, col in parts:
        for w in text.split(" "):
            if not w:
                continue
            ww = d.textlength(w + " ", font=F(104, w.startswith("radiator")))
            if x + ww > x0 + maxw and cur:
                lines.append(cur); cur, x = [], x0
            cur.append((x, w, col)); x += ww
    lines.append(cur)
    return lines


def parts_of(frags, accent, lead=False, trail=False):
    out = [("[...] ", (90, 90, 90))] if lead else []
    for i, fr in enumerate(frags):
        if i:
            out.append((" [...] ", (90, 90, 90)))
        bits = fr.replace("radiator", "\0radiator\0").split("\0")
        for b in bits:
            if b:
                out.append((b, accent if b == "radiator" else INK))
    if trail:
        out.append((" [...]", (90, 90, 90)))
    return out


d.text((20, 10), "Query:  ", fill=(90, 90, 90), font=F(120, False))
qw = d.textlength("Query:  ", font=F(120, False))
d.text((20 + qw, 10), "“radiator”", fill=(217, 130, 43), font=F(120))
y = 190
for path, eid, frags, colour, subject, verdict in SRC:
    cap = next(e for e in json.load(open(path))["entities"] if e["id"] == eid)["description"]
    for fr in frags:
        assert fr in cap, (eid, fr)
    print(f"entity {eid} <- {path}\n   {cap}")
    font = F(104, False)
    lines = words_layout(parts_of(frags, (217, 130, 43), lead=(eid == 436), trail=(eid == 49)), font, W - 2 * 70 - 40, 70)
    bh = 60 + len(lines) * 140 + 200
    rgb = tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))
    d.rounded_rectangle([20, y, W - 20, y + bh], radius=60, fill=(255, 255, 255, 255), outline=rgb + (255,), width=14)
    ty = y + 40
    for ln in lines:
        for x, w, col in ln:
            d.text((x, ty), w, fill=col, font=F(104, w.startswith("radiator")))
        ty += 140
    d.text((70, ty + 20), subject, fill=rgb + (255,), font=F(104))
    tag = "✓ " + verdict if verdict == "accept" else "⚠ " + verdict
    tw = d.textlength(tag, font=F(104))
    d.text((W - 70 - tw, ty + 20), tag, fill=rgb + (255,), font=F(104))
    y += bh + 70
im.crop((0, 0, W, y)).save("acra_paper/figures/verify_card_captions.png")
print("saved", im.crop((0, 0, W, y)).size)
