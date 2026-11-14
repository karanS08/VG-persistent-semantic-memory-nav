"""Match DAAAM's tracked boxes to DAAAM's captions for the corner view by CONTENT (2026-10-03).
The tracker's ids and DAAAM's object ids are different numberings, so box id == caption id is wrong. Each box crop (from the
3x render) is embedded with the same CLIP model the system uses (ViT-L-14 / openai); each caption is embedded as text; the
assignment maximises total image-text similarity (Hungarian), one caption per box. Boxes whose best similarity is low are
dropped. Out: acra_paper/figures/corner_grounding/matches.json"""
import json, sys
import numpy as np, open_clip, torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

D = "acra_paper/figures/corner_grounding"
im = Image.open(f"{D}/raw_3x.png").convert("RGB")
T = json.load(open(f"{D}/tracks.json")); C = json.load(open(f"{D}/captions.json"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai"); model = model.to(dev).eval()
tok = open_clip.get_tokenizer("ViT-L-14")
bk = list(T); ck = list(C)
crops = []
for k in bk:
    x0, y0, x1, y1 = [int(v * 3) for v in T[k]]
    px = max(8, int(0.08 * max(x1 - x0, y1 - y0)))
    crops.append(pre(im.crop((max(0, x0 - px), max(0, y0 - px), min(im.width, x1 + px), min(im.height, y1 + px)))))
with torch.no_grad():
    fi = model.encode_image(torch.stack(crops).to(dev)); fi /= fi.norm(dim=-1, keepdim=True)
    ft = model.encode_text(tok([C[k][:200] for k in ck]).to(dev)); ft /= ft.norm(dim=-1, keepdim=True)
S = (fi @ ft.T).cpu().numpy()
r, c = linear_sum_assignment(-S)
out = {}
for i, j in zip(r, c):
    out[bk[i]] = {"caption_id": ck[j], "caption": C[ck[j]], "sim": round(float(S[i, j]), 3), "best_other": round(float(np.sort(S[i])[-2]), 3)}
json.dump(out, open(f"{D}/matches.json", "w"), indent=1)
for k, v in sorted(out.items(), key=lambda kv: -kv[1]["sim"]):
    print(f"box {k:>3} <- caption {v['caption_id']:>3} sim {v['sim']:.3f} (2nd {v['best_other']:.3f}) {v['caption'][:70]}")
