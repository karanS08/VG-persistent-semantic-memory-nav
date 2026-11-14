"""Image check at verification (pipeline-v8, --image-check).

The submitted pipeline verifies on caption text only (CLIP text embedding of DAAAM's caption + lexical gate), so
when DAAAM's caption itself is wrong -- v7 THOR: dining chairs and an armchair captioned "a blue upholstered sofa",
5 of 11 false successes -- no text check can catch it. This check looks at the pixels instead: the verified entry is
projected into the CURRENT frame, a crop around it is embedded with CLIP's image tower (the same ViT-L/14 already
loaded for text), and the query must rank FIRST among a fixed, target-independent household vocabulary. There is no
threshold to tune: top-1 or not. Vocabulary entries that are synonyms of the query (same head noun, or text-embedding
cosine >= SYNONYM_COS with the query prompt) are dropped so e.g. "couch" cannot beat "sofa".

If the entry does not project into view (behind the camera, outside the image, or occluded per the depth image), the
check reports `not_visible` and verification simply waits for a better view -- nothing is written to memory.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

from memory_layer.geometry import crop_object, expected_visible

# Fixed a priori (common indoor categories, MP3D/HM3D style); NOT tuned per target or per scene.
VOCABULARY = (
    "chair", "armchair", "sofa", "table", "desk", "bed", "cabinet", "shelf", "bookshelf", "dresser",
    "nightstand", "wardrobe", "stool", "bench", "ottoman", "cushion", "pillow", "rug", "curtain", "window",
    "door", "mirror", "painting", "lamp", "television", "plant", "toilet", "sink", "bathtub", "shower",
    "refrigerator", "oven", "microwave", "stove", "countertop", "washing machine", "radiator", "fireplace",
    "towel", "box", "trash can", "clock", "vase", "laptop", "wall", "floor",
)
SYNONYM_COS = 0.90
_ARTICLES = {"a", "an", "the"}


def _prompt(label: str) -> str:
    return f"a photo of a {label}."


def _head(text: str) -> str:
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _ARTICLES]
    return words[-1] if words else text.lower()


class ImageChecker:
    def __init__(self, encoder, vocabulary=VOCABULARY, top_k: int = 1):
        self.encoder = encoder
        self.vocabulary = tuple(vocabulary)
        self.top_k = top_k
        self._cache: Dict[str, tuple] = {}

    def _labels_for(self, query: str):
        if query not in self._cache:
            target = re.sub(r"^(a|an|the)\s+", "", query.strip().lower())
            head = _head(target)
            vocab = [w for w in self.vocabulary if _head(w) != head and not head.startswith(_head(w))
                     and not _head(w).startswith(head)]
            emb = self.encoder.embed_texts([_prompt(target)] + [_prompt(w) for w in vocab])
            keep = [i for i in range(1, len(emb)) if float(emb[0] @ emb[i]) < SYNONYM_COS]
            labels = [target] + [vocab[i - 1] for i in keep]
            self._cache[query] = (labels, emb[[0] + keep])
        return self._cache[query]

    def crop(self, entity_position: List[float], rgb: np.ndarray, depth, pose: Dict, intrinsics: Dict,
             cam_height: float, max_range: float = 3.0) -> Optional[np.ndarray]:
        """Crop of the entry's region in the current frame, or None if it is not in view. occlusion_tol 1.0 m: the
        stored point can lie up to ~half the object's depth behind its front surface; the test only has to reject
        a wall or door standing in front of it."""
        if not expected_visible(entity_position, pose, depth, intrinsics, cam_height, min_range=0.0,
                                max_range=max_range, margin_px=0, occlusion_tol=1.0, clamp_vertical=True):
            return None
        return crop_object(rgb, entity_position, pose, intrinsics, cam_height)

    def scores(self, query: str, crop: np.ndarray):
        """(labels, similarities); labels[0] is the query target."""
        labels, text_emb = self._labels_for(query)
        return labels, text_emb @ self.encoder.embed_image(crop)

    def check(self, query: str, entity_position: List[float], rgb: np.ndarray, depth, pose: Dict,
              intrinsics: Dict, cam_height: float, max_range: float = 3.0) -> Dict:
        crop = self.crop(entity_position, rgb, depth, pose, intrinsics, cam_height, max_range)
        if crop is None:
            return {"status": "not_visible"}
        labels, sims = self.scores(query, crop)
        order = np.argsort(-sims)
        rank = int(np.where(order == 0)[0][0]) + 1
        top = labels[int(order[0])]
        runner_up = labels[int(order[1])] if order[0] == 0 else top
        return {"status": "pass" if rank <= self.top_k else "fail", "rank": rank, "top_label": top,
                "margin": round(float(sims[0] - np.max(np.delete(sims, 0))), 4), "runner_up": runner_up,
                "crop_px": list(crop.shape[:2])}
