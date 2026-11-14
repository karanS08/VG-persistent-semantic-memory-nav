"""Relevance score s(q, d_i): semantic similarity between a query and a stored
entity description, per CLAUDE.md Section 6. Implemented via CLIP text-embedding
cosine similarity, using the SAME open_clip model/pretrained-weights DAAAM's own
grounding worker uses for its stored `embedding` (see configs/memory.yaml's
embedding section) -- so a fresh query embedding lands in the same space as the
embeddings already attached to MemoryEntity objects.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import open_clip
import torch


class ClipTextEncoder:
    def __init__(self, model_name: str, pretrained: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy()

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Batch of L2-normalised text embeddings, shape (len(texts), d). pipeline-v8 image check."""
        features = self.model.encode_text(self.tokenizer(texts).to(self.device))
        return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()

    @torch.no_grad()
    def embed_image(self, rgb: np.ndarray) -> np.ndarray:
        """L2-normalised CLIP image embedding of an HxWx3 uint8 array. pipeline-v8 image check; uses the same
        already-loaded ViT-L/14 (its visual tower was loaded but unused before), so no extra GPU memory."""
        from PIL import Image
        x = self.preprocess(Image.fromarray(np.ascontiguousarray(rgb[:, :, :3]))).unsqueeze(0).to(self.device)
        features = self.model.encode_image(x)
        return (features / features.norm(dim=-1, keepdim=True))[0].cpu().numpy()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def compute_relevance(query_embedding: List[float], entity_embedding: Optional[List[float]]) -> float:
    """s(q, d_i). Returns 0.0 if the entity has no embedding (e.g. depth-filtered
    detections never got grounded) rather than raising -- a missing embedding is a
    legitimate real-world case per Phase 2 findings, not an error.
    """
    if entity_embedding is None:
        return 0.0
    return cosine_similarity(query_embedding, entity_embedding)
