"""pipeline-v8 image check, with a fake encoder (no CLIP download in unit tests)."""
import math

import numpy as np

from frontier_agent.image_check import ImageChecker

INTR = {"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0}
POSE = {"position": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
DEPTH = np.full((480, 640), 10.0, np.float32)
RGB = np.zeros((480, 640, 3), np.uint8)


class FakeEncoder:
    """Each label gets a one-hot text vector; the image vector says what the crop 'looks like'."""
    def __init__(self, looks_like, synonyms=()):
        self.looks_like, self.synonyms, self.index = looks_like, set(synonyms), {}

    def _vec(self, label):
        label = label.replace("a photo of a ", "").rstrip(".")
        key = "sofa" if label in self.synonyms else label
        i = self.index.setdefault(key, len(self.index))
        v = np.zeros(256); v[i] = 1.0
        return v

    def embed_texts(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_image(self, rgb):
        return self._vec(f"a photo of a {self.looks_like}.") * 0.3 + 0.01


def test_pass_when_crop_looks_like_query():
    r = ImageChecker(FakeEncoder("sofa")).check("a sofa", [0.0, 0.5, -1.5], RGB, DEPTH, POSE, INTR, 1.5)
    assert r["status"] == "pass" and r["rank"] == 1 and r["margin"] > 0


def test_fail_when_crop_looks_like_chair():
    r = ImageChecker(FakeEncoder("chair")).check("a sofa", [0.0, 0.5, -1.5], RGB, DEPTH, POSE, INTR, 1.5)
    assert r["status"] == "fail" and r["top_label"] == "chair" and r["rank"] > 1


def test_query_word_itself_removed_from_vocabulary_and_synonyms_dropped():
    ic = ImageChecker(FakeEncoder("couch", synonyms={"couch"}), vocabulary=("couch", "sofa", "chair"))
    labels, _ = ic._labels_for("a sofa")
    assert labels == ["sofa", "chair"]  # "sofa" not duplicated; "couch" (same text vector) dropped as synonym
    assert ic.check("a sofa", [0.0, 0.5, -1.5], RGB, DEPTH, POSE, INTR, 1.5)["status"] == "pass"


def test_not_visible_when_behind_or_occluded():
    ic = ImageChecker(FakeEncoder("sofa"))
    assert ic.check("a sofa", [0.0, 0.5, 1.5], RGB, DEPTH, POSE, INTR, 1.5)["status"] == "not_visible"
    near_wall = np.full((480, 640), 0.3, np.float32)  # tol is 1.0 m for the image check
    assert ic.check("a sofa", [0.0, 0.5, -1.5], RGB, near_wall, POSE, INTR, 1.5)["status"] == "not_visible"
