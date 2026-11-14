"""Test 3.4 / Task A.1: is HM3D semantic ground truth available for the paper's
scenes (00770-NBg5UqG3di3 tuning, 00337-CFVBbU9Rsyb held-out)?

Read-only. Checks, from files already on disk (no network, no credentials):
  1. which *.semantic.glb / *.semantic.txt files exist under data/hm3d/;
  2. the contents of the public, ungated example-tier semantic tarballs in
     .deps/habitat-matterport-3dresearch/example/;
  3. the official HM3D-Semantics scene lists that ship (publicly) inside
     hm3d-example-semantic-configs-v0.{1,2}.tar -> hm3d_annotated_basis.scene_dataset_config.json
     (these list EVERY annotated scene across train/val/minival/test, not just the example tier);
  4. HM3DSem-v0.2 per-scene statistics (statistics/HM3DSem-v0.2.zip);
  5. metadata.csv split membership.

Writes data/results_postdraft/gt_audit_precision/gt_availability.json.
Usage: python scripts/postdraft/gt_semantic_availability.py
"""
import csv
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / ".deps/habitat-matterport-3dresearch"
OUT = ROOT / "data/results_postdraft/gt_audit_precision/gt_availability.json"
SCENES = ["00770-NBg5UqG3di3", "00337-CFVBbU9Rsyb", "00861-GLAQ4DNUx5U"]
ID_RE = re.compile(r"(\d{5}-[A-Za-z0-9]{11})")


def main():
    res = {"scenes": {}, "tarballs": {}, "annotated_lists": {}, "notes": []}

    for s in SCENES:
        d = ROOT / "data/hm3d/example" / s
        files = sorted(p.name for p in d.iterdir()) if d.exists() else []
        res["scenes"][s] = {
            "on_disk_files": files,
            "has_semantic_glb": any(f.endswith(".semantic.glb") for f in files),
            "has_semantic_txt": any(f.endswith(".semantic.txt") for f in files),
        }

    ex = REPO / "example"
    for name in ["hm3d-example-semantic-annots-v0.2.tar", "hm3d-example-semantic-annots-v0.1.tar.gz"]:
        p = ex / name
        if p.exists():
            with tarfile.open(p) as t:
                res["tarballs"][name] = sorted(t.getnames())

    for name in ["hm3d-example-semantic-configs-v0.2.tar", "hm3d-example-semantic-configs-v0.1.tar"]:
        p = ex / name
        if not p.exists():
            continue
        with tarfile.open(p) as t:
            for m in t.getmembers():
                if m.name.endswith("hm3d_annotated_basis.scene_dataset_config.json"):
                    txt = t.extractfile(m).read().decode()
                    ids = sorted(set(ID_RE.findall(txt)))
                    res["annotated_lists"][f"{name}:{m.name}"] = {
                        "n_annotated_scenes": len(ids),
                        **{f"contains_{s}": (s in ids) for s in SCENES},
                    }

    z = REPO / "statistics/HM3DSem-v0.2.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            txt = zf.read("Per_Scene_Neighborhood_Stats.csv").decode()
            rows = list(csv.DictReader(io.StringIO(txt)))
            names = {r["Scene Name"] for r in rows}
            res["hm3dsem_v0.2_stats"] = {
                "n_scenes": len(names),
                **{f"contains_{s}": (s in names) for s in SCENES},
            }

    md = REPO / "metadata.csv"
    if md.exists():
        with open(md) as f:
            res["metadata_split"] = {r["scene"]: r["split"] for r in csv.DictReader(f) if r["scene"] in SCENES}

    res["notes"].append(
        "hm3d_annotated_basis.scene_dataset_config.json is Matterport's own list of every scene with "
        "HM3D-Semantics annotations (train+val+minival+test). If a scene is absent from it, no semantic "
        "annotation exists for that scene in that release, so credentials would not help.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
