"""Extract distinct real DAAAM captions from a .rrd recording, for held-out
query selection on a new scene -- same methodology as Section 7.7's original
scene-00770 mining (rerun.dataframe.load_recording scanning
world/daaam/live_captions), now saved as a reusable script rather than
one-off analysis, and applied to scene 00337 for a genuine held-out test set
(critique.md SS9/critique_2 SS9: dev/test split, threshold tuning only on
dev).

Usage: python scripts/mine_captions_from_rrd.py <path-to-episode.rrd>
"""
import re
import sys
from collections import Counter

import rerun.dataframe as rrd

path = sys.argv[1] if len(sys.argv) > 1 else "data/runs/phase4_batch_scene00337mining/episode_0.rrd"

recording = rrd.load_recording(path)
view = recording.view(index="log_time", contents="/world/daaam/live_captions/**")
table = view.select().read_all()
df = table.to_pandas()

caption_cols = [c for c in df.columns if "live_captions" in c and c.endswith(":Text")]
print(f"columns matching live_captions :Text: {len(caption_cols)}")

all_captions = []
for col in caption_cols:
    vals = df[col].dropna()
    for v in vals:
        # rerun TextLog columns store list-like cell values
        if isinstance(v, (list, tuple)):
            all_captions.extend([str(x) for x in v if x])
        elif v:
            all_captions.append(str(v))

distinct = sorted(set(all_captions))
print(f"total caption events: {len(all_captions)}, distinct captions: {len(distinct)}")

# frequency of candidate category words appearing anywhere in a caption --
# same "real caption hits" check Section 7.7 used before trusting a query
candidates = ["chair", "table", "cabinet", "shelf", "bookshelf", "sofa", "couch",
              "bed", "desk", "lamp", "mirror", "curtain", "painting", "picture",
              "clock", "radiator", "vent", "fireplace", "bin", "trash",
              "staircase", "stairs", "bicycle", "window", "door", "plant",
              "vase", "rug", "carpet", "cushion", "pillow"]
counts = Counter()
for cap in distinct:
    low = cap.lower()
    for c in candidates:
        if c in low:
            counts[c] += 1

print("\nCandidate category hit counts (distinct captions containing the word):")
for c, n in counts.most_common():
    print(f"  {c:12} {n}")

out_path = path.rsplit(".", 1)[0] + "_distinct_captions.txt"
with open(out_path, "w") as f:
    f.write("\n".join(distinct))
print(f"\nwrote {len(distinct)} distinct captions to {out_path}")
