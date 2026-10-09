import json, glob, os, sys, collections
import numpy as np

def resample_sequence(seq, target_len=48):
    t_orig = len(seq)
    if t_orig == target_len:
        return seq.copy().astype(np.float32)
    if t_orig == 0:
        return np.zeros((target_len, 225), dtype=np.float32)
    idx = np.linspace(0, t_orig - 1, target_len)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, t_orig - 1)
    w = (idx - lo)[:, None]
    return ((1.0 - w) * seq[lo] + w * seq[hi]).astype(np.float32)

def sequence_distance(s1, s2):
    return float(np.mean(np.linalg.norm(s1 - s2, axis=1)))

KP = os.path.expanduser("~/datasets/include_keypoints")
SPLITS = ["train", "val", "test"]
EXISTING = "models/signs.json"
OUT = "models/signs_include_candidate.json"
ASPECT_Y = 16 / 9  # renderer unstretches y by 9/16

def take_to_frames(d):
    """Build (T, 225) with the same layout as the renderer:
    left hand 0-62, right hand 63-125, pose 126-224 (x,y,z interleaved)."""
    frames = []
    for t in range(d["n_frames"]):
        f = np.zeros(225, dtype=np.float32)
        px, py = d["pose_x"][t], d["pose_y"][t]
        pts = []  # (index_into_225, x, y)
        if len(px) == 33:
            for i in range(33):
                pts.append((126 + i * 3, px[i], py[i]))
        for name, off in (("hand1", 0), ("hand2", 63)):
            hx, hy = d[f"{name}_x"][t], d[f"{name}_y"][t]
            if len(hx) == 21:
                for i in range(21):
                    pts.append((off + i * 3, hx[i], hy[i]))
        # Per-frame normalization: centre on shoulders, scale by shoulder width
        if len(px) == 33:
            mx = (px[11] + px[12]) / 2; my = (py[11] + py[12]) / 2
            w = max(np.hypot(px[11] - px[12], py[11] - py[12]), 1e-4)
        else:
            mx, my, w = 0.5, 0.5, 1.0
        for idx, x, y in pts:
            if np.isnan(x) or np.isnan(y):
                continue  # leave as 0 = "not detected"
            f[idx] = (x - mx) / w
            f[idx + 1] = (y - my) / w * ASPECT_Y
        frames.append(f)
    return np.array(frames, dtype=np.float32)

# 1. Group takes by word label stored inside each file
takes = collections.defaultdict(list)
for split in SPLITS:
    for path in glob.glob(f"{KP}/include_{split}_keypoints/*.json"):
        d = json.load(open(path))
        if d.get("n_frames", 0) > 0:
            takes[d["label"]].append(take_to_frames(d))

# 2. Medoid take per word, resampled to 48 frames
candidates = {}
for word, seqs in takes.items():
    res = [resample_sequence(s, 48) for s in seqs]
    if len(res) == 1:
        best = res[0]
    else:
        scores = [sum(sequence_distance(a, b) for b in res) for a in res]
        best = res[int(np.argmin(scores))]
    candidates[word] = best.tolist()

# 3. Merge: keep every existing entry untouched
existing = json.load(open(EXISTING))
overlap = sorted(set(candidates) & set(existing))
new_words = sorted(set(candidates) - set(existing))
merged = dict(existing)
for w in new_words:
    merged[w] = candidates[w]
json.dump(merged, open(OUT, "w"))

print(f"INCLUDE words with usable takes: {len(candidates)}")
print(f"Already in signs.json (kept as-is): {len(overlap)} -> {overlap[:15]}")
print(f"New words added: {len(new_words)}")
print(f"Wrote {OUT} with {len(merged)} entries")
