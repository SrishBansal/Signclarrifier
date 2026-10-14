import json, shutil
import numpy as np

p = "models/signs.json"
shutil.copy(p, p + ".bak_scale")
d = json.load(open(p))

def nz_std(arr):
    nz = arr[arr != 0]
    return float(nz.std()) if nz.size else 0.0

stds = {}
for k, v in d.items():
    arr = np.array(v, dtype=np.float32)
    stds[k] = nz_std(arr)

vals = np.array([s for s in stds.values() if s > 0])
median_std = float(np.median(vals))
print(f"Total signs: {len(d)} | median scale: {median_std:.4f}")

changed = []
for k, v in d.items():
    arr = np.array(v, dtype=np.float32)
    s = stds[k]
    if s <= 0:
        continue
    ratio = s / median_std
    if ratio > 1.4 or ratio < 0.7:
        factor = median_std / s
        mask = arr != 0
        arr[mask] = arr[mask] * factor
        d[k] = arr.tolist()
        changed.append((k, round(s, 3), round(factor, 3)))

json.dump(d, open(p, "w"))
print(f"Rescaled {len(changed)} signs to match the common scale.")
for c in changed[:15]:
    print(" ", c)
