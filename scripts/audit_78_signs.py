import json
import numpy as np

signs_78 = (
    "afternoon alive alright bad beautiful blind clean clothing colour cool curved dead deaf deep dirty dry "
    "evening famous fast female flat friday goodafternoon goodevening goodmorning goodnight happy hard healthy "
    "heavy high hour howareyou long loose loud low male minute monday month morning narrow nice night old "
    "pleased pocket poor quiet rich sad saturday shallow short sick slow soft strong sunday tall thick thin "
    "thursday tight today tomorrow tuesday ugly warm weak wednesday week wet wide year yesterday young"
).split()

with open("models/signs.json") as f:
    signs = json.load(f)

LH = slice(0, 63)
RH = slice(63, 126)
POSE = slice(126, 225)

report = []
for s_name in sorted(signs_78):
    if s_name not in signs:
        report.append({"sign": s_name, "error": "not found"})
        continue
    arr = np.array(signs[s_name], dtype=np.float32)
    T = len(arr)
    lh_nz = int(np.sum(np.any(arr[:, LH] != 0, axis=1)))
    rh_nz = int(np.sum(np.any(arr[:, RH] != 0, axis=1)))

    all_zero_frames = int(np.sum(np.all(arr == 0, axis=1)))
    zero_frac = all_zero_frames / T if T > 0 else 1.0

    arr_3d = arr.reshape(T, 75, 3)
    diff = np.diff(arr_3d, axis=0)
    dist = np.linalg.norm(diff, axis=-1)
    max_jump = float(np.max(dist)) if len(dist) > 0 else 0.0

    suspect_reasons = []
    if zero_frac > 0.05:
        suspect_reasons.append(f"{round(zero_frac*100)}% zero frames")
    if max_jump > 2.0:
        suspect_reasons.append(f"high jump {max_jump:.2f}")
    if (lh_nz > 15 and rh_nz <= 5) or (rh_nz > 15 and lh_nz <= 5):
        if 0 < min(lh_nz, rh_nz) <= 5:
            suspect_reasons.append(f"hand flicker (LH:{lh_nz}, RH:{rh_nz})")
    if lh_nz == 0 and rh_nz == 0:
        suspect_reasons.append("no hands detected")
    elif max(lh_nz, rh_nz) < 15:
        suspect_reasons.append(f"low hand frames (max {max(lh_nz, rh_nz)}/{T})")

    score = zero_frac * 100.0 + max_jump * 5.0
    if max(lh_nz, rh_nz) < 15:
        score += 50
    if 0 < min(lh_nz, rh_nz) <= 5 and max(lh_nz, rh_nz) > 15:
        score += 30

    report.append({
        "sign": s_name,
        "frames": T,
        "lh_nz": lh_nz,
        "rh_nz": rh_nz,
        "zero_frac": round(zero_frac, 3),
        "max_jump": round(max_jump, 2),
        "reasons": ", ".join(suspect_reasons) if suspect_reasons else "clean",
        "score": score
    })

report.sort(key=lambda x: x["score"], reverse=True)

print(f"{len(report)} signs analyzed.")
print("=== ALL 78 SIGNS QUALITY TABLE ===")
header = f"{'Sign':16} | {'Frames':6} | {'LH nz':5} | {'RH nz':5} | {'Zero %':6} | {'Max Jump':8} | {'Suspect / Note'}"
print(header)
print("-" * len(header))
for r in report:
    s = r["sign"]
    f_cnt = r["frames"]
    lh = r["lh_nz"]
    rh = r["rh_nz"]
    zf = f"{r['zero_frac']*100:.1f}%"
    mj = f"{r['max_jump']:.2f}"
    rs = r["reasons"]
    print(f"{s:16} | {f_cnt:<6} | {lh:<5} | {rh:<5} | {zf:<6} | {mj:<8} | {rs}")

print("\n=== TOP 15 SUSPECT SIGNS (review first) ===")
for r in report[:15]:
    s = r["sign"]
    sc = f"{r['score']:.1f}"
    lh = r["lh_nz"]
    rh = r["rh_nz"]
    zf = f"{r['zero_frac']*100:.1f}%"
    mj = f"{r['max_jump']:.2f}"
    rs = r["reasons"]
    print(f"{s:16} | score={sc:5} | LH={lh:2} RH={rh:2} | zero={zf} | jump={mj} | {rs}")
