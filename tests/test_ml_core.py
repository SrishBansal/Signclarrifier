import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clarifysign_ml.features import (landmarks_to_features, frame_activity, trim_active, to_fixed,
                                     prepare_sequence, FEATURE_DIM, SEQ_LEN, POSE)
from clarifysign_ml.dataset import session_groups, group_split
from clarifysign_ml.stream import StreamingSession


class LM:
    def __init__(self, x, y, z=0.0): self.x, self.y, self.z = x, y, z


def pose_lms(wrist_y):
    p = [LM(0.5, 0.9) for _ in range(33)]
    p[11], p[12] = LM(0.4, 0.4), LM(0.6, 0.4)          # shoulders, width 0.2
    p[15] = p[16] = LM(0.5, wrist_y)
    return p


def feat(raised):
    wy = 0.4 + (0.1 if raised else 0.5)                # raised: 0.5 widths below shoulders; rest: 2.5 widths
    hand = [LM(0.5 + 0.01 * i, 0.5) for i in range(21)]
    return landmarks_to_features(hand, None, pose_lms(wy))


def test_feature_shape_and_zeros():
    f = feat(True)
    assert f.shape == (FEATURE_DIM,) and f.dtype == np.float32
    assert not f[63:126].any()                          # missing right hand -> zeros, not invented
    assert f[:63].any() and f[POSE].any()
    assert np.allclose(landmarks_to_features(None, None, None), 0)


def test_pose_is_shoulder_normalised():
    p = feat(True)[POSE].reshape(33, 3)
    assert np.allclose((p[11] + p[12]) / 2, 0, atol=1e-5)
    assert abs(np.linalg.norm(p[11, :2] - p[12, :2]) - 1.0) < 1e-4


def test_activity_trim_resample():
    assert frame_activity(feat(True)) and not frame_activity(feat(False))
    seq = np.stack([feat(False)] * 20 + [feat(True)] * 30 + [feat(False)] * 25)
    t = trim_active(seq)
    assert 30 <= len(t) <= 38
    fixed = prepare_sequence(seq)
    assert fixed.shape == (SEQ_LEN, FEATURE_DIM)
    rows = {r.tobytes() for r in fixed}
    assert rows <= {r.tobytes() for r in seq}           # only real frames, nothing blended
    assert to_fixed(seq[:1]).shape == (SEQ_LEN, FEATURE_DIM)


def test_split_has_no_session_leak():
    labels, mvis = [], []
    for c in range(5):
        for base in range(6):                           # 6 sessions of 3 takes each
            for k in range(3):
                labels.append(f"c{c}"); mvis.append(1000 * c + base * 50 + k)
    labels, mvis = np.array(labels), np.array(mvis)
    split, low = group_split(labels, mvis, seed=1)
    assert not low
    for c in np.unique(labels):
        ix = np.flatnonzero(labels == c)
        gid = session_groups(mvis[ix])
        for g in np.unique(gid):
            assert len(set(split[ix[gid == g]])) == 1   # a session never spans splits
        assert {"train", "val", "test"} <= set(split[ix])


def test_stream_one_commit_per_sign():
    seen = []
    def pf(x):
        seen.append(x.shape); return np.array([0.9, 0.1])
    s = StreamingSession(pf, fps=15, provisional_every=0)
    frames = [feat(False)] * 10 + [feat(True)] * 25 + [feat(False)] * 20 + [feat(True)] * 25 + [feat(False)] * 20
    ends = [e for f in frames for e in s.push(f) if e["type"] == "sign_end"]
    assert len(ends) == 2 and all(sh == (SEQ_LEN, FEATURE_DIM) for sh in seen)


def test_stream_idle_never_commits_and_long_hold_needs_rest():
    s = StreamingSession(lambda x: np.array([1.0]), fps=15, provisional_every=0)
    assert not [e for _ in range(100) for e in s.push(feat(False))]
    s2 = StreamingSession(lambda x: np.array([1.0]), fps=15, max_s=2.0, provisional_every=0)
    ends = [e for _ in range(200) for e in s2.push(feat(True)) if e["type"] == "sign_end"]
    assert len(ends) == 1                               # held hands: one forced commit, then wait for rest


def test_extract_video_subsamples_to_target_fps():
    import cv2, tempfile
    from clarifysign_ml.features import extract_video
    path = os.path.join(tempfile.mkdtemp(), "v.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for _ in range(60): w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    class Stub:
        last = {"lh": True, "rh": False, "pose": True}
        def __call__(self, fr): return np.ones(FEATURE_DIM, np.float32)
    seq, fps, det = extract_video(path, Stub(), target_fps=15.0)
    assert 28 <= len(seq) <= 32 and abs(fps - 15) < 1 and det["lh"] == 1.0 and det["rh"] == 0.0


def test_stream_adapts_to_low_fps():
    s = StreamingSession(lambda x: np.array([1.0]), fps=15, provisional_every=0)
    s.set_fps(6)
    frames = [feat(False)] * 6 + [feat(True)] * 12 + [feat(False)] * 10
    ends = [e for f in frames for e in s.push(f) if e["type"] == "sign_end"]
    assert len(ends) == 1


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
