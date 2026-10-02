"""Landmark features. SINGLE source of truth for training AND live serving.

Layout (225 dims): left hand 63 | right hand 63 | pose 99.
- hands: wrist-centred, divided by hand size (xy extent). Hand SHAPE only.
- pose : shoulder-centre origin, divided by shoulder width. Carries hand LOCATION via wrists (15, 16).
- missing body part -> zeros. Nothing is interpolated or invented.
- Detector runs in IMAGE mode per frame (no tracking state), so train and serve behave identically.
- Feed UNMIRRORED frames. If the browser preview is mirrored, send the raw frame, not the mirrored one.
"""
import numpy as np

FEATURE_DIM = 225
SEQ_LEN = 48
LH, RH, POSE = slice(0, 63), slice(63, 126), slice(126, 225)
ACT_Y = 1.3  # a wrist higher than this (shoulder widths below shoulder line) counts as "signing". Verify per dataset.


def _arr(lms, n):
    if not lms:
        return None
    if not hasattr(lms[0], "x"):  # nested list form
        lms = lms[0]
    if len(lms) != n:
        return None
    return np.array([[l.x, l.y, l.z] for l in lms], np.float32)


def _norm_hand(a):
    if a is None:
        return np.zeros(63, np.float32)
    c = a - a[0]
    s = float(np.linalg.norm(c[:, :2], axis=1).max())
    if s < 1e-6:
        return np.zeros(63, np.float32)
    return (c / s).reshape(-1).astype(np.float32)


def _norm_pose(a):
    if a is None:
        return np.zeros(99, np.float32)
    ctr = (a[11] + a[12]) / 2
    w = float(np.linalg.norm(a[11, :2] - a[12, :2]))
    if w < 1e-3:
        return np.zeros(99, np.float32)
    return ((a - ctr) / w).reshape(-1).astype(np.float32)


def landmarks_to_features(left_hand, right_hand, pose):
    """Raw MediaPipe landmark lists (or None) -> (225,) float32."""
    return np.concatenate([_norm_hand(_arr(left_hand, 21)),
                           _norm_hand(_arr(right_hand, 21)),
                           _norm_pose(_arr(pose, 33))])


def frame_activity(feat, act_y=ACT_Y):
    p = feat[POSE].reshape(33, 3)
    if not p.any():
        return False
    return bool(min(p[15, 1], p[16, 1]) < act_y)


def trim_active(seq, act_y=ACT_Y, pad=4, min_active=8):
    """Cut idle lead-in/out (arms down). Falls back to the whole clip if too little activity."""
    act = np.array([frame_activity(f, act_y) for f in seq])
    idx = np.flatnonzero(act)
    if len(idx) < min_active:
        return seq
    return seq[max(0, idx[0] - pad): idx[-1] + pad + 1]


def to_fixed(seq, n=SEQ_LEN):
    """Nearest-frame temporal resampling to n frames. Only real frames are used (no blending with zeros)."""
    t = len(seq)
    if t == 0:
        raise ValueError("empty sequence")
    idx = np.round(np.linspace(0, t - 1, n)).astype(int)
    return np.asarray(seq)[idx].astype(np.float32)


def prepare_sequence(seq, act_y=ACT_Y):
    """The ONE preprocessing used for training and serving: trim idle, resample to 48."""
    return to_fixed(trim_active(np.asarray(seq, np.float32), act_y))


class LandmarkExtractor:
    """MediaPipe Holistic Landmarker (Tasks API, CPU). Raises if the model cannot load. No silent fallback."""

    def __init__(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        self._mp = mp
        opts = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self.det = vision.HolisticLandmarker.create_from_options(opts)
        self.last = {"lh": False, "rh": False, "pose": False}

    def __call__(self, frame_bgr):
        import cv2
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self.det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        self.last = {"lh": bool(res.left_hand_landmarks), "rh": bool(res.right_hand_landmarks),
                     "pose": bool(res.pose_landmarks)}
        return landmarks_to_features(res.left_hand_landmarks, res.right_hand_landmarks, res.pose_landmarks)

    def close(self):
        self.det.close()


def extract_video(path, extractor, max_width=640, target_fps=15.0):
    """Video file -> (raw per-frame features (n,225), effective fps, per-part detection rates).

    Frames are subsampled to ~target_fps (a live webcam stream runs at a similar rate, and it halves extraction time).
    """
    import cv2
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1.0, src_fps / target_fps)
    rows, det, i, nxt = [], {"lh": 0, "rh": 0, "pose": 0}, 0, 0.0
    while True:
        if i < nxt:
            if not cap.grab():
                break
            i += 1
            continue
        ok, fr = cap.read()
        if not ok:
            break
        i += 1; nxt += step
        h, w = fr.shape[:2]
        if w > max_width:
            fr = cv2.resize(fr, (max_width, int(h * max_width / w)))
        rows.append(extractor(fr))
        for k in det:
            det[k] += int(extractor.last[k])
    cap.release()
    n = max(len(rows), 1)
    eff = src_fps / step
    return (np.stack(rows) if rows else np.zeros((0, FEATURE_DIM), np.float32)), float(eff), {k: v / n for k, v in det.items()}
