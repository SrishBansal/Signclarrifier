"""Landmark features. SINGLE source of truth for training AND live serving.

Layout (225 dims): left hand 63 | right hand 63 | pose 99.
- hands: wrist-centred, divided by hand size (xy extent). Hand SHAPE only.
- pose : shoulder-centre origin, divided by shoulder width. Carries hand LOCATION via wrists (15, 16).
- missing body part -> zeros. Nothing is interpolated or invented.
- Detector runs in IMAGE mode per frame (no tracking state), so train and serve behave identically.
- Feed UNMIRRORED frames. If the browser preview is mirrored, send the raw frame, not the mirrored one.
"""
import os
import logging
import numpy as np

FEATURE_DIM = 225
SEQ_LEN = 48
LH, RH, POSE = slice(0, 63), slice(63, 126), slice(126, 225)
ACT_Y = 1.3  # a wrist higher than this (shoulder widths below shoulder line) counts as "signing". Verify per dataset.

# Training data was extracted at 16:9 (y multiplied by 1.8 in Kaggle notebook).
# Live webcam frames may differ; LandmarkExtractor scales y to match.
FEATURE_ASPECT = float(os.environ.get("FEATURE_ASPECT", str(16.0 / 9.0)))


def _first_landmark_group(groups):
    """Return one landmark group whether Tasks returned a flat or nested list."""
    if not groups:
        return None
    return groups if hasattr(groups[0], "x") else groups[0]


def assign_hands_to_pose(hand_groups, pose_groups):
    """Map detected hands to the pose left/right wrist slots by image position.

    Task handedness labels are defined for mirrored/selfie input.  The application
    deliberately sends unmirrored frames, so pose-wrist proximity is the stable
    convention shared with the Holistic features used during training.
    """
    groups = [g for g in hand_groups if g and len(g) == 21]
    pose = _first_landmark_group(pose_groups)
    if not groups or not pose or len(pose) != 33:
        return None, None
    wrists = [(float(pose[15].x), float(pose[15].y)),
              (float(pose[16].x), float(pose[16].y))]

    def distance(hand, wrist):
        return (float(hand[0].x) - wrist[0]) ** 2 + (float(hand[0].y) - wrist[1]) ** 2

    if len(groups) == 1:
        slot = 0 if distance(groups[0], wrists[0]) <= distance(groups[0], wrists[1]) else 1
        return (groups[0], None) if slot == 0 else (None, groups[0])

    # Select the two hands whose wrists best agree with the pose wrists.  This
    # also makes the behaviour deterministic if a future model returns extras.
    best = min(((distance(left, wrists[0]) + distance(right, wrists[1]), left, right)
                for left in groups for right in groups if left is not right), key=lambda x: x[0])
    return best[1], best[2]


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
    """MediaPipe Holistic Landmarker (Tasks API, CPU). Raises if the model cannot load. No silent fallback.
    Resizes every frame to a fixed 640px width to avoid SegmentationSmoothingCalculator size-change errors.
    """

    _TARGET_W = 640

    def __init__(self, model_path, hand_model_path=None):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        self._mp = mp
        self._vision = vision
        self._BaseOptions = BaseOptions
        self._model_path = model_path
        self._hand_model_path = hand_model_path or os.environ.get(
            "HAND_LANDMARKER_TASK", os.path.join(os.path.dirname(model_path), "hand_landmarker.task"))
        self._hand_detection_confidence = float(os.environ.get("HAND_DETECTION_CONFIDENCE", "0.25"))
        self._hand_presence_confidence = float(os.environ.get("HAND_PRESENCE_CONFIDENCE", "0.25"))
        self._hand_tracking_confidence = float(os.environ.get("HAND_TRACKING_CONFIDENCE", "0.25"))
        self.det = None
        self.hand_det = None
        self._last_hw = None  # (h, w) after resize
        self.last = {"lh": False, "rh": False, "pose": False, "aspect": None, "y_scale": 1.0}
        self._build_detector()

    def _build_detector(self):
        opts = self._vision.HolisticLandmarkerOptions(
            base_options=self._BaseOptions(
                model_asset_path=self._model_path,
                delegate=self._BaseOptions.Delegate.CPU),
            running_mode=self._vision.RunningMode.IMAGE,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        if self.det is not None:
            try: self.det.close()
            except Exception: pass
        if self.hand_det is not None:
            try: self.hand_det.close()
            except Exception: pass
        self.hand_det = None
        self.det = self._vision.HolisticLandmarker.create_from_options(opts)
        if not os.path.isfile(self._hand_model_path):
            logging.getLogger("clarifysign.features").warning(
                "Hand fallback disabled: task model is missing at %s", self._hand_model_path)
            return
        hand_opts = self._vision.HandLandmarkerOptions(
            base_options=self._BaseOptions(
                model_asset_path=self._hand_model_path,
                delegate=self._BaseOptions.Delegate.CPU),
            running_mode=self._vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=self._hand_detection_confidence,
            min_hand_presence_confidence=self._hand_presence_confidence,
            min_tracking_confidence=self._hand_tracking_confidence,
        )
        self.hand_det = self._vision.HandLandmarker.create_from_options(hand_opts)

    def __call__(self, frame_bgr):
        import cv2
        h_orig, w_orig = frame_bgr.shape[:2]

        # Resize to fixed width = 640, preserve aspect
        if w_orig != self._TARGET_W:
            new_h = max(1, int(h_orig * self._TARGET_W / max(1, w_orig)))
            frame_bgr = cv2.resize(frame_bgr, (self._TARGET_W, new_h))

        h, w = frame_bgr.shape[:2]
        aspect_orig = w_orig / h_orig if h_orig > 0 else 1.0
        k = FEATURE_ASPECT / aspect_orig if aspect_orig > 1e-3 else 1.0

        # Recreate detector if frame size changed (prevents SegmentationSmoothingCalculator crash)
        hw = (h, w)
        if self._last_hw is not None and hw != self._last_hw:
            logging.getLogger("clarifysign.features").info(
                "Frame size changed %s->%s, recreating detector", self._last_hw, hw)
            self._build_detector()
        self._last_hw = hw

        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        try:
            res = self.det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        except Exception as exc:
            logging.getLogger("clarifysign.features").warning("holistic detect() failed: %s", exc)
            self.last = {"lh": False, "rh": False, "pose": False,
                         "aspect": round(aspect_orig, 3), "y_scale": round(k, 4)}
            return np.zeros(FEATURE_DIM, np.float32)

        def _scale_y(lm_list):
            if not lm_list:
                return lm_list
            class _LM:
                __slots__ = ("x", "y", "z")
                def __init__(self, x, y, z):
                    self.x = x; self.y = y; self.z = z
            scaled = []
            for grp in ([lm_list] if not hasattr(lm_list[0], "__iter__") else lm_list):
                try:
                    scaled.append([_LM(lm.x, lm.y * k, lm.z) for lm in grp])
                except TypeError:
                    scaled.append([_LM(lm.x, lm.y * k, lm.z) for lm in lm_list])
                    break
            return scaled if scaled else lm_list

        raw_lh = _first_landmark_group(res.left_hand_landmarks)
        raw_rh = _first_landmark_group(res.right_hand_landmarks)
        source_lh = "holistic" if raw_lh else "none"
        source_rh = "holistic" if raw_rh else "none"
        if self.hand_det is not None and (not raw_lh or not raw_rh):
            try:
                hand_res = self.hand_det.detect(self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB, data=rgb))
                fallback_lh, fallback_rh = assign_hands_to_pose(
                    hand_res.hand_landmarks, res.pose_landmarks)
                if not raw_lh and fallback_lh:
                    raw_lh, source_lh = fallback_lh, "hand_fallback"
                if not raw_rh and fallback_rh:
                    raw_rh, source_rh = fallback_rh, "hand_fallback"
            except Exception as exc:
                logging.getLogger("clarifysign.features").warning("hand fallback detect() failed: %s", exc)

        lh   = _scale_y(raw_lh) if raw_lh else None
        rh   = _scale_y(raw_rh) if raw_rh else None
        pose = _scale_y(res.pose_landmarks)        if res.pose_landmarks        else None

        # Keep lightweight normalized coordinates for the debug-frame endpoint.
        # They are intentionally the detector's original image coordinates, not
        # normalized training features.
        def _debug_points(groups):
            if not groups:
                return []
            points = groups[0] if hasattr(groups[0], "__iter__") else groups
            return [(float(lm.x), float(lm.y)) for lm in points]

        self.last = {"lh": bool(raw_lh), "rh": bool(raw_rh),
                     "pose": bool(res.pose_landmarks), "aspect": round(aspect_orig, 3), "y_scale": round(k, 4),
                     "hand_source": {"lh": source_lh, "rh": source_rh},
                     "landmarks": {"lh": _debug_points(raw_lh),
                                   "rh": _debug_points(raw_rh),
                                   "pose": _debug_points(res.pose_landmarks)}}
        return landmarks_to_features(lh, rh, pose)

    def close(self):
        if self.det is not None:
            try: self.det.close()
            except Exception: pass
        if self.hand_det is not None:
            try: self.hand_det.close()
            except Exception: pass




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
