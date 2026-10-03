"""Fingerspelling generator for ISL alphabet (A-Z, 0-9).

Generates keyframe sequences for individual letters and spelled words.
Dominant hand (right hand) performs the articulated letter shape in
front of chest/shoulder, left hand remains in relaxed rest position.
"""
from typing import Dict, List, Optional
import numpy as np
from .rest_pose import get_canonical_rest_frame, make_canonical_pose, make_canonical_hand


def _make_letter_handshape(letter: str) -> np.ndarray:
    """Generate 21-landmark normalized hand shape for English/ISL alphabet letter."""
    letter = letter.upper()
    h = np.zeros((21, 3), dtype=np.float32)

    # Wrist is always at (0, 0, 0)
    # Different letter poses:
    if letter == "A":
        # Fist with thumb along index side
        # Thumb straight alongside index knuckle
        h[1] = [0.2, 0.2, 0.0]
        h[2] = [0.35, 0.38, 0.0]
        h[3] = [0.45, 0.55, 0.0]
        h[4] = [0.45, 0.70, 0.0]
        # 4 fingers folded into fist
        for base, idx in [(0.15, 5), (0.0, 9), (-0.15, 13), (-0.3, 17)]:
            h[idx]   = [base, 0.35, 0.0]
            h[idx+1] = [base, 0.52, -0.15]
            h[idx+2] = [base, 0.40, -0.28]
            h[idx+3] = [base, 0.25, -0.30]

    elif letter == "B":
        # 4 fingers straight up together, thumb crossed in front of palm
        h[1] = [0.15, 0.15, -0.05]
        h[2] = [0.22, 0.28, -0.10]
        h[3] = [0.15, 0.35, -0.12]
        h[4] = [0.05, 0.38, -0.12]
        # 4 fingers straight vertical
        for base, idx in [(0.12, 5), (0.02, 9), (-0.08, 13), (-0.18, 17)]:
            h[idx]   = [base, 0.35, 0.0]
            h[idx+1] = [base, 0.60, 0.0]
            h[idx+2] = [base, 0.80, 0.0]
            h[idx+3] = [base, 0.98, 0.0]

    elif letter == "C":
        # Curved C shape for all fingers and thumb
        h[1] = [0.22, 0.15, -0.05]
        h[2] = [0.38, 0.28, -0.10]
        h[3] = [0.42, 0.42, -0.10]
        h[4] = [0.35, 0.55, -0.08]
        for base, idx in [(0.15, 5), (0.02, 9), (-0.10, 13), (-0.22, 17)]:
            h[idx]   = [base, 0.35, 0.0]
            h[idx+1] = [base, 0.55, -0.15]
            h[idx+2] = [base, 0.50, -0.30]
            h[idx+3] = [base, 0.35, -0.38]

    elif letter == "D":
        # Index straight up, other 3 touch thumb in circle
        # Index straight
        h[5] = [0.10, 0.35, 0.0]; h[6] = [0.10, 0.60, 0.0]; h[7] = [0.10, 0.80, 0.0]; h[8] = [0.10, 1.0, 0.0]
        # Thumb
        h[1] = [0.18, 0.18, -0.05]; h[2] = [0.25, 0.30, -0.12]; h[3] = [0.18, 0.40, -0.15]; h[4] = [0.05, 0.42, -0.15]
        # Middle, ring, pinky curved to touch thumb
        for base, idx in [(0.0, 9), (-0.12, 13), (-0.24, 17)]:
            h[idx]   = [base, 0.32, 0.0]
            h[idx+1] = [base, 0.45, -0.12]
            h[idx+2] = [base, 0.42, -0.22]
            h[idx+3] = [base, 0.32, -0.20]

    elif letter == "I":
        # Pinky straight up, fist for the rest
        h[17] = [-0.20, 0.32, 0.0]; h[18] = [-0.22, 0.55, 0.0]; h[19] = [-0.22, 0.75, 0.0]; h[20] = [-0.22, 0.95, 0.0]
        # Thumb over folded fingers
        h[1] = [0.15, 0.18, -0.05]; h[2] = [0.22, 0.30, -0.12]; h[3] = [0.12, 0.38, -0.15]; h[4] = [0.0, 0.38, -0.15]
        for base, idx in [(0.12, 5), (0.0, 9), (-0.10, 13)]:
            h[idx]   = [base, 0.32, 0.0]
            h[idx+1] = [base, 0.48, -0.12]
            h[idx+2] = [base, 0.38, -0.22]
            h[idx+3] = [base, 0.22, -0.22]

    elif letter == "L":
        # Index straight up, thumb straight out at 90 deg, others folded
        h[1] = [0.25, 0.15, 0.0]; h[2] = [0.45, 0.22, 0.0]; h[3] = [0.65, 0.25, 0.0]; h[4] = [0.85, 0.25, 0.0]
        h[5] = [0.08, 0.35, 0.0]; h[6] = [0.08, 0.60, 0.0]; h[7] = [0.08, 0.80, 0.0]; h[8] = [0.08, 1.0, 0.0]
        for base, idx in [(0.0, 9), (-0.12, 13), (-0.24, 17)]:
            h[idx]   = [base, 0.32, 0.0]
            h[idx+1] = [base, 0.48, -0.12]
            h[idx+2] = [base, 0.38, -0.22]
            h[idx+3] = [base, 0.22, -0.22]

    elif letter == "V":
        # Index & Middle extended in V shape, others folded
        h[5] = [0.15, 0.35, 0.0]; h[6] = [0.20, 0.60, 0.0]; h[7] = [0.25, 0.80, 0.0]; h[8] = [0.30, 1.0, 0.0]
        h[9] = [0.00, 0.35, 0.0]; h[10] = [-0.05, 0.60, 0.0]; h[11] = [-0.10, 0.80, 0.0]; h[12] = [-0.15, 1.0, 0.0]
        h[1] = [0.15, 0.18, -0.05]; h[2] = [0.20, 0.30, -0.12]; h[3] = [0.10, 0.38, -0.15]; h[4] = [0.0, 0.38, -0.15]
        for base, idx in [(-0.12, 13), (-0.24, 17)]:
            h[idx]   = [base, 0.32, 0.0]
            h[idx+1] = [base, 0.48, -0.12]
            h[idx+2] = [base, 0.38, -0.22]
            h[idx+3] = [base, 0.22, -0.22]

    else:
        # Open 5-spread palm facing outward as clean fallback
        h[1] = [0.25, 0.15, 0.0]; h[2] = [0.45, 0.25, 0.0]; h[3] = [0.60, 0.35, 0.0]; h[4] = [0.72, 0.45, 0.0]
        for base, idx in [(0.15, 5), (0.02, 9), (-0.12, 13), (-0.25, 17)]:
            h[idx]   = [base, 0.32, 0.0]
            h[idx+1] = [base, 0.58, 0.0]
            h[idx+2] = [base, 0.78, 0.0]
            h[idx+3] = [base, 0.95, 0.0]

    # Normalize max norm to 1.0
    norms = np.linalg.norm(h[:, :2], axis=1)
    max_norm = float(norms.max())
    if max_norm > 1e-5:
        h = h / max_norm
    return h.reshape(-1)


def generate_letter_sequence(letter: str, n_frames: int = 48) -> np.ndarray:
    """Generate a 48-frame sequence for a fingerspelled letter."""
    seq = np.zeros((n_frames, 225), dtype=np.float32)
    rest_frame = get_canonical_rest_frame()
    rest_pose = rest_frame[126:].reshape(33, 3)

    lh_rest = rest_frame[:63]
    rh_letter = _make_letter_handshape(letter)

    for i in range(n_frames):
        t = i / (n_frames - 1)
        if t < 0.25:
            s = 0.5 - 0.5 * np.cos(np.pi * (t / 0.25))
        elif t < 0.75:
            s = 1.0
        else:
            s = 0.5 + 0.5 * np.cos(np.pi * ((t - 0.75) / 0.25))

        # Left hand rests, Right hand transitions to letter shape
        lh = lh_rest
        rh = (1 - s) * rest_frame[63:126] + s * rh_letter

        # Pose: right arm moves up to chest signing space
        p = rest_pose.copy()
        # Right wrist (16): lifts to upper chest (-0.35, 0.35, -0.25)
        p[16, 0] = (1 - s) * rest_pose[16, 0] + s * (-0.35)
        p[16, 1] = (1 - s) * rest_pose[16, 1] + s * 0.40
        p[16, 2] = (1 - s) * rest_pose[16, 2] + s * (-0.25)

        # Right elbow (14)
        p[14, 0] = (1 - s) * rest_pose[14, 0] + s * (-0.60)
        p[14, 1] = (1 - s) * rest_pose[14, 1] + s * 0.85
        p[14, 2] = (1 - s) * rest_pose[14, 2] + s * (-0.10)

        seq[i] = np.concatenate([lh, rh, p.reshape(-1)])

    return seq
