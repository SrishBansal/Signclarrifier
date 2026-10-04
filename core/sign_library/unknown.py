"""Visible 'Unknown Sign' gesture sequence (48 frames, 225 dims).

Signer lifts both open palms upward at chest height in a clear,
expressive 'unknown / what / shrug' sign.
Ensures missing signs NEVER result in a silent skip.
"""
import numpy as np
from .rest_pose import get_canonical_rest_frame, make_canonical_pose, make_canonical_hand


def make_open_palm_up_hand(is_left: bool = False) -> np.ndarray:
    """Hand shape: flat open hand, fingers spread naturally, palm facing up."""
    h = np.zeros((21, 3), dtype=np.float32)
    sign = -1.0 if is_left else 1.0

    # Wrist at (0, 0, 0)
    # Thumb spread outward
    h[1] = [sign * 0.25, 0.08, -0.02]
    h[2] = [sign * 0.45, 0.16, -0.05]
    h[3] = [sign * 0.58, 0.25, -0.08]
    h[4] = [sign * 0.70, 0.35, -0.10]

    # Index finger extended straight forward/up
    h[5] = [sign * 0.18, 0.32, 0.0]
    h[6] = [sign * 0.20, 0.58, 0.02]
    h[7] = [sign * 0.20, 0.78, 0.04]
    h[8] = [sign * 0.20, 0.95, 0.05]

    # Middle finger extended straight
    h[9]  = [0.0, 0.35, 0.0]
    h[10] = [0.0, 0.62, 0.02]
    h[11] = [0.0, 0.82, 0.04]
    h[12] = [0.0, 1.00, 0.05]

    # Ring finger extended
    h[13] = [-sign * 0.16, 0.32, 0.0]
    h[14] = [-sign * 0.18, 0.56, 0.02]
    h[15] = [-sign * 0.18, 0.76, 0.04]
    h[16] = [-sign * 0.18, 0.92, 0.05]

    # Pinky finger extended slightly flared
    h[17] = [-sign * 0.30, 0.28, 0.0]
    h[18] = [-sign * 0.34, 0.48, 0.02]
    h[19] = [-sign * 0.36, 0.66, 0.04]
    h[20] = [-sign * 0.38, 0.82, 0.05]

    norms = np.linalg.norm(h[:, :2], axis=1)
    max_norm = float(norms.max())
    if max_norm > 1e-5:
        h = h / max_norm
    return h.reshape(-1)


def generate_unknown_sign_sequence(n_frames: int = 48) -> np.ndarray:
    """Generates an expressive 48-frame sequence for UNKNOWN_SIGN.
    
    Phases:
    - 0 to 14: Transition from rest into chest-level palms-up shrug
    - 14 to 34: Hold expressive unknown shrug gesture with subtle natural sway
    - 34 to 47: Smooth return toward rest pose
    """
    seq = np.zeros((n_frames, 225), dtype=np.float32)
    rest_frame = get_canonical_rest_frame()
    rest_pose = rest_frame[126:].reshape(33, 3)

    lh_open = make_open_palm_up_hand(is_left=True)
    rh_open = make_open_palm_up_hand(is_left=False)

    for i in range(n_frames):
        t = i / (n_frames - 1)
        
        # Smooth weight curve: 0 -> 1 -> 0
        if t < 0.3:
            s = 0.5 - 0.5 * np.cos(np.pi * (t / 0.3))
        elif t < 0.7:
            s = 1.0 + 0.05 * np.sin(np.pi * 2 * (t - 0.3) / 0.4)
        else:
            s = 0.5 + 0.5 * np.cos(np.pi * ((t - 0.7) / 0.3))

        # Hands blend
        lh = (1 - s) * rest_frame[:63] + s * lh_open
        rh = (1 - s) * rest_frame[63:126] + s * rh_open

        # Pose computation
        p = rest_pose.copy()

        # Lift shoulders slightly during shrug
        p[11, 1] -= 0.12 * s
        p[12, 1] -= 0.12 * s

        # Tilt head slightly questioning
        p[0, 0] += 0.06 * s
        p[0, 1] -= 0.03 * s

        # Wrists move up to chest level and spread outward
        # Left wrist (15)
        p[15, 0] = (1 - s) * rest_pose[15, 0] + s * 0.65
        p[15, 1] = (1 - s) * rest_pose[15, 1] + s * 0.75
        p[15, 2] = (1 - s) * rest_pose[15, 2] + s * (-0.2)

        # Right wrist (16)
        p[16, 0] = (1 - s) * rest_pose[16, 0] + s * (-0.65)
        p[16, 1] = (1 - s) * rest_pose[16, 1] + s * 0.75
        p[16, 2] = (1 - s) * rest_pose[16, 2] + s * (-0.2)

        # Elbows bend naturally to accommodate lifted wrists
        p[13, 0] = (1 - s) * rest_pose[13, 0] + s * 0.72
        p[13, 1] = (1 - s) * rest_pose[13, 1] + s * 1.05
        p[14, 0] = (1 - s) * rest_pose[14, 0] + s * (-0.72)
        p[14, 1] = (1 - s) * rest_pose[14, 1] + s * 1.05

        seq[i] = np.concatenate([lh, rh, p.reshape(-1)])

    return seq
