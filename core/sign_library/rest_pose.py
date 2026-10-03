"""Canonical rest pose for ISL signer avatar.

Normalized coordinate frame:
- Origin (0, 0, 0) is shoulder center.
- X goes left/right (shoulder span is 1.0; 11 left shoulder is +0.5, 12 right shoulder is -0.5).
- Y points downwards (MediaPipe convention: negative Y is head, positive Y is torso/hips).
- 225 dims = 63 (LH) + 63 (RH) + 99 (Pose).
"""
import numpy as np

def make_canonical_hand(is_left: bool = False) -> np.ndarray:
    """Generate a relaxed natural hand pose (21 landmarks, 3D)."""
    # 21 landmarks: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
    h = np.zeros((21, 3), dtype=np.float32)
    h[0] = [0.0, 0.0, 0.0]  # wrist

    sign = -1.0 if is_left else 1.0

    # Thumb: relaxed curve inward
    h[1] = [sign * 0.15, 0.10, -0.05]
    h[2] = [sign * 0.28, 0.20, -0.10]
    h[3] = [sign * 0.35, 0.32, -0.15]
    h[4] = [sign * 0.38, 0.42, -0.18]

    # Index finger: slight natural curl
    h[5] = [sign * 0.10, 0.35, 0.0]
    h[6] = [sign * 0.11, 0.55, -0.05]
    h[7] = [sign * 0.11, 0.72, -0.12]
    h[8] = [sign * 0.10, 0.85, -0.18]

    # Middle finger: longest, slight curl
    h[9]  = [0.0, 0.38, 0.0]
    h[10] = [0.0, 0.60, -0.05]
    h[11] = [0.0, 0.78, -0.12]
    h[12] = [0.0, 0.92, -0.18]

    # Ring finger
    h[13] = [-sign * 0.09, 0.35, 0.0]
    h[14] = [-sign * 0.10, 0.54, -0.05]
    h[15] = [-sign * 0.10, 0.70, -0.12]
    h[16] = [-sign * 0.10, 0.82, -0.18]

    # Pinky finger
    h[17] = [-sign * 0.18, 0.30, 0.0]
    h[18] = [-sign * 0.20, 0.46, -0.05]
    h[19] = [-sign * 0.20, 0.60, -0.12]
    h[20] = [-sign * 0.20, 0.72, -0.18]

    # Normalize max norm to 1.0
    norms = np.linalg.norm(h[:, :2], axis=1)
    max_norm = float(norms.max())
    if max_norm > 1e-5:
        h = h / max_norm

    return h.reshape(-1)


def make_canonical_pose() -> np.ndarray:
    """Generate a neutral, relaxed standing/signing pose (33 landmarks, 3D)."""
    p = np.zeros((33, 3), dtype=np.float32)

    # Head / Face
    p[0] = [0.0, -1.0, -0.3]   # Nose
    p[1] = [0.08, -1.06, -0.3]  # Left eye inner
    p[2] = [0.14, -1.06, -0.3]  # Left eye
    p[3] = [0.20, -1.06, -0.3]  # Left eye outer
    p[4] = [-0.08, -1.06, -0.3] # Right eye inner
    p[5] = [-0.14, -1.06, -0.3] # Right eye
    p[6] = [-0.20, -1.06, -0.3] # Right eye outer
    p[7] = [0.35, -1.0, 0.0]   # Left ear
    p[8] = [-0.35, -1.0, 0.0]  # Right ear
    p[9] = [0.12, -0.85, -0.2]  # Mouth left
    p[10] = [-0.12, -0.85, -0.2]# Mouth right

    # Shoulders (span = 1.0 centered at 0)
    p[11] = [0.5, 0.0, 0.0]    # Left shoulder
    p[12] = [-0.5, 0.0, 0.0]   # Right shoulder

    # Elbows: naturally angled down along the torso
    p[13] = [0.58, 1.45, 0.15]  # Left elbow
    p[14] = [-0.58, 1.45, 0.15] # Right elbow

    # Wrists: resting near upper thigh / waist (below activity threshold)
    p[15] = [0.48, 2.75, 0.0]   # Left wrist
    p[16] = [-0.48, 2.75, 0.0]  # Right wrist

    # Hands indices 17-22 in pose
    p[17] = [0.50, 2.95, 0.0]   # Left pinky
    p[18] = [-0.50, 2.95, 0.0]  # Right pinky
    p[19] = [0.46, 2.98, 0.0]   # Left index
    p[20] = [-0.46, 2.98, 0.0]  # Right index
    p[21] = [0.44, 2.88, 0.0]   # Left thumb
    p[22] = [-0.44, 2.88, 0.0]  # Right thumb

    # Hips
    p[23] = [0.28, 2.50, 0.0]   # Left hip
    p[24] = [-0.28, 2.50, 0.0]  # Right hip

    # Legs (simplified for half-body signer)
    p[25] = [0.30, 4.20, 0.0]
    p[26] = [-0.30, 4.20, 0.0]
    p[27] = [0.32, 6.00, 0.0]
    p[28] = [-0.32, 6.00, 0.0]
    p[29] = [0.32, 6.20, 0.0]
    p[30] = [-0.32, 6.20, 0.0]
    p[31] = [0.32, 6.30, 0.3]
    p[32] = [-0.32, 6.30, 0.3]

    return p.reshape(-1)


def get_canonical_rest_frame() -> np.ndarray:
    """Returns a full 225-dim canonical rest frame (float32)."""
    lh = make_canonical_hand(is_left=True)
    rh = make_canonical_hand(is_left=False)
    pose = make_canonical_pose()
    return np.concatenate([lh, rh, pose]).astype(np.float32)
