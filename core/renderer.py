"""Decoupled Avatar Renderer module (zero dependencies on NLU/planner).

Features:
- Proper articulated avatar: torso, head, rounded thick bones, hands with per-finger bones.
- Real connection tables for pose and hands (no dot clouds).
- Two-bone analytical IK for arms (shoulder -> elbow -> wrist).
- Timeline ease-in-out LERP between clips and rest pose.
- Deterministic canvas / image hashing for headless verification.
"""
import os
import math
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from PIL import Image, ImageDraw

from .sign_library.library import SignLibrary, SignTimeline, lerp_frames, ease_in_out
from .sign_library.rest_pose import get_canonical_rest_frame

# Real MediaPipe Hand Connections (21 landmarks)
HAND_PALM_CONNECTIONS = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
HAND_FINGER_CONNECTIONS = [
    # Thumb (1-4)
    [(1, 2), (2, 3), (3, 4)],
    # Index (5-8)
    [(5, 6), (6, 7), (7, 8)],
    # Middle (9-12)
    [(9, 10), (10, 11), (11, 12)],
    # Ring (13-16)
    [(13, 14), (14, 15), (15, 16)],
    # Pinky (17-20)
    [(17, 18), (18, 19), (19, 20)],
]

# Real MediaPipe Pose Connections (Torso & Face)
TORSO_CONNECTIONS = [
    (11, 12),  # Shoulders
    (11, 23),  # Left side
    (12, 24),  # Right side
    (23, 24),  # Hips
]


def solve_two_bone_ik(
    shoulder: np.ndarray,
    wrist: np.ndarray,
    elbow_hint: Optional[np.ndarray] = None,
    is_left: bool = True,
    l1: float = 1.45,
    l2: float = 1.35
) -> np.ndarray:
    """Analytical two-bone inverse kinematics for arm in 2D plane.

    Returns the solved elbow joint position.
    """
    sw = wrist[:2] - shoulder[:2]
    d = float(np.linalg.norm(sw))

    if d < 1e-4:
        # Wrist at shoulder: point elbow outward
        offset = np.array([l1 if is_left else -l1, 0.0], dtype=np.float32)
        return np.array([shoulder[0] + offset[0], shoulder[1] + offset[1], shoulder[2]], dtype=np.float32)

    # Unit vector from shoulder to wrist
    u = sw / d
    # Outward normal vector
    norm_sign = 1.0 if is_left else -1.0
    v = np.array([-u[1] * norm_sign, u[0] * norm_sign], dtype=np.float32)

    if elbow_hint is not None:
        # Determine bending direction from hint if available
        eh = elbow_hint[:2] - shoulder[:2]
        proj = float(np.dot(eh, v))
        if proj < 0:
            v = -v

    # Clamp distance within reachable range
    if d >= (l1 + l2):
        # Arm fully extended
        elbow_2d = shoulder[:2] + (l1 / d) * sw
    elif d <= abs(l1 - l2):
        # Fully folded
        elbow_2d = shoulder[:2] + l1 * u
    else:
        # Law of cosines
        cos_alpha = (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        sin_alpha = math.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha))

        elbow_2d = shoulder[:2] + l1 * (cos_alpha * u + sin_alpha * v)

    z = float((shoulder[2] + wrist[2]) * 0.5)
    return np.array([elbow_2d[0], elbow_2d[1], z], dtype=np.float32)


class AvatarRenderer:
    """Headless 2D Avatar Renderer producing proper articulated avatar graphics."""

    def __init__(
        self,
        sign_library: Optional[SignLibrary] = None,
        width: int = 640,
        height: int = 520,
        use_ik: bool = True
    ):
        self.library = sign_library or SignLibrary()
        self.width = width
        self.height = height
        self.use_ik = use_ik

        # Coordinate transformation parameters
        self.scale = height * 0.28
        self.origin_x = width * 0.50
        self.origin_y = height * 0.38

        # Styling palette: modern clean avatar
        self.palette = {
            "bg": (248, 249, 252),
            "torso": (65, 84, 130),
            "torso_line": (40, 55, 90),
            "neck": (224, 182, 160),
            "head": (238, 202, 178),
            "head_stroke": (195, 155, 130),
            "eyes": (45, 52, 65),
            "arm_left": (78, 110, 175),
            "arm_right": (62, 92, 155),
            "joint": (35, 48, 80),
            "palm": (230, 192, 168),
            "finger_thumb": (225, 112, 85),
            "finger_index": (9, 132, 227),
            "finger_middle": (0, 184, 148),
            "finger_ring": (108, 92, 231),
            "finger_pinky": (253, 121, 168),
        }

    def _to_screen(self, p: Union[np.ndarray, List[float]]) -> Tuple[float, float]:
        """Convert normalized (x, y) coordinates to screen pixel coordinates."""
        # p[0] is X, p[1] is Y (positive down)
        # Note: in mirror/self-view, x is unmirrored: left arm (11) is x > 0
        x = self.origin_x + float(p[0]) * self.scale
        y = self.origin_y + float(p[1]) * self.scale
        return (x, y)

    def render_frame(
        self,
        frame: np.ndarray,
        gloss_caption: str = "",
        status_label: str = ""
    ) -> Image.Image:
        """Renders a single 225-dim landmark frame to a high-quality PIL Image."""
        img = Image.new("RGB", (self.width, self.height), self.palette["bg"])
        draw = ImageDraw.Draw(img)

        # Unpack frame (225,)
        lh_raw = frame[:63].reshape(21, 3)
        rh_raw = frame[63:126].reshape(21, 3)
        pose_raw = frame[126:].reshape(33, 3)

        # 1. Torso Polygon & Collar
        p_l_sh = pose_raw[11]
        p_r_sh = pose_raw[12]
        p_l_hip = pose_raw[23]
        p_r_hip = pose_raw[24]

        # Check pose validity
        has_pose = bool(np.any(pose_raw[11:13] != 0))
        if has_pose:
            pt_l_sh = self._to_screen(p_l_sh)
            pt_r_sh = self._to_screen(p_r_sh)
            pt_l_hip = self._to_screen(p_l_hip)
            pt_r_hip = self._to_screen(p_r_hip)

            # Neck anchor
            neck_center = ((p_l_sh + p_r_sh) * 0.5)
            pt_neck = self._to_screen(neck_center)

            # Torso fill
            torso_poly = [pt_l_sh, pt_r_sh, pt_r_hip, pt_l_hip]
            draw.polygon(torso_poly, fill=self.palette["torso"])

            # Torso boundary strokes
            for p1, p2 in [(pt_l_sh, pt_r_sh), (pt_r_sh, pt_r_hip), (pt_r_hip, pt_l_hip), (pt_l_hip, pt_l_sh)]:
                draw.line([p1, p2], fill=self.palette["torso_line"], width=6)

            # Spine line
            pt_mid_hip = ((pt_l_hip[0] + pt_r_hip[0]) * 0.5, (pt_l_hip[1] + pt_r_hip[1]) * 0.5)
            draw.line([pt_neck, pt_mid_hip], fill=self.palette["torso_line"], width=3)

            # 2. Head and Face
            # Center of head
            nose = pose_raw[0]
            pt_nose = self._to_screen(nose)
            head_center = (pt_neck[0] * 0.2 + pt_nose[0] * 0.8, pt_neck[1] * 0.2 + pt_nose[1] * 0.8)

            # Neck line
            draw.line([pt_neck, head_center], fill=self.palette["neck"], width=14)

            # Head oval
            head_rx = self.scale * 0.28
            head_ry = self.scale * 0.35
            head_box = [
                head_center[0] - head_rx, head_center[1] - head_ry,
                head_center[0] + head_rx, head_center[1] + head_ry
            ]
            draw.ellipse(head_box, fill=self.palette["head"], outline=self.palette["head_stroke"], width=4)

            # Eyes
            eye_offset_x = head_rx * 0.38
            eye_y = head_center[1] - head_ry * 0.08
            eye_r = max(3.0, self.scale * 0.035)
            draw.ellipse([head_center[0] - eye_offset_x - eye_r, eye_y - eye_r,
                          head_center[0] - eye_offset_x + eye_r, eye_y + eye_r], fill=self.palette["eyes"])
            draw.ellipse([head_center[0] + eye_offset_x - eye_r, eye_y - eye_r,
                          head_center[0] + eye_offset_x + eye_r, eye_y + eye_r], fill=self.palette["eyes"])

            # Eyebrows
            draw.line([
                (head_center[0] - eye_offset_x - eye_r * 1.5, eye_y - eye_r * 2),
                (head_center[0] - eye_offset_x + eye_r * 1.5, eye_y - eye_r * 2.2)
            ], fill=self.palette["eyes"], width=3)
            draw.line([
                (head_center[0] + eye_offset_x - eye_r * 1.5, eye_y - eye_r * 2.2),
                (head_center[0] + eye_offset_x + eye_r * 1.5, eye_y - eye_r * 2)
            ], fill=self.palette["eyes"], width=3)

            # Smile / mouth
            mouth_y = head_center[1] + head_ry * 0.45
            draw.line([
                (head_center[0] - head_rx * 0.22, mouth_y),
                (head_center[0] + head_rx * 0.22, mouth_y)
            ], fill=self.palette["head_stroke"], width=3)

            # 3. Arms with Two-Bone IK
            # Left Arm (pose 11 -> 13 -> 15)
            w_left = pose_raw[15]
            if self.use_ik and np.any(w_left != 0):
                e_left = solve_two_bone_ik(p_l_sh, w_left, pose_raw[13], is_left=True)
            else:
                e_left = pose_raw[13]

            pt_e_l = self._to_screen(e_left)
            pt_w_l = self._to_screen(w_left)

            # Right Arm (pose 12 -> 14 -> 16)
            w_right = pose_raw[16]
            if self.use_ik and np.any(w_right != 0):
                e_right = solve_two_bone_ik(p_r_sh, w_right, pose_raw[14], is_left=False)
            else:
                e_right = pose_raw[14]

            pt_e_r = self._to_screen(e_right)
            pt_w_r = self._to_screen(w_right)

            # Draw arm bones with thick rounded strokes
            # Left upper arm & forearm
            draw.line([pt_l_sh, pt_e_l], fill=self.palette["arm_left"], width=12)
            draw.line([pt_e_l, pt_w_l], fill=self.palette["arm_left"], width=10)
            draw.ellipse([pt_e_l[0]-6, pt_e_l[1]-6, pt_e_l[0]+6, pt_e_l[1]+6], fill=self.palette["joint"])

            # Right upper arm & forearm
            draw.line([pt_r_sh, pt_e_r], fill=self.palette["arm_right"], width=12)
            draw.line([pt_e_r, pt_w_r], fill=self.palette["arm_right"], width=10)
            draw.ellipse([pt_e_r[0]-6, pt_e_r[1]-6, pt_e_r[0]+6, pt_e_r[1]+6], fill=self.palette["joint"])

        # 4. Hands with Per-Finger Articulated Bones
        finger_colors = [
            self.palette["finger_thumb"],
            self.palette["finger_index"],
            self.palette["finger_middle"],
            self.palette["finger_ring"],
            self.palette["finger_pinky"],
        ]

        for hand_idx, (h_raw, w_pose_idx) in enumerate([(lh_raw, 15), (rh_raw, 16)]):
            if not np.any(h_raw != 0) or not has_pose:
                continue

            w_center = pose_raw[w_pose_idx]
            pt_w = self._to_screen(w_center)
            hand_scale = self.scale * 0.40

            # Calculate 2D screen coordinates for all 21 hand joints
            joints = []
            for j_idx in range(21):
                # Landmarks are normalized: add to wrist anchor
                jx = pt_w[0] + float(h_raw[j_idx, 0]) * hand_scale
                jy = pt_w[1] + float(h_raw[j_idx, 1]) * hand_scale
                joints.append((jx, jy))

            # Palm solid plate
            palm_pts = [joints[0], joints[1], joints[5], joints[9], joints[13], joints[17]]
            draw.polygon(palm_pts, fill=self.palette["palm"])

            # Palm outline
            for a, b in HAND_PALM_CONNECTIONS:
                draw.line([joints[a], joints[b]], fill=self.palette["head_stroke"], width=4)

            # Per-finger bones with rounded thick strokes
            for f_i, f_conns in enumerate(HAND_FINGER_CONNECTIONS):
                f_color = finger_colors[f_i]
                for seg_a, seg_b in f_conns:
                    draw.line([joints[seg_a], joints[seg_b]], fill=f_color, width=5)
                    # Joint cap
                    draw.ellipse([joints[seg_b][0]-3, joints[seg_b][1]-3,
                                  joints[seg_b][0]+3, joints[seg_b][1]+3], fill=f_color)

        # 5. Caption / Gloss Banner overlay if provided
        if gloss_caption:
            draw.rectangle([16, self.height - 54, self.width - 16, self.height - 16],
                           fill=(30, 41, 59))
            caption_text = f"GLOSS: {gloss_caption}"
            if status_label:
                caption_text += f" ({status_label})"
            draw.text((32, self.height - 43), caption_text, fill=(255, 255, 255))

        return img

    def render_frame_hash(self, frame: np.ndarray) -> str:
        """Returns deterministic SHA-256 hash of rendered frame pixels."""
        img = self.render_frame(frame)
        return hashlib.sha256(img.tobytes()).hexdigest()

    def play(
        self,
        sequence: List[str],
        speed: float = 1.0,
        transition_frames: int = 8,
        rest_frames: int = 6
    ) -> Dict[str, Any]:
        """Compiles and renders a sequence of sign IDs.

        Returns playback timeline results with rendered frame hashes.
        """
        timeline = self.library.build_timeline(
            sequence=sequence,
            speed=speed,
            transition_frames=transition_frames,
            rest_frames=rest_frames
        )

        hashes = []
        for i in range(min(timeline.total_frames, 50)):  # sample frames
            f, meta = timeline.get_frame(i)
            hashes.append(self.render_frame_hash(f))

        return {
            "sequence": sequence,
            "total_frames": timeline.total_frames,
            "duration": timeline.duration,
            "gloss_strip": timeline.gloss_strip,
            "sample_hashes": hashes,
            "timeline": timeline
        }
