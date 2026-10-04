"""Data-driven SignLibrary module.

Manages sign keyframe sequences (pose + both hands), eases transitions
with ease-in-out LERP, coordinates rest poses, and guarantees that missing
signs fallback to fingerspelling or a visible 'unknown sign' marker, NEVER a silent skip.
"""
import os
import json
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

from .rest_pose import get_canonical_rest_frame
from .unknown import generate_unknown_sign_sequence
from .fingerspell import generate_letter_sequence


def ease_in_out(t: float) -> float:
    """Smooth cubic hermite ease-in-out curve (smoothstep): 3t^2 - 2t^3."""
    t = max(0.0, min(1.0, t))
    return float(t * t * (3.0 - 2.0 * t))


def lerp_frames(f1: np.ndarray, f2: np.ndarray, alpha: float) -> np.ndarray:
    """Linearly interpolate between two 225-dim landmark frames with easing."""
    return (1.0 - alpha) * f1 + alpha * f2


class SignTimeline:
    """A compiled multi-sign playback timeline with transition easing and per-sign synchronization."""

    def __init__(
        self,
        frames: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        gloss_strip: List[Dict[str, Any]],
        fps: float = 25.0
    ):
        self.frames = frames
        self.metadata = metadata
        self.gloss_strip = gloss_strip
        self.fps = fps
        self.total_frames = len(frames)
        self.duration = self.total_frames / fps if fps > 0 else 0.0

    def get_frame(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        idx = max(0, min(self.total_frames - 1, index))
        return self.frames[idx], self.metadata[idx]


class SignLibrary:
    """Data-driven library of ISL signs with robust fallbacks and timeline compilation."""

    def __init__(self, source: Optional[Union[str, Dict[str, Any]]] = None):
        self._signs: Dict[str, np.ndarray] = {}
        self._rest_frame = get_canonical_rest_frame()

        if source is not None:
            self.load(source)
        else:
            # Default auto-discovery of models/signs.json
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "models", "signs.json"
            )
            if os.path.exists(default_path):
                self.load(default_path)

    def load(self, source: Union[str, Dict[str, Any]]) -> None:
        """Loads sign dictionary from JSON path or in-memory dict."""
        if isinstance(source, str):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = source

        self._signs = {
            k.lower(): np.array(v, dtype=np.float32) for k, v in data.items()
        }

    def has_sign(self, sign_id: str) -> bool:
        """Check if sign is available in the library."""
        return sign_id.lower() in self._signs

    def all_signs(self) -> List[str]:
        """Returns sorted list of all available sign IDs."""
        return sorted(self._signs.keys())

    def get_sign(self, sign_id: str) -> Tuple[np.ndarray, str]:
        """Retrieves 48-frame sequence for sign_id.

        If missing:
        - If fingerspell token (e.g. 'FS_A' or 'FS_WORD'): generates fingerspelling sequence.
        - Otherwise: returns visible UNKNOWN_SIGN marker sequence.
        NEVER silently skips.

        Returns (sequence_array, status) where status is 'native', 'fingerspell', or 'unknown'.
        """
        key = sign_id.strip().lower()
        if key in self._signs:
            return self._signs[key].copy(), "native"

        # Check for fingerspelling prefix
        if key.startswith("fs_") or key.startswith("fs-"):
            letters = key[3:].upper()
            if len(letters) == 1:
                return generate_letter_sequence(letters), "fingerspell"
            elif len(letters) > 1:
                # Multi-letter word fingerspell
                seqs = [generate_letter_sequence(ch, n_frames=24) for ch in letters]
                combined = np.concatenate(seqs, axis=0)
                return combined, "fingerspell"

        # Visible unknown sign marker fallback (NEVER a silent skip)
        return generate_unknown_sign_sequence(48), "unknown"

    def rest_pose(self) -> np.ndarray:
        """Returns the 225-dim canonical rest frame."""
        return self._rest_frame.copy()

    def build_timeline(
        self,
        sequence: List[str],
        speed: float = 1.0,
        transition_frames: int = 8,
        rest_frames: int = 6,
        fps: float = 25.0
    ) -> SignTimeline:
        """Compiles an ordered list of sign IDs into a continuous animated timeline.

        Features:
        - Natural rest pose at intro and outro.
        - Ease-in-out LERP between consecutive signs and rest poses.
        - Per-frame metadata with active sign, progress, and gloss highlighting.
        - Configurable speed multiplier.
        """
        if not sequence:
            # Just rest pose
            frames = [self._rest_frame.copy()] * max(rest_frames, 10)
            meta = [{
                "sign_id": "REST",
                "sign_index": -1,
                "is_transition": False,
                "progress": 0.0,
                "gloss": "Rest",
                "status": "rest"
            } for _ in range(len(frames))]
            return SignTimeline(frames, meta, [], fps)

        compiled_frames: List[np.ndarray] = []
        compiled_meta: List[Dict[str, Any]] = []
        gloss_strip: List[Dict[str, Any]] = []

        # 1. Intro: rest frames
        for _ in range(rest_frames):
            compiled_frames.append(self._rest_frame.copy())
            compiled_meta.append({
                "sign_id": "REST",
                "sign_index": -1,
                "is_transition": False,
                "progress": 0.0,
                "gloss": "Ready",
                "status": "rest"
            })

        prev_last_frame = self._rest_frame.copy()

        # 2. Sequential signs with ease-in-out LERP transitions
        for seq_idx, sign_id in enumerate(sequence):
            sign_seq, status = self.get_sign(sign_id)
            n_sign_frames = len(sign_seq)

            # Adjust sign frames by speed
            if speed != 1.0 and speed > 0:
                target_len = max(8, int(round(n_sign_frames / speed)))
                from .builder import resample_sequence
                sign_seq = resample_sequence(sign_seq, target_len)
                n_sign_frames = len(sign_seq)

            # Transition from previous pose into first frame of this sign
            trans_len = max(2, int(round(transition_frames / speed)))
            first_frame = sign_seq[0]
            for t_step in range(trans_len):
                alpha = ease_in_out(t_step / float(trans_len))
                trans_frame = lerp_frames(prev_last_frame, first_frame, alpha)
                compiled_frames.append(trans_frame)
                compiled_meta.append({
                    "sign_id": sign_id,
                    "sign_index": seq_idx,
                    "is_transition": True,
                    "progress": 0.0,
                    "gloss": f"→ {sign_id.upper()}",
                    "status": status
                })

            start_frame_idx = len(compiled_frames)

            # Add actual sign frames
            for f_idx in range(n_sign_frames):
                progress = f_idx / float(max(1, n_sign_frames - 1))
                compiled_frames.append(sign_seq[f_idx])
                compiled_meta.append({
                    "sign_id": sign_id,
                    "sign_index": seq_idx,
                    "is_transition": False,
                    "progress": round(progress, 3),
                    "gloss": sign_id.upper(),
                    "status": status
                })

            end_frame_idx = len(compiled_frames) - 1
            gloss_strip.append({
                "sign_id": sign_id,
                "label": sign_id.upper(),
                "index": seq_idx,
                "start_frame": start_frame_idx,
                "end_frame": end_frame_idx,
                "status": status
            })

            prev_last_frame = sign_seq[-1]

        # 3. Outro transition to rest pose
        outro_len = max(2, int(round(transition_frames / speed)))
        for t_step in range(outro_len):
            alpha = ease_in_out(t_step / float(outro_len))
            trans_frame = lerp_frames(prev_last_frame, self._rest_frame, alpha)
            compiled_frames.append(trans_frame)
            compiled_meta.append({
                "sign_id": "REST",
                "sign_index": -1,
                "is_transition": True,
                "progress": 1.0,
                "gloss": "→ Rest",
                "status": "rest"
            })

        for _ in range(rest_frames):
            compiled_frames.append(self._rest_frame.copy())
            compiled_meta.append({
                "sign_id": "REST",
                "sign_index": -1,
                "is_transition": False,
                "progress": 1.0,
                "gloss": "Rest",
                "status": "rest"
            })

        return SignTimeline(compiled_frames, compiled_meta, gloss_strip, fps=fps)

    def save(self, path: str) -> None:
        """Saves current sign library to JSON."""
        data = {k: np.round(v, 4).tolist() for k, v in self._signs.items()}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
