"""Sign Library for Indian Sign Language (ISL) Avatar Animations.

Data-driven keyframe library with:
- Canonical rest poses and ease-in-out LERP transitions
- Visible 'Unknown Sign' gesture and fingerspelling fallbacks (no silent skips)
- Medoid / DTW build aggregation tools
- Timeline compiler with synchronized gloss strips
"""

from .library import SignLibrary, SignTimeline, ease_in_out, lerp_frames
from .rest_pose import get_canonical_rest_frame, make_canonical_pose, make_canonical_hand
from .unknown import generate_unknown_sign_sequence, make_open_palm_up_hand
from .fingerspell import generate_letter_sequence
from .builder import medoid_sequence, dtw_distance, dtw_average_sequence, build_library_from_dir, build_library_from_dict

__all__ = [
    "SignLibrary",
    "SignTimeline",
    "ease_in_out",
    "lerp_frames",
    "get_canonical_rest_frame",
    "make_canonical_pose",
    "make_canonical_hand",
    "generate_unknown_sign_sequence",
    "make_open_palm_up_hand",
    "generate_letter_sequence",
    "medoid_sequence",
    "dtw_distance",
    "dtw_average_sequence",
    "build_library_from_dir",
    "build_library_from_dict",
]
