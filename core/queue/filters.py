"""Quality and privacy filters for captured candidates.
The filter returns ``True`` if the example passes all checks.
"""

from typing import List, Dict, Any

from .config import MIN_CONFIDENCE, MIN_WINDOW_LENGTH, REQUIRE_BOTH_HANDS


def _average_confidence(landmarks: List[Dict[str, Any]]) -> float:
    """Compute average hand‑tracking confidence if present.
    Expected each frame dict to contain a ``confidence`` key (0‑1).
    """
    confidences = [frame.get("confidence", 0.0) for frame in landmarks]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def _has_both_hands(landmarks: List[Dict[str, Any]]) -> bool:
    """Check that at least one frame contains landmarks for both hands.
    We assume each frame may have ``hands`` list with entries ``"left"``/``"right"``.
    """
    for frame in landmarks:
        hands = {h.get("side") for h in frame.get("hands", [])}
        if {"left", "right"}.issubset(hands):
            return True
    return False


def quality_filter(record: Dict[str, Any]) -> bool:
    """Return ``True`` if the candidate passes quality & privacy thresholds.

    Checks performed:
    * average confidence >= ``MIN_CONFIDENCE``
    * sequence length >= ``MIN_WINDOW_LENGTH``
    * both hands present if ``REQUIRE_BOTH_HANDS`` is ``True``
    """
    landmarks = record.get("landmarks", [])
    if len(landmarks) < MIN_WINDOW_LENGTH:
        return False
    if _average_confidence(landmarks) < MIN_CONFIDENCE:
        return False
    if REQUIRE_BOTH_HANDS and not _has_both_hands(landmarks):
        return False
    return True

__all__ = ["quality_filter"]
