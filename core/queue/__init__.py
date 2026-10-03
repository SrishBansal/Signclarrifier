"""Queue package public API for candidate-example collection.
Provides functions to capture landmark sequences, filter them, and manage a
versioned dataset for offline fine‑tuning.
"""

from .capture import capture_candidate
from .filters import quality_filter
from .admin import review_pending, approve_candidate, reject_candidate, relabel_candidate
from .offline import fine_tune_and_evaluate
from .registry import ModelRegistry

__all__ = [
    "capture_candidate",
    "quality_filter",
    "review_pending",
    "approve_candidate",
    "reject_candidate",
    "relabel_candidate",
    "fine_tune_and_evaluate",
    "ModelRegistry",
]
