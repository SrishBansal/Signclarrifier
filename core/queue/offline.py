"""Offline fine‑tuning and evaluation utilities.
The prototype does **not** perform real training; instead it simulates the
workflow by generating dummy metrics. The API mirrors what a production script
would expose.
"""

import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List

from .config import VERSIONS_DIR, REGISTRY_FILE, REGRESSION_TOLERANCE
from .registry import ModelRegistry


def _load_latest_version() -> Path:
    """Return the Path to the newest version directory (vN)."""
    base = Path(VERSIONS_DIR)
    versions = sorted(base.glob("v*"), key=lambda p: int(p.name[1:]))
    if not versions:
        raise FileNotFoundError("No approved dataset versions found.")
    return versions[-1]


def _mock_metrics() -> Dict[str, float]:
    """Generate dummy metrics for illustration.
    Returns a dict with keys ``accuracy``, ``regression_error``.
    """
    return {
        "accuracy": round(random.uniform(0.80, 0.95), 4),
        "regression_error": round(random.uniform(0.01, 0.05), 4),
    }


def fine_tune_and_evaluate() -> Dict[str, Any]:
    """Run the offline fine‑tuning pipeline on the latest approved dataset.

    Steps performed (simulated):
    1. Load the most recent version directory.
    2. Register a new model version in ``ModelRegistry``.
    3. Generate *before* metrics using a placeholder previous model.
    4. Simulate fine‑tuning and produce *after* metrics.
    5. Compare the regression error; if it exceeds ``REGRESSION_TOLERANCE`` the
       new model is **not** promoted.
    """
    # Load dataset (we only need the path for the report)
    latest_dir = _load_latest_version()

    registry = ModelRegistry()
    new_version = f"model_{len(registry.entries) + 1}"

    # Simulate previous model metrics (could be fetched from registry, but we use dummy)
    before = _mock_metrics()

    # Simulate fine‑tuning – in reality you would invoke a training script here.
    after = _mock_metrics()

    regression_increase = after["regression_error"] - before["regression_error"]
    promote = regression_increase <= REGRESSION_TOLERANCE

    # Record the new model in the registry
    registry.register(
        version=new_version,
        path=str(latest_dir),
        metrics=after,
        promoted=promote,
    )

    report: Dict[str, Any] = {
        "dataset_version": latest_dir.name,
        "new_model_version": new_version,
        "metrics_before": before,
        "metrics_after": after,
        "regression_change": regression_increase,
        "promotion_allowed": promote,
    }
    return report

__all__ = ["fine_tune_and_evaluate"]
