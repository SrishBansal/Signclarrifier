"""Thin shim that re‑exports the core clarification API.

The original implementation lived here, but all logic has been moved to
``core.clarify``.  Keeping this shim preserves the public import path used by the
rest of the codebase (e.g. ``from app.clarify import Dialogue``) without pulling
in any web‑framework dependencies.
"""

from core.clarify import (
    Dialogue,
    candidates,
    uncertainty,
    H,
    calibrate_probabilities,
    compute_ece,
    plan,
    info_gain,
    update_posterior,
    resolve_dialogue,
    load_config,
)

__all__ = [
    "Dialogue",
    "candidates",
    "uncertainty",
    "H",
    "calibrate_probabilities",
    "compute_ece",
    "plan",
    "info_gain",
    "update_posterior",
    "resolve_dialogue",
    "load_config",
]
