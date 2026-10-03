"""Capture module for candidate‑example queue.

The function `capture_candidate` is called by the front‑end when a user has
consented to store the raw landmark sequence together with the recogniser output.
All data is persisted as a JSON file in the pending directory.
"""

import os
import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .config import PENDING_DIR, MAX_PENDING_ITEMS

# Ensure pending directory exists
Path(PENDING_DIR).mkdir(parents=True, exist_ok=True)


def _salted_session_id() -> str:
    """Generate a pseudonymous session identifier.
    A random UUID is salted with a secret (hard‑coded for prototype) and hashed.
    The raw UUID never leaves the device.
    """
    secret = "clarify_queue_salt"
    raw = uuid.uuid4().hex
    return hashlib.sha256((raw + secret).encode()).hexdigest()


def _prune_pending() -> None:
    """Enforce the retention limit on pending items.
    Oldest files are removed until the count is below `MAX_PENDING_ITEMS`.
    """
    files = sorted(Path(PENDING_DIR).glob("*.json"), key=lambda p: p.stat().st_mtime)
    while len(files) > MAX_PENDING_ITEMS:
        oldest = files.pop(0)
        try:
            oldest.unlink()
        except OSError:
            pass


def capture_candidate(
    *,
    landmarks: List[Dict[str, Any]],
    top_k: List[Dict[str, Any]],
    clarification_outcome: str,
    model_version: str,
    consent: bool,
) -> str:
    """Persist a candidate example.

    Parameters
    ----------
    landmarks:
        Sequence of landmark dictionaries as produced by the hand‑tracking model.
    top_k:
        List of the recogniser's top‑k predictions (each dict contains ``label``
        and ``prob`` fields).
    clarification_outcome:
        One of ``"resolved"``, ``"rejected"`` or ``"skipped"``.
    model_version:
        Identifier of the model that produced the predictions.
    consent:
        Explicit user consent flag – the function raises ``PermissionError`` if
        ``False``.

    Returns
    -------
    str
        Path of the created JSON file (relative to project root).
    """
    if not consent:
        raise PermissionError("User has not given consent to store example data.")

    record: Dict[str, Any] = {
        "session_id": _salted_session_id(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_version": model_version,
        "landmarks": landmarks,
        "top_k": top_k,
        "outcome": clarification_outcome,
    }

    # Write to a uniquely named file
    filename = f"{uuid.uuid4().hex}.json"
    path = Path(PENDING_DIR) / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    # Enforce retention policy
    _prune_pending()

    return str(path)

__all__ = ["capture_candidate"]
