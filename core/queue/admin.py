"""Administrative review utilities for the candidate‑example queue.
The UI is intentionally minimal – functions can be called from a simple CLI or
integrated in a web admin panel later.
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from .config import PENDING_DIR, VERSIONS_DIR, MAX_DATASET_VERSIONS
from .filters import quality_filter


def _list_pending_files() -> List[Path]:
    return sorted(Path(PENDING_DIR).glob("*.json"))


def review_pending() -> List[Dict[str, Any]]:
    """Return all pending candidate records as Python dictionaries.
    Only records that pass ``quality_filter`` are included; others are marked as
    ``"rejected": true`` in the returned dict.
    """
    records = []
    for p in _list_pending_files():
        with p.open("r", encoding="utf-8") as f:
            rec = json.load(f)
        rec["_file"] = str(p)
        rec["_passes_filter"] = quality_filter(rec)
        records.append(rec)
    return records


def _next_version_dir() -> Path:
    """Determine the next version directory ``vN``.
    Existing versions are sorted lexicographically; we increment the highest N.
    """
    base = Path(VERSIONS_DIR)
    base.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("v")]
    if not existing:
        return base / "v1"
    nums = [int(name[1:]) for name in existing]
    return base / f"v{max(nums) + 1}"


def _write_manifest(version_path: Path, files: List[Path]) -> None:
    manifest = []
    for f in files:
        with f.open("rb") as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        manifest.append({"filename": f.name, "sha256": h})
    (version_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def approve_candidate(pending_path: str) -> str:
    """Approve a pending candidate.
    The candidate JSON file is moved into a new versioned dataset directory.
    Returns the path of the version directory the file was added to.
    """
    src = Path(pending_path)
    if not src.is_file():
        raise FileNotFoundError(f"Pending file not found: {pending_path}")
    # Load and verify it passes the filter – otherwise reject.
    with src.open("r", encoding="utf-8") as f:
        rec = json.load(f)
    if not quality_filter(rec):
        raise ValueError("Candidate does not meet quality criteria and cannot be approved.")
    # Determine version directory (create a new one if needed)
    version_dir = _next_version_dir()
    version_dir.mkdir(parents=True, exist_ok=True)
    dst = version_dir / src.name
    shutil.move(str(src), str(dst))
    # Update manifest for this version
    _write_manifest(version_dir, list(version_dir.glob("*.json")))
    # Enforce retention of only the last N versions
    all_versions = sorted(Path(VERSIONS_DIR).glob("v*"), key=lambda p: int(p.name[1:]))
    while len(all_versions) > MAX_DATASET_VERSIONS:
        old = all_versions.pop(0)
        shutil.rmtree(old)
    return str(version_dir)


def reject_candidate(pending_path: str) -> None:
    """Delete a pending candidate.
    """
    p = Path(pending_path)
    if p.is_file():
        p.unlink()


def relabel_candidate(pending_path: str, new_top_k: List[Dict[str, Any]]) -> None:
    """Replace the ``top_k`` field of a pending record with a new list.
    This is useful when an admin corrects the recogniser's predictions.
    """
    p = Path(pending_path)
    if not p.is_file():
        raise FileNotFoundError(pending_path)
    with p.open("r", encoding="utf-8") as f:
        rec = json.load(f)
    rec["top_k"] = new_top_k
    with p.open("w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

__all__ = [
    "review_pending",
    "approve_candidate",
    "reject_candidate",
    "relabel_candidate",
]
