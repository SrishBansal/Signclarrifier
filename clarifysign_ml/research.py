"""Reproducible research-data controls for ClarifySign.

This module intentionally does not download datasets or retain raw camera video.
It validates external-data manifests and manages an explicit-consent queue of
landmark sequences that can be reviewed before a periodic training run.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .features import FEATURE_DIM, SEQ_LEN


@dataclass(frozen=True)
class ManifestSample:
    """One external sample; group_id prevents source-video leakage."""
    dataset: str
    sample_id: str
    group_id: str
    split: str
    feature_path: str
    signer_id: str = ""
    text: str = ""


def validate_manifest(samples: Iterable[ManifestSample]) -> List[ManifestSample]:
    """Validate unique samples and ensure each source group appears in one split."""
    rows = list(samples)
    seen_samples = set()
    group_splits: Dict[tuple, str] = {}
    allowed_splits = {"train", "val", "test"}
    for row in rows:
        if not all((row.dataset, row.sample_id, row.group_id, row.feature_path)):
            raise ValueError("dataset, sample_id, group_id, and feature_path are required")
        if row.split not in allowed_splits:
            raise ValueError(f"invalid split {row.split!r}")
        sample_key = (row.dataset, row.sample_id)
        if sample_key in seen_samples:
            raise ValueError(f"duplicate sample {sample_key!r}")
        seen_samples.add(sample_key)
        group_key = (row.dataset, row.group_id)
        existing = group_splits.setdefault(group_key, row.split)
        if existing != row.split:
            raise ValueError(f"group {group_key!r} crosses {existing!r} and {row.split!r}")
    return rows


def load_manifest(path: str | Path) -> List[ManifestSample]:
    """Load a JSONL manifest after validating its video/signer group boundaries."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(ManifestSample(**json.loads(line)))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"bad manifest row {line_number}") from exc
    return validate_manifest(rows)


def write_manifest(path: str | Path, samples: Iterable[ManifestSample]) -> None:
    """Write a validated manifest; only metadata, never third-party data itself."""
    rows = validate_manifest(samples)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")


def manifest_fingerprint(samples: Iterable[ManifestSample]) -> str:
    """Stable dataset fingerprint for model cards and offline regression reports."""
    rows = validate_manifest(samples)
    payload = "\n".join(json.dumps(asdict(row), sort_keys=True, ensure_ascii=False) for row in rows)
    return sha256(payload.encode("utf-8")).hexdigest()


def queue_confirmed_features(root: str | Path, features: np.ndarray, label: str,
                             consent: bool, source: str = "local") -> Dict[str, Any]:
    """Queue an explicitly consented landmark sequence for later human review.

    The queue stores no raw camera video.  A reviewer must approve records before
    they are copied into a versioned training manifest.
    """
    if not consent:
        raise PermissionError("explicit consent is required before queueing a learning example")
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape != (SEQ_LEN, FEATURE_DIM):
        raise ValueError(f"expected features shaped {(SEQ_LEN, FEATURE_DIM)}, got {arr.shape}")
    if not str(label).strip():
        raise ValueError("label is required")
    queue_root = Path(root)
    feature_dir = queue_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256(arr.tobytes() + str(label).encode("utf-8")).hexdigest()
    feature_path = feature_dir / f"{digest}.npy"
    np.save(feature_path, arr)
    record = {"id": digest, "label": str(label), "source": str(source),
              "feature_path": str(feature_path), "raw_video_stored": False,
              "review_status": "pending"}
    with (queue_root / "pending.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record
