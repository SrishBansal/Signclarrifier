"""Train and gate a candidate recognizer without touching the installed model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .features import FEATURE_DIM, SEQ_LEN, prepare_sequence
from .predict import Recognizer, _load_checkpoint
from .research import ManifestSample, group_evaluation, load_manifest, manifest_fingerprint
from .train import ece, evaluate, fit, fit_temperature, logits_of, save_checkpoint
from sklearn.metrics import f1_score


def _load_feature(row: ManifestSample) -> np.ndarray:
    array = np.load(row.feature_path).astype(np.float32)
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM or not len(array):
        raise ValueError(f"{row.sample_id}: expected non-empty (*, {FEATURE_DIM}) features, got {array.shape}")
    return prepare_sequence(array)


def load_candidate_arrays(manifest_path: str | Path, required_classes: Iterable[str]):
    """Load a fully-labelled, leakage-checked manifest into train/val/test arrays."""
    rows = load_manifest(manifest_path)
    classes = list(required_classes)
    class_index = {label: index for index, label in enumerate(classes)}
    missing_label = [row.sample_id for row in rows if not row.label]
    unknown_label = sorted({row.label for row in rows if row.label and row.label not in class_index})
    if missing_label:
        raise ValueError(f"training manifest has rows without labels: {missing_label[:5]!r}")
    if unknown_label:
        raise ValueError(f"training manifest contains labels outside baseline vocabulary: {unknown_label!r}")

    packed: dict[str, list[tuple[np.ndarray, int, str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        packed[row.split].append((_load_feature(row), class_index[row.label], row.group_id))
    for split, values in packed.items():
        if not values:
            raise ValueError(f"training manifest has no {split} rows")
    result = {}
    for split, values in packed.items():
        result[split] = (
            np.stack([value[0] for value in values]),
            np.asarray([value[1] for value in values], dtype=int),
            [value[2] for value in values],
        )
    return result, rows


def promotion_decision(baseline: dict[str, Any], candidate: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Evaluate configured promotion rules. This never copies a model into models/."""
    checks = {
        "top1_gain": candidate["top1"] - baseline["top1"] >= float(policy["min_top1_gain"]),
        "macro_f1_gain": candidate["macro_f1"] - baseline["macro_f1"] >= float(policy["min_macro_f1_gain"]),
        "ece": candidate["ece"] <= baseline["ece"] + float(policy["max_ece_increase"]),
    }
    return {"eligible": all(checks.values()), "checks": checks,
            "baseline": baseline, "candidate": candidate}


def evaluate_probabilities(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Metrics for a checkpoint evaluated on the candidate's frozen test split."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if p.ndim != 2 or len(p) != len(y) or not len(y):
        raise ValueError("probabilities and labels must be non-empty and aligned")
    predicted = p.argmax(1)
    top3 = np.argsort(-p, axis=1)[:, :min(3, p.shape[1])]
    return {"n": int(len(y)), "top1": float((predicted == y).mean()),
            "top3": float(np.mean([y[index] in top3[index] for index in range(len(y))])),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "ece": ece(p, y)}


def train_candidate(manifest_path: str | Path, baseline_path: str | Path,
                    output_dir: str | Path,
                    policy: dict[str, Any], device: str = "cpu", epochs: int = 80,
                    patience: int = 12, seed: int = 0) -> dict[str, Any]:
    """Train a versioned candidate and return its report; baseline assets stay read-only."""
    checkpoint = _load_checkpoint(baseline_path, device)
    classes = list(checkpoint["classes"])
    arrays, rows = load_candidate_arrays(manifest_path, classes)
    Xtr, ytr, _ = arrays["train"]
    Xva, yva, _ = arrays["val"]
    Xte, yte, test_groups = arrays["test"]
    model = fit(Xtr, ytr, Xva, yva, len(classes), device=device, epochs=epochs,
                patience=patience, seed=seed)
    val_logits = logits_of(model, Xva, device)
    temperature = fit_temperature(val_logits, yva)
    test_logits = logits_of(model, Xte, device)
    metrics = evaluate(test_logits, yte, temperature)
    probs = torch.softmax(test_logits / temperature, 1).numpy()
    group_metrics = group_evaluation(probs, yte, test_groups)
    baseline_recognizer = Recognizer(baseline_path, device=device)
    baseline_probs = np.stack([baseline_recognizer.probs(sequence) for sequence in Xte])
    baseline_metrics = evaluate_probabilities(baseline_probs, yte)
    decision = promotion_decision(baseline_metrics, metrics, policy)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "clarifysign_bilstm_candidate.pt"
    meta = {"source_manifest_sha256": manifest_fingerprint(rows), "candidate": True,
            "baseline_path": str(baseline_path), "class_count": len(classes)}
    save_checkpoint(model_path, model, classes, temperature, meta)
    report = {"metrics": metrics, "baseline_same_split_metrics": baseline_metrics,
              "group_metrics": group_metrics, "promotion": decision,
              "manifest_fingerprint": meta["source_manifest_sha256"], "classes": classes}
    (target / "candidate_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {**report, "model_path": str(model_path), "report_path": str(target / "candidate_metrics.json")}
