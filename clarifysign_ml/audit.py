"""Quality checks for a recognizer and its recorded landmark demonstrations.

The check deliberately uses the same preprocessing used by live recognition.
It does not make a model better; it prevents the product from claiming that an
unvalidated model is ready for a real-world camera demonstration.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .features import prepare_sequence


def audit_demo_library(recognizer: Any, signs_path: str | Path, policy: dict[str, Any]) -> dict[str, Any]:
    """Return reproducible agreement metrics for recorded sign demonstrations.

    ``signs.json`` is a UI animation library, so this is a smoke test rather
    than a substitute for an independent evaluation set.  It is nevertheless
    a valuable release gate: a recognizer should at least recognise most of
    the supplied demonstrations under its own preprocessing contract.
    """
    with Path(signs_path).open(encoding="utf-8") as handle:
        signs = json.load(handle)
    if not isinstance(signs, dict) or not signs:
        raise ValueError("sign library must be a non-empty JSON object")

    classes = list(recognizer.classes)
    class_index = {label: index for index, label in enumerate(classes)}
    results = []
    for label, sequence in signs.items():
        if label not in class_index:
            continue
        array = np.asarray(sequence, dtype=np.float32)
        probs = np.asarray(recognizer.probs(prepare_sequence(array)), dtype=np.float32)
        predicted = classes[int(np.argmax(probs))]
        results.append({
            "label": label,
            "predicted": predicted,
            "label_probability": round(float(probs[class_index[label]]), 6),
            "correct": predicted == label,
        })
    if not results:
        raise ValueError("no sign-library labels match recognizer classes")

    min_top1 = float(policy["min_demo_top1"])
    min_probability = float(policy["min_demo_label_probability"])
    top1 = sum(row["correct"] for row in results) / len(results)
    mean_probability = sum(row["label_probability"] for row in results) / len(results)
    failures = [row["label"] for row in results if not row["correct"] or row["label_probability"] < min_probability]
    return {
        "labels_checked": len(results),
        "top1": round(top1, 6),
        "mean_label_probability": round(mean_probability, 6),
        "policy": {"min_demo_top1": min_top1, "min_demo_label_probability": min_probability},
        "passing": top1 >= min_top1 and not failures,
        "failing_labels": failures,
    }
