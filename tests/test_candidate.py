import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clarifysign_ml.candidate import evaluate_probabilities, load_candidate_arrays, promotion_decision
from clarifysign_ml.research import ManifestSample, write_manifest


def test_candidate_arrays_require_labels_and_keep_split_shapes(tmp_path):
    feature = tmp_path / "example.npy"
    np.save(feature, np.zeros((12, 225), dtype=np.float32))
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        ManifestSample("source", "train", "video-train", "train", str(feature), label="hello"),
        ManifestSample("source", "val", "video-val", "val", str(feature), label="hello"),
        ManifestSample("source", "test", "video-test", "test", str(feature), label="hello"),
    ]
    write_manifest(manifest, rows)
    arrays, _ = load_candidate_arrays(manifest, ["hello"])
    assert arrays["train"][0].shape == (1, 48, 225)
    assert arrays["test"][2] == ["video-test"]


def test_candidate_arrays_reject_missing_label(tmp_path):
    feature = tmp_path / "example.npy"
    np.save(feature, np.zeros((12, 225), dtype=np.float32))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [ManifestSample("source", "x", "video", "train", str(feature))])
    with pytest.raises(ValueError, match="without labels"):
        load_candidate_arrays(manifest, ["hello"])


def test_candidate_promotion_requires_every_configured_improvement():
    baseline = {"top1": 0.4, "macro_f1": 0.3, "ece": 0.1}
    candidate = {"top1": 0.44, "macro_f1": 0.34, "ece": 0.1}
    decision = promotion_decision(baseline, candidate, {
        "min_top1_gain": 0.03, "min_macro_f1_gain": 0.03, "max_ece_increase": 0,
    })
    assert decision["eligible"]


def test_probability_evaluation_uses_the_provided_test_split():
    result = evaluate_probabilities(np.array([[0.8, 0.2], [0.1, 0.9]]), np.array([0, 1]))
    assert result["n"] == 2
    assert result["top1"] == 1.0
