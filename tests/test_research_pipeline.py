import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clarifysign_ml.features import FEATURE_DIM, SEQ_LEN
from clarifysign_ml.research import (ManifestSample, manifest_fingerprint,
                                     queue_confirmed_features, validate_manifest,
                                     write_manifest, load_manifest,
                                     promote_reviewed_features, group_evaluation)


def _sample(sample_id, group_id, split="train"):
    return ManifestSample("research", sample_id, group_id, split, f"features/{sample_id}.npy")


def test_manifest_rejects_source_group_leakage():
    with pytest.raises(ValueError, match="crosses"):
        validate_manifest([_sample("one", "video-1", "train"), _sample("two", "video-1", "test")])


def test_manifest_round_trip_and_fingerprint(tmp_path):
    samples = [_sample("one", "video-1", "train"), _sample("two", "video-2", "test")]
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, samples)
    loaded = load_manifest(path)
    assert loaded == samples
    assert manifest_fingerprint(loaded) == manifest_fingerprint(samples)


def test_learning_queue_requires_consent_and_never_stores_raw_video(tmp_path):
    features = np.zeros((SEQ_LEN, FEATURE_DIM), np.float32)
    with pytest.raises(PermissionError):
        queue_confirmed_features(tmp_path, features, "shirt", consent=False)
    record = queue_confirmed_features(tmp_path, features, "shirt", consent=True)
    assert record["raw_video_stored"] is False
    assert (tmp_path / "pending.jsonl").exists()
    assert (tmp_path / "features" / f"{record['id']}.npy").exists()


def test_reviewer_promotion_checks_allowed_labels(tmp_path):
    record = queue_confirmed_features(tmp_path, np.zeros((SEQ_LEN, FEATURE_DIM)), "shirt", consent=True)
    manifest = tmp_path / "training.jsonl"
    promoted = promote_reviewed_features(tmp_path, [record["id"]], manifest, {"shirt"})
    assert load_manifest(manifest) == promoted
    with pytest.raises(ValueError, match="not an allowed"):
        promote_reviewed_features(tmp_path, [record["id"]], tmp_path / "bad.jsonl", {"pen"})


def test_group_evaluation_reports_each_signer_group():
    probs = np.array([[.9, .1], [.2, .8], [.7, .3]], dtype=np.float32)
    report = group_evaluation(probs, [0, 1, 1], ["signer-a", "signer-a", "signer-b"])
    assert report["overall"]["n"] == 3
    assert set(report["by_group"]) == {"signer-a", "signer-b"}
