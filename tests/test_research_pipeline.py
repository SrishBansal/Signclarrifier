import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clarifysign_ml.features import FEATURE_DIM, SEQ_LEN
from clarifysign_ml.research import (ManifestSample, manifest_fingerprint,
                                     queue_confirmed_features, validate_manifest,
                                     write_manifest, load_manifest)


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
