"""Tests for core/sign_library builder utilities (medoid, DTW aggregation)."""
import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.sign_library import (
    medoid_sequence,
    dtw_distance,
    dtw_average_sequence,
    build_library_from_dict,
)


def test_medoid_selection():
    np.random.seed(42)
    base = np.zeros((48, 225), dtype=np.float32)
    base[:, 126 + 15 * 3 + 1] = 0.5
    takes = [base + np.random.normal(0, 0.05, (48, 225)).astype(np.float32) for _ in range(5)]
    medoid = medoid_sequence(takes, target_len=48)
    assert medoid.shape == (48, 225)
    matches = [np.allclose(medoid, t, atol=1e-5) for t in takes]
    assert any(matches)


def test_dtw_distance_and_average():
    s1 = np.zeros((48, 225), dtype=np.float32)
    s2 = np.zeros((48, 225), dtype=np.float32)
    s1[:, 0] = 1.0
    s2[:, 0] = 1.0
    dist, path = dtw_distance(s1, s2)
    assert dist == pytest.approx(0.0, abs=1e-4)
    assert len(path) >= 48

    avg = dtw_average_sequence([s1, s2], target_len=48)
    assert avg.shape == (48, 225)


def test_build_library_from_dict():
    test_data = {
        "sign1": [np.zeros((48, 225), dtype=np.float32), np.ones((48, 225), dtype=np.float32)],
        "sign2": [np.zeros((30, 225), dtype=np.float32)]
    }
    lib_dict = build_library_from_dict(test_data, method="medoid", target_len=48)
    assert "sign1" in lib_dict
    assert "sign2" in lib_dict
    assert len(lib_dict["sign1"]) == 48
    assert len(lib_dict["sign2"]) == 48
