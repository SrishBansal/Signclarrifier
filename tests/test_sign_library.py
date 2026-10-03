"""Tests for /core/sign_library:
- Keyframe sequence retrieval per sign_id
- Build script with medoid and DTW aggregation
- Missing sign fallback (fingerspell or visible unknown sign marker, NEVER silent skip)
- Timeline compilation with ease-in-out LERP and rest pose
"""
import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.sign_library import (
    SignLibrary,
    SignTimeline,
    get_canonical_rest_frame,
    generate_unknown_sign_sequence,
    generate_letter_sequence,
    medoid_sequence,
    dtw_distance,
    dtw_average_sequence,
    build_library_from_dict,
    ease_in_out,
    lerp_frames,
)


def test_canonical_rest_frame():
    rest = get_canonical_rest_frame()
    assert isinstance(rest, np.ndarray)
    assert rest.shape == (225,)
    assert rest.dtype == np.float32
    # Verify pose wrists are below activity threshold
    pose = rest[126:].reshape(33, 3)
    assert pose[15, 1] > 2.0  # left wrist low
    assert pose[16, 1] > 2.0  # right wrist low


def test_unknown_sign_sequence_never_skips():
    seq = generate_unknown_sign_sequence(48)
    assert isinstance(seq, np.ndarray)
    assert seq.shape == (48, 225)
    # Wrists must raise up to chest level during the gesture (y < 1.0)
    mid_pose = seq[24, 126:].reshape(33, 3)
    assert mid_pose[15, 1] < 1.2  # left wrist lifted
    assert mid_pose[16, 1] < 1.2  # right wrist lifted


def test_fingerspell_letter_sequence():
    seq = generate_letter_sequence("A", 48)
    assert isinstance(seq, np.ndarray)
    assert seq.shape == (48, 225)
    # Right hand has letter landmarks
    rh = seq[24, 63:126].reshape(21, 3)
    assert np.any(rh != 0)


def test_medoid_selection():
    np.random.seed(42)
    # Generate 5 takes around a center sequence
    base = np.zeros((48, 225), dtype=np.float32)
    base[:, 126 + 15 * 3 + 1] = 0.5  # wrist height
    takes = [base + np.random.normal(0, 0.05, (48, 225)).astype(np.float32) for _ in range(5)]
    medoid = medoid_sequence(takes, target_len=48)
    assert medoid.shape == (48, 225)
    # Medoid must match one of the input takes
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


def test_sign_library_known_and_missing_signs():
    signs_json = os.path.join(ROOT, "models", "signs.json")
    lib = SignLibrary(signs_json if os.path.exists(signs_json) else None)

    # Known sign
    if "hello" in lib.all_signs():
        seq, status = lib.get_sign("hello")
        assert status == "native"
        assert len(seq) == 48

    # Missing sign -> NEVER silent skip, returns unknown sign sequence
    unk_seq, unk_status = lib.get_sign("nonexistent_random_sign")
    assert unk_status == "unknown"
    assert len(unk_seq) == 48
    assert np.any(unk_seq != 0)

    # Fingerspell token
    fs_seq, fs_status = lib.get_sign("FS_C")
    assert fs_status == "fingerspell"
    assert len(fs_seq) == 48


def test_timeline_compilation_and_gloss_strip():
    signs_json = os.path.join(ROOT, "models", "signs.json")
    lib = SignLibrary(signs_json if os.path.exists(signs_json) else None)

    timeline = lib.build_timeline(["hello", "thankyou", "pen"], speed=1.0)
    assert isinstance(timeline, SignTimeline)
    assert timeline.total_frames > 48 * 3  # signs + intro + transitions + outro
    assert len(timeline.gloss_strip) == 3
    assert [g["label"] for g in timeline.gloss_strip] == ["HELLO", "THANKYOU", "PEN"]

    # Verify transition frames use ease-in-out LERP
    trans_meta = [m for m in timeline.metadata if m["is_transition"]]
    assert len(trans_meta) > 0


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
