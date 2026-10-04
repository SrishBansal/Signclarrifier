import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clarifysign_ml.audit import audit_demo_library


class DemoRecognizer:
    classes = ["hello", "thanks"]

    def probs(self, sequence):
        return np.array([0.95, 0.05], dtype=np.float32)


def test_audit_returns_policy_gate_and_failed_labels(tmp_path):
    library = tmp_path / "signs.json"
    library.write_text(json.dumps({"hello": [[0.0] * 225], "thanks": [[0.0] * 225]}))
    result = audit_demo_library(DemoRecognizer(), library, {
        "min_demo_top1": 0.9, "min_demo_label_probability": 0.5,
    })
    assert result["labels_checked"] == 2
    assert not result["passing"]
    assert result["failing_labels"] == ["thanks"]
