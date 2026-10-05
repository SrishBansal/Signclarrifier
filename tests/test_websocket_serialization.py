import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.server import _json_native


def test_json_native_converts_nested_numpy_payload_values():
    event = {
        "type": "diag",
        "research": {
            "fps": np.float32(23.9),
            "frames": np.int64(48),
            "pose": np.bool_(True),
            "top": np.array([np.float32(0.38), np.float32(0.18)]),
        },
    }
    payload = _json_native(event)
    assert payload == {"type": "diag", "research": {"fps": 23.899999618530273,
                                                         "frames": 48, "pose": True,
                                                         "top": [0.3799999952316284, 0.18000000715255737]}}
    json.dumps(payload, allow_nan=False)
