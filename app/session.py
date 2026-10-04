import time
import numpy as np
from clarifysign_ml.stream import StreamingSession

# Direct key mapping for the frontend WebSocket JSON
POSE = "pose"
LH = "lh"
RH = "rh"

class SignSession:
    def __init__(self, recognizer, config=None):
        self.rec = recognizer
        self.stream = StreamingSession(recognizer, config or {})
        self._fps = 15.0
        self._t = None
        self._n = 0

    def _top(self, probs, k=3):
        idx = np.argsort(-np.array(probs))[:k]
        return [{"sign": self.rec.classes[i], "score": round(float(probs[i]), 3)} for i in idx]

    def on_frame(self, feat_dict: dict):
        now = time.time()
        if self._t is not None:
            inst = 1.0 / max(now - self._t, 1e-3)
            self._fps = 0.9 * self._fps + 0.1 * inst
            self._n += 1
            if self._n % 15 == 0:
                self.stream.set_fps(min(max(self._fps, 4.0), 20.0))
        self._t = now

        feat = {
            POSE: np.array(feat_dict.get("pose", [0]*99), dtype=np.float32),
            LH: np.array(feat_dict.get("lh", [0]*63), dtype=np.float32),
            RH: np.array(feat_dict.get("rh", [0]*63), dtype=np.float32)
        }
        
        p = feat[POSE].reshape(33, 3)
        has_pose = bool(p.any())
        wrist_y = float(min(p[15, 1], p[16, 1])) if has_pose else None

        diag_frame = {
            "fps": round(self._fps, 1),
            "lh": bool(feat[LH].any()), "rh": bool(feat[RH].any()), "pose": has_pose,
            "wrist_y": round(wrist_y, 3) if wrist_y is not None else None,
            "act_y_threshold": self.stream.act_y,
            "activity": has_pose, 
            "state": self.stream.state
        }

        out = [{"type": "diag", "research": diag_frame}]

        for e in self.stream.push(feat):
            if e["type"] == "sign_start":
                diag_frame["state"] = "signing"
                out.append({"type": "state", "state": "signing"})
            elif e["type"] == "provisional":
                out.append({"type": "provisional", "top": self._top(e["probs"])})
            elif e["type"] == "sign_end":
                probs = e["probs"]
                diag = {
                    "frames": e["n_frames"], 
                    "top": self._top(probs, 5), 
                    "fps": round(self._fps, 1), 
                    "state": "interpreting"
                }
                out.append({"type": "state", "state": "interpreting"})
                
                # Dynamically commit the actual model's prediction
                idx = int(np.argmax(probs))
                out.append({
                    "type": "sign_committed", 
                    "sign": self.rec.classes[idx], 
                    "confidence": round(float(probs[idx]), 3), 
                    "diag": diag
                })
        return out