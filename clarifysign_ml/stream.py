"""Continuous webcam -> one prediction per sign. Transport-agnostic: push one (225,) feature per frame."""
from collections import deque
import numpy as np
from .features import frame_activity, prepare_sequence, ACT_Y, SEQ_LEN


class StreamingSession:
    """idle -> signing (hands up for start_s) -> end (hands down for end_s, or max_s) -> predict on the segment.

    The same trim + resample used in training is applied to the segment. A sign is committed ONCE, at its end;
    provisional probabilities during signing are for UI feedback only.
    predict_fn: (48,225) float32 -> probability vector.
    """

    def __init__(self, predict_fn, fps=15, act_y=ACT_Y, start_s=0.3, end_s=0.7, max_s=8.0, provisional_every=5):
        self.predict_fn, self.act_y, self.every = predict_fn, act_y, provisional_every
        self.start_s, self.end_s, self.max_s = start_s, end_s, max_s
        self.set_fps(fps)
        self.reset()

    def set_fps(self, fps):
        """Re-derive frame-count thresholds from the real incoming frame rate."""
        self.start_n, self.end_n = max(2, round(self.start_s * fps)), max(3, round(self.end_s * fps))
        self.max_n = max(10, int(self.max_s * fps))

    def reset(self):
        self.state, self.buf, self.act_run, self.idle_run, self.need_rest = "idle", deque(maxlen=SEQ_LEN), 0, 0, False
        self.pre = deque(maxlen=64)

    def push(self, feat):
        a = frame_activity(feat, self.act_y)
        ev = []
        if self.state == "idle":
            if self.need_rest:
                self.need_rest = a
                return ev
            self.pre.append(feat)
            while len(self.pre) > self.start_n + 4:
                self.pre.popleft()
            self.act_run = self.act_run + 1 if a else 0
            if self.act_run >= self.start_n:
                self.state, self.act_run, self.idle_run = "signing", 0, 0
                self.buf = deque(self.pre, maxlen=SEQ_LEN)
                self.pre.clear()
                ev.append({"type": "sign_start"})
            return ev
        self.buf.append(feat)  # deque will drop oldest when exceeding SEQ_LEN
        self.idle_run = 0 if a else self.idle_run + 1
        # Only emit provisional predictions when we have a full sequence of SEQ_LEN frames.
        if self.every and len(self.buf) % self.every == 0 and len(self.buf) >= SEQ_LEN:
            ev.append({"type": "provisional", "probs": self.predict_fn(prepare_sequence(np.stack(self.buf), self.act_y))})
        # Only emit sign_end (final prediction) when we have enough frames for a real sequence.
        if (self.idle_run >= self.end_n or len(self.buf) >= self.max_n) and len(self.buf) > 0:
            seq = np.stack(self.buf)
            ev.append({"type": "sign_end", "n_frames": len(seq),
                       "probs": self.predict_fn(prepare_sequence(seq, self.act_y))})
            forced = self.idle_run < self.end_n
            self.reset(); self.need_rest = forced and a
        return ev
