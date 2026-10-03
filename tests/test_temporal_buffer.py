"""Temporal buffer / rolling window tests for StreamingSession."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from clarifysign_ml.stream import StreamingSession
from clarifysign_ml.features import ACT_Y, SEQ_LEN

class DummyRecognizer:
    def __init__(self):
        self.last_seq = None
    def probs(self, seq48):
        self.last_seq = np.asarray(seq48)
        return np.zeros(10)


def _active_feat(i=0):
    """Feature whose wrist_y < ACT_Y (=1.3) so frame_activity returns True.
    POSE = slice(126,225). lm15.y = POSE.start + 15*3+1 = 126+46 = 172
                            lm16.y = POSE.start + 16*3+1 = 126+49 = 175
    """
    f = np.zeros(225, dtype=np.float32)
    f[172] = 0.5   # left wrist Y < ACT_Y(1.3) → active
    f[175] = 0.5   # right wrist Y
    f[0] = float(i)  # encode frame index for distinctness
    return f


def _idle_feat():
    """Feature whose wrist_y >= ACT_Y so frame_activity returns False."""
    f = np.zeros(225, dtype=np.float32)
    f[172] = 2.0; f[175] = 2.0  # wrists below threshold (y ≥ ACT_Y)
    return f


def _pump_to_signing(sess, start_n_extra=0):
    """Push enough active frames to transition idle → signing."""
    # StreamingSession.start_n frames needed; default fps=15, start_s=0.3 → 4-5 frames
    for _ in range(sess.start_n + start_n_extra):
        sess.push(_active_feat())


def test_no_inference_before_48_frames():
    """sign_end must not fire while we have fewer than SEQ_LEN frames in the buffer."""
    rec = DummyRecognizer()
    sess = StreamingSession(rec.probs, fps=15, act_y=ACT_Y, provisional_every=5)
    _pump_to_signing(sess)
    # Now in signing state – push SEQ_LEN - 1 more active frames
    all_events = []
    for i in range(SEQ_LEN - 1):
        evs = sess.push(_active_feat(i))
        all_events.extend(evs)
    assert not any(e["type"] == "sign_end" for e in all_events), \
        "sign_end fired before SEQ_LEN frames"


def test_inference_triggered_by_idle_frames():
    """sign_end fires when end_n idle frames seen after signing starts."""
    rec = DummyRecognizer()
    sess = StreamingSession(rec.probs, fps=15, act_y=ACT_Y, provisional_every=5)
    _pump_to_signing(sess)
    all_events = []
    # Push enough active frames to fill the buffer
    for i in range(SEQ_LEN + 2):
        evs = sess.push(_active_feat(i))
        all_events.extend(evs)
    # Now end with idle frames to trigger sign_end
    for _ in range(sess.end_n + 1):
        evs = sess.push(_idle_feat())
        all_events.extend(evs)
    assert any(e["type"] == "sign_end" for e in all_events), \
        "sign_end never fired after idle termination"
    assert rec.last_seq is not None
    assert rec.last_seq.shape == (SEQ_LEN, 225)


def test_rolling_buffer_is_real_frames():
    """Buffer should contain distinct real frames (not tiled copies)."""
    rec = DummyRecognizer()
    sess = StreamingSession(rec.probs, fps=15, act_y=ACT_Y, provisional_every=5)
    _pump_to_signing(sess)
    # Push distinct active frames then idle to end
    for i in range(SEQ_LEN + 5):
        sess.push(_active_feat(i))
    # Now idle to trigger sign_end
    for _ in range(sess.end_n + 1):
        sess.push(_idle_feat())
    # After sign_end, the committed sequence must have distinct rows
    if rec.last_seq is not None:
        assert rec.last_seq.shape[1] == 225
        # Consecutive rows should not all be identical (no np.tile)
        n_unique = len({tuple(row.tolist()[:3]) for row in rec.last_seq})
        assert n_unique > 1, "All rows identical – looks like np.tile was used"


def test_all_zero_frame_skipped_in_idle():
    """An all-zero frame pushed in idle state should not advance the buffer."""
    rec = DummyRecognizer()
    sess = StreamingSession(rec.probs, fps=15, act_y=ACT_Y, provisional_every=5)
    ev = sess.push(np.zeros(225, dtype=np.float32))
    # all-zero → no activity → stays idle → no sign_start
    assert not any(e["type"] == "sign_start" for e in ev)
    assert sess.state == "idle"
