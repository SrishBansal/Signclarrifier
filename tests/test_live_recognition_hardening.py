import pytest
import numpy as np
from core.clarify import Dialogue
from clarifysign_ml.stream import StreamingSession
from clarifysign_ml.features import is_sign_candidate, FEATURE_DIM, SEQ_LEN, ACT_Y
from app.session import SignSession


class MockRecognizer:
    def __init__(self, top_class="hello", p1=0.95, top2_class="cold", p2=0.03):
        self.classes = [
            "hello", "thankyou", "good", "cold", "hot",
            "cheap", "expensive", "new", "smalllittle", "biglarge",
            "hat", "pant", "shirt", "shoes", "skirt",
            "suit", "tshirt", "black", "blue", "white",
            "card", "pen", "book", "soap", "ring",
            "i", "yesterday", "tight", "narrow", "alive"
        ]
        self.top_class = top_class
        self.p1 = p1
        self.top2_class = top2_class
        self.p2 = p2

    def probs(self, seq):
        p = np.full(len(self.classes), 0.001, dtype=np.float32)
        idx1 = self.classes.index(self.top_class)
        idx2 = self.classes.index(self.top2_class)
        p[idx1] = self.p1
        p[idx2] = self.p2
        p /= p.sum()
        return p


class MockExtractor:
    def __init__(self, emit_hands=True):
        self.emit_hands = emit_hands
        self.last = {"pose": True, "lh": emit_hands, "rh": False, "aspect": 1.77, "y_scale": 1.0}

    def __call__(self, frame):
        f = np.zeros(FEATURE_DIM, dtype=np.float32)
        # Pose with wrists at y=0.5 (above ACT_Y)
        f[126:] = 0.1
        f[172] = 0.5  # pose wrist
        f[175] = 0.5
        if self.emit_hands:
            f[:63] = 0.5  # left hand present
        return f


def test_is_sign_candidate_rejects_empty_and_no_hands():
    empty = np.zeros((48, FEATURE_DIM), dtype=np.float32)
    valid, reason = is_sign_candidate(empty)
    assert not valid
    assert "insufficient hand presence" in reason


def test_is_sign_candidate_accepts_valid_hand_gesture():
    seq = np.zeros((48, FEATURE_DIM), dtype=np.float32)
    # 20 frames with hands moving
    for i in range(20):
        seq[i, :63] = 0.2 + 0.01 * i
    valid, reason = is_sign_candidate(seq)
    assert valid
    assert reason.startswith("ok")


def test_session_discards_no_hand_segment_without_predicting():
    """frame_activity now requires hand presence.  Frames with no detected hands
    must never advance the stream to 'signing', and therefore must never emit
    resolved / clarify / resign events."""
    extractor = MockExtractor(emit_hands=False)
    rec = MockRecognizer(top_class="i", p1=0.65)
    sess = SignSession(rec, extractor)

    # Push many frames — even with raised wrists the stream must stay idle
    # because MockExtractor(emit_hands=False) produces zero LH and RH feature blocks.
    events = []
    for _ in range(sess.stream.start_n + 10):
        events.extend(sess.on_frame(b"frame"))

    # Stream must remain idle; no signing started
    assert sess.stream.state == "idle", f"Expected idle, got {sess.stream.state}"

    # No resolved, clarify, or resign messages should have been emitted
    assert not any(e["type"] == "resolved" for e in events)
    assert not any(e["type"] == "clarify" for e in events)
    assert not any(e["type"] == "resign" for e in events)


def test_low_confidence_prediction_leads_to_resign():
    # When top probability is below the Dialogue accept threshold, it should resign.
    # Spread probability evenly across 30 classes (~3.3% each) — no clear winner.
    n = 30
    probs = [1.0 / n] * n
    classes = [
        "hello", "thankyou", "good", "cold", "hot",
        "cheap", "expensive", "new", "smalllittle", "biglarge",
        "hat", "pant", "shirt", "shoes", "skirt",
        "suit", "tshirt", "black", "blue", "white",
        "card", "pen", "book", "soap", "ring",
        "i", "yesterday", "tight", "narrow", "alive"
    ]
    d = Dialogue(probs, classes)
    res = d.start()
    # Should resign — no candidate has enough probability to accept or ask about
    assert res["action"] == "resign"


def test_eig_pair_question_only_for_credible_alternatives():
    # Two credible candidates: shirt (0.45) vs tshirt (0.35)
    d = Dialogue([0.45, 0.35, 0.20], ["shirt", "tshirt", "red"])
    res = d.start()
    assert res["action"] == "ask"
    assert res["question"]["id"] == "pair"
    assert set(res["question"]["options"][0]["value"] for _ in [1])  # has options

    # Non-credible alternative: shirt (0.50) vs tiny noise (0.05)
    # The second candidate does not meet min_candidate_p (0.15), so no pair question
    d2 = Dialogue([0.50, 0.05] + [0.02] * 20, ["shirt", "tshirt"] + [f"w{i}" for i in range(20)])
    res2 = d2.start()
    # Should resign rather than asking a question about a 5% alternative
    assert res2["action"] == "resign"


def test_streaming_session_buffer_preserves_long_signs():
    # Buffer should not drop frames when sign is > 48 frames
    sess = StreamingSession(lambda x: np.zeros(10), fps=15, max_s=8.0)
    assert sess.buf.maxlen >= 120
