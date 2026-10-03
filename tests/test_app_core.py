import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.semantics import CONCEPTS, REALIZATIONS, LANGS, EN, realize, match_text
from app.clarify import Dialogue, info_gain, plan, uncertainty, H
from app.session import SignSession
from clarifysign_ml.features import landmarks_to_features

CLASSES = sorted(CONCEPTS)


def test_every_language_realizes_every_concept_natively():
    for lang in LANGS:
        for c in CONCEPTS:
            w = realize(c, lang)
            assert w and w != EN[c] and not w.isascii(), (lang, c, w)
    try:
        realize("bank", "or"); assert False
    except KeyError:
        pass                                            # unsupported language is an error, never a fallback


def test_reverse_matching():
    assert match_text("hello bank", "hi") == ["hello", "bank"]
    assert match_text("नमस्ते", "hi") == ["hello"]
    assert match_text("I want blue shoes", "ta") == ["blue", "shoes"]
    assert match_text("big", "mr") == ["biglarge"] and match_text("xyz unknown", "hi") == []


def test_ig_matches_theory():
    cand = {"a": 0.5, "b": 0.5}
    ig = info_gain(cand, ["a", "b", "neither"], lambda c: c, 1.0)
    assert abs(ig - 1.0) < 1e-9                         # perfect answer to a 50/50 pair = exactly 1 bit
    assert info_gain(cand, ["x"], lambda c: "x", 0.95) == 0.0


def test_planner_picks_sensible_question():
    assert plan({"red": .5, "blue": .5}, [])["id"] == "pair"      # same kind: attribute question is useless
    assert plan({"red": .5, "shoes": .5}, [])["id"] == "kind"     # different kinds: cheap attribute question wins


def vec(**kw):
    p = np.full(len(CLASSES), 1e-3)
    for k, v in kw.items(): p[CLASSES.index(k)] = v
    return list(p / p.sum())


def test_dialogue_flows():
    assert Dialogue(vec(red=0.95), CLASSES).start()["action"] == "resolved"
    assert Dialogue([1 / 17] * 17, CLASSES).start()["action"] == "resign"
    d = Dialogue(vec(red=.5, blue=.4), CLASSES); a = d.start()
    assert a["action"] == "ask" and a["question"]["id"] == "pair"
    r = d.answer("red"); assert r["action"] == "resolved" and r["concept"] == "red" and r["how"] == "clarified"
    d2 = Dialogue(vec(red=.5, blue=.4), CLASSES); d2.start()
    try: d2.answer("not-an-option"); assert False
    except ValueError: pass


class LM:
    def __init__(self, x, y, z=0.0): self.x, self.y, self.z = x, y, z


def feat(raised):
    p = [LM(0.5, 0.9) for _ in range(33)]
    p[11], p[12] = LM(0.4, 0.4), LM(0.6, 0.4); p[15] = p[16] = LM(0.5, 0.5 if raised else 0.9)
    return landmarks_to_features([LM(0.5 + .01 * i, .5) for i in range(21)], None, p)


class Rec:
    classes = CLASSES
    def __init__(self, probs): self.p = np.array(probs)
    def probs(self, seq): assert seq.shape == (48, 225); return self.p


def run(session, seq):
    it = iter(seq); session.ext = lambda frame: next(it); out = []
    for _ in seq: out += session.on_frame(b"x")
    return out


SEQ = [feat(False)] * 25 + [feat(True)] * 25 + [feat(False)] * 25


def test_session_confident_sign_resolves_in_selected_language():
    s = SignSession(Rec(vec(red=0.95)), None, lang="te", decode=lambda b: b)
    ev = run(s, SEQ); res = [e for e in ev if e["type"] == "resolved"]
    assert len(res) == 1 and res[0]["concept"] == "red" and res[0]["text"] == realize("red", "te") and res[0]["tag"] == "te-IN"
    assert {"top1", "margin", "entropy"} <= set(res[0]["research"])


def test_session_clarifies_then_resolves_and_ignores_frames_while_waiting():
    s = SignSession(Rec(vec(red=.5, blue=.4)), None, lang="hi", decode=lambda b: b)
    ev = run(s, SEQ); q = [e for e in ev if e["type"] == "clarify"]
    assert len(q) == 1 and not [e for e in ev if e["type"] == "resolved"]
    assert run(s, SEQ) == []                           # dialogue open: more signing is ignored
    out = s.on_answer("red")
    assert out[0]["type"] == "resolved" and out[0]["text"] == realize("red", "hi")
    try: s.set_lang("or"); assert False
    except ValueError: pass


def test_replay_real_sign_signs_json():
    import json
    from clarifysign_ml.stream import StreamingSession
    signs_path = os.path.join(os.path.dirname(__file__), "..", "models", "signs.json")
    if not os.path.exists(signs_path):
        return
    with open(signs_path, "r") as f:
        signs = json.load(f)
    assert "hello" in signs
    hello_frames = [np.array(fr, dtype=np.float32) for fr in signs["hello"]]
    assert len(hello_frames) == 48

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "clarifysign_bilstm.pt")
    if os.path.exists(model_path):
        from clarifysign_ml.predict import Recognizer
        rec = Recognizer(model_path)
        probs_fn = rec.probs
    else:
        probs_fn = lambda x: np.array([0.95 if i == CLASSES.index("hello") else 0.05 / 16 for i in range(len(CLASSES))])

    s = StreamingSession(probs_fn, fps=15, provisional_every=0)
    idle_frame = feat(False)
    stream_frames = [idle_frame] * 15 + hello_frames + [idle_frame] * 20
    ends = [e for f in stream_frames for e in s.push(f) if e["type"] == "sign_end"]
    assert len(ends) == 1
    assert "probs" in ends[0]


def test_clarification_timeout():
    import time
    s = SignSession(Rec(vec(red=.5, blue=.4)), None, lang="hi", decode=lambda b: b, clarify_timeout=0.05)
    ev = run(s, SEQ)
    q = [e for e in ev if e["type"] == "clarify"]
    assert len(q) == 1
    time.sleep(0.08)
    timeout_ev = s.on_frame(feat(False))
    states = [e for e in timeout_ev if e.get("type") == "state" and e.get("state") == "listening"]
    resigns = [e for e in timeout_ev if e.get("type") == "resign"]
    assert len(states) == 1
    assert len(resigns) == 1
    assert "timed out" in resigns[0]["message"]


def test_websocket_with_dummy_extractor():
    import cv2
    from starlette.testclient import TestClient
    from app.server import create_app

    dummy_rec = Rec(vec(hello=0.95))
    dummy_feat = feat(True)
    should_fail = [False]

    class DummyExtractor:
        last = {"lh": True, "rh": False, "pose": True}
        def __call__(self, frame):
            if should_fail[0]:
                raise RuntimeError("Injected extraction failure")
            return dummy_feat
        def close(self):
            pass

    app = create_app(recognizer=dummy_rec, extractor_factory=lambda: DummyExtractor())
    client = TestClient(app)

    valid_jpeg = cv2.imencode('.jpg', np.zeros((64, 64, 3), np.uint8))[1].tobytes()

    with client.websocket_connect("/ws/sign") as ws:
        init_state = ws.receive_json()
        assert init_state == {"type": "state", "state": "listening"}

        # Send valid binary frame
        ws.send_bytes(valid_jpeg)
        received = []
        while True:
            msg = ws.receive_json()
            received.append(msg)
            if msg.get("type") == "ack":
                break
        types = [m.get("type") for m in received]
        assert "diag" in types
        assert "ack" in types

        # Send invalid JSON text message -> should receive error event and not crash
        ws.send_text("{bad_json")
        err_msg = ws.receive_json()
        assert err_msg.get("type") == "error"
        assert "Invalid message" in err_msg.get("message", "")

        # Send unknown message type
        ws.send_text('{"type": "unknown_action"}')
        err_msg2 = ws.receive_json()
        assert err_msg2.get("type") == "error"

        # Send reset message
        ws.send_text('{"type": "reset"}')
        reset_msg = ws.receive_json()
        assert reset_msg == {"type": "state", "state": "listening"}

        # Test error injection on frame processing
        should_fail[0] = True
        ws.send_bytes(valid_jpeg)
        frame_err_received = []
        while True:
            msg = ws.receive_json()
            frame_err_received.append(msg)
            if msg.get("type") == "ack":
                break
        err_types = [m.get("type") for m in frame_err_received]
        assert "error" in err_types
        assert "ack" in err_types



def test_hello_real_sign_resolves_or_clarifies():
    import json
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "clarifysign_bilstm.pt")
    signs_path = os.path.join(os.path.dirname(__file__), "..", "models", "signs.json")
    if not os.path.exists(model_path) or not os.path.exists(signs_path):
        return
    from clarifysign_ml.predict import Recognizer
    rec = Recognizer(model_path)
    with open(signs_path, "r") as f:
        signs = json.load(f)
    hello_frames = [np.array(fr, dtype=np.float32) for fr in signs["hello"]]
    idle_frame = feat(False)
    seq = [idle_frame] * 15 + hello_frames + [idle_frame] * 20

    s = SignSession(rec, None, lang="hi", decode=lambda b: b)
    ev = run(s, seq)
    action_events = [e for e in ev if e.get("type") in ("resolved", "clarify")]
    assert len(action_events) >= 1
    if action_events[0]["type"] == "resolved":
        assert action_events[0]["concept"] in CONCEPTS
    elif action_events[0]["type"] == "clarify":
        assert "question" in action_events[0]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)


