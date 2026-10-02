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


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
