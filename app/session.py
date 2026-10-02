"""Per-connection logic, independent of FastAPI. Feed it JPEG frames and JSON messages; it returns JSON events."""
import time
import numpy as np
from clarifysign_ml.stream import StreamingSession
from .clarify import Dialogue, uncertainty
from .semantics import realize, LANGS, EN


class SignSession:
    def __init__(self, recognizer, extractor, lang="hi", decode=None):
        self.rec, self.ext, self.lang, self.decode = recognizer, extractor, lang, decode
        self.stream = StreamingSession(self._probs, fps=10, provisional_every=6)
        self.dialogue, self._t, self._fps, self._n = None, None, 10.0, 0

    def _probs(self, seq48):
        return self.rec.probs(seq48)

    def set_lang(self, lang):
        if lang not in LANGS:
            raise ValueError(f"unsupported language {lang}")
        self.lang = lang

    def _top(self, probs, n=3):
        c = self.rec.classes
        return [{"concept": c[i], "label": EN[c[i]], "p": float(probs[i])} for i in np.argsort(-probs)[:n]]

    def _resolved(self, concept, how, diag):
        return {"type": "resolved", "concept": concept, "how": how, "english": EN[concept],
                "text": realize(concept, self.lang), "lang": self.lang, "tag": LANGS[self.lang][1], "research": diag}

    def _events_from_action(self, act, diag):
        if act["action"] == "resolved":
            self.dialogue = None
            return [self._resolved(act["concept"], act["how"], diag)]
        if act["action"] == "ask":
            labels = []
            for o in act["question"]["options"]:
                o = dict(o)
                if o["value"] in EN and act["question"]["id"] == "pair":
                    o["text_lang"] = realize(o["value"], self.lang)
                labels.append(o)
            act["question"]["options"] = labels
            return [{"type": "clarify", "question": act["question"], "research": {**diag, "ig": act["ig"], "utility": act["utility"]}}]
        self.dialogue = None
        return [{"type": "resign", "message": "I couldn't tell. Please sign again.", "research": diag}]

    def on_frame(self, jpeg):
        now = time.time()
        if self._t is not None:
            inst = 1.0 / max(now - self._t, 1e-3)
            self._fps = 0.9 * self._fps + 0.1 * inst; self._n += 1
            if self._n % 15 == 0:
                self.stream.set_fps(min(max(self._fps, 4.0), 20.0))
        self._t = now
        if self.dialogue is not None:  # waiting for an answer: ignore signing
            return []
        feat = self.ext(self.decode(jpeg))
        out = []
        for e in self.stream.push(feat):
            if e["type"] == "sign_start":
                out.append({"type": "state", "state": "signing"})
            elif e["type"] == "provisional":
                out.append({"type": "provisional", "top": self._top(e["probs"])})
            elif e["type"] == "sign_end":
                probs = e["probs"]
                diag = {**uncertainty(list(probs)), "frames": e["n_frames"], "top": self._top(probs, 5), "fps": round(self._fps, 1)}
                out.append({"type": "state", "state": "interpreting"})
                self.dialogue = Dialogue(list(map(float, probs)), self.rec.classes)
                out += self._events_from_action(self.dialogue.start(), diag)
        return out

    def on_answer(self, value):
        if self.dialogue is None:
            return []
        diag = {"top": [{"concept": c, "label": EN[c], "p": float(p)} for c, p in sorted(self.dialogue.cand.items(), key=lambda kv: -kv[1])]}
        return self._events_from_action(self.dialogue.answer(value), diag)

    def reset(self):
        self.dialogue = None; self.stream.reset()
