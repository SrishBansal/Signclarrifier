"""Per-connection logic, independent of FastAPI. Feed it JPEG frames and JSON messages; it returns JSON events."""
import time
import numpy as np
from clarifysign_ml.stream import StreamingSession
from clarifysign_ml.features import POSE, LH, RH, frame_activity
from .clarify import Dialogue, uncertainty
from .semantics import realize, LANGS, EN


class SignSession:
    def __init__(self, recognizer, extractor, lang="hi", decode=None, clarify_timeout=15.0):
        self.rec, self.ext, self.lang, self.decode = recognizer, extractor, lang, decode
        self.stream = StreamingSession(self._probs, fps=10, provisional_every=6)
        self.dialogue, self._t, self._fps, self._n = None, None, 10.0, 0
        self.dialogue_start_time = None
        self.clarify_timeout = clarify_timeout

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
            self.dialogue_start_time = None
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
        self.dialogue_start_time = None
        return [{"type": "resign", "message": "I couldn't tell. Please sign again.", "research": diag}]

    def on_frame(self, jpeg):
        now = time.time()
        if self._t is not None:
            inst = 1.0 / max(now - self._t, 1e-3)
            self._fps = 0.9 * self._fps + 0.1 * inst
            self._n += 1
            if self._n % 15 == 0:
                self.stream.set_fps(min(max(self._fps, 4.0), 20.0))
        self._t = now

        if self.dialogue is not None:  # waiting for an answer: check timeout
            if self.dialogue_start_time is not None and (now - self.dialogue_start_time) > self.clarify_timeout:
                self.dialogue = None
                self.dialogue_start_time = None
                return [
                    {"type": "state", "state": "listening"},
                    {"type": "resign", "message": "Clarification timed out after 15 seconds. Please sign again.",
                     "research": {"timeout": True, "fps": round(self._fps, 1)}}
                ]
            return []

        frame_input = self.decode(jpeg) if self.decode is not None else jpeg
        feat = self.ext(frame_input)

        p = feat[POSE].reshape(33, 3)
        has_pose = bool(p.any())
        wrist_y = float(min(p[15, 1], p[16, 1])) if has_pose else None
        lh_det = bool(feat[LH].any())
        rh_det = bool(feat[RH].any())
        act = frame_activity(feat, self.stream.act_y)

        diag_frame = {
            "fps": round(self._fps, 1),
            "lh": lh_det,
            "rh": rh_det,
            "pose": has_pose,
            "wrist_y": round(wrist_y, 3) if wrist_y is not None else None,
            "act_y_threshold": self.stream.act_y,
            "activity": act,
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
                    **uncertainty(list(probs)),
                    "frames": e["n_frames"],
                    "top": self._top(probs, 5),
                    "fps": round(self._fps, 1),
                    "wrist_y": round(wrist_y, 3) if wrist_y is not None else None,
                    "lh": lh_det,
                    "rh": rh_det,
                    "pose": has_pose,
                    "activity": act,
                    "state": "interpreting"
                }
                out.append({"type": "state", "state": "interpreting"})
                self.dialogue = Dialogue(list(map(float, probs)), self.rec.classes)
                act_res = self.dialogue.start()
                if act_res["action"] == "ask":
                    self.dialogue_start_time = now
                else:
                    self.dialogue_start_time = None
                out += self._events_from_action(act_res, diag)
        return out

    def on_answer(self, value):
        if self.dialogue is None:
            return []
        diag = {"top": [{"concept": c, "label": EN[c], "p": float(p)} for c, p in sorted(self.dialogue.cand.items(), key=lambda kv: -kv[1])]}
        act = self.dialogue.answer(value)
        if act["action"] == "ask":
            self.dialogue_start_time = time.time()
        else:
            self.dialogue = None
            self.dialogue_start_time = None
        return self._events_from_action(act, diag)

    def reset(self):
        self.dialogue = None
        self.dialogue_start_time = None
        self.stream.reset()
