"""Per-connection logic. Imports from core only; no app.semantics / app.clarify."""
import time
import numpy as np
from clarifysign_ml.stream import StreamingSession
from clarifysign_ml.features import POSE, LH, RH, frame_activity
from core.clarify import Dialogue, uncertainty
from core.ontology import get_ontology
from core.dialogue import DialogueManager
from core.semantics import NaturalLanguageRealizer

# language code -> (name, BCP-47 tag)
_LANG_MAP = {"en": ("English", "en-IN"), "hi": ("Hindi", "hi-IN"), "ta": ("Tamil", "ta-IN")}


class SignSession:
    def __init__(self, recognizer, extractor, dm: DialogueManager = None, lang="en",
                 decode=None, clarify_timeout=15.0):
        self.rec = recognizer
        self.ext = extractor
        self.ont = get_ontology()
        self.dm = dm or DialogueManager(self.ont)
        self.lang = lang
        self.decode = decode
        self.stream = StreamingSession(self._probs, fps=10, provisional_every=6)
        self.dialogue = None
        self._t = None
        self._fps = 10.0
        self._n = 0
        self.dialogue_start_time = None
        self.clarify_timeout = clarify_timeout

    def _probs(self, seq48):
        return self.rec.probs(seq48)

    def set_lang(self, lang):
        if lang not in _LANG_MAP:
            raise ValueError(f"unsupported language {lang!r}")
        self.lang = lang

    def _label(self, concept_id):
        try: return self.ont.label(concept_id, self.lang)
        except KeyError: return concept_id.lower()

    def _top(self, probs, n=3):
        c = self.rec.classes
        return [{"concept": c[i], "label": self._label(c[i].upper()), "p": float(probs[i])}
                for i in np.argsort(-probs)[:n]]

    def _resolved_event(self, concept_id, how, diag):
        """Build Direction B resolved event from concept_id (a sign_id / INCLUDE class)."""
        ont_concept = self.ont.concept_for_sign(concept_id)
        if ont_concept is None:
            # sign_id not in ontology — should be blocked at boot, but guard anyway
            ont_concept = concept_id.upper()
        real = NaturalLanguageRealizer.realize(self.dm.state, _LANG_MAP[self.lang][0], self.ont)
        _name, tag = _LANG_MAP.get(self.lang, ("English", "en-IN"))
        return {"type": "resolved", "label": concept_id, "concept": ont_concept,
                "how": how, "text": real["text"],
                "english": self.ont.label(ont_concept, "en") if self.ont.is_valid_concept(ont_concept) else concept_id,
                "lang": self.lang, "tag": tag, "research": diag}

    def _events_from_action(self, act, probs, diag):
        if act["action"] == "resolved":
            concept_id = act["concept"]  # this is a recognizer class = sign_id
            # Apply to dialogue manager
            try:
                dm_res = self.dm.update_from_deaf_sign(concept_id, probs[self.rec.classes.index(concept_id)]
                                                       if concept_id in self.rec.classes else 1.0,
                                                       _LANG_MAP[self.lang][0])
            except (ValueError, KeyError):
                dm_res = None
            self.dialogue = None
            self.dialogue_start_time = None
            return [self._resolved_event(concept_id, act["how"], diag)]
        if act["action"] == "ask":
            opts = []
            for o in act["question"]["options"]:
                o = dict(o)
                cid_upper = o["value"].upper()
                ont_c = self.ont.concept_for_sign(o["value"])
                if ont_c:
                    o["label"] = self._label(ont_c)
                opts.append(o)
            act["question"]["options"] = opts
            return [{"type": "clarify", "question": act["question"],
                     "research": {**diag, "ig": act["ig"], "utility": act["utility"]}}]
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

        if self.dialogue is not None:
            if (self.dialogue_start_time and
                    now - self.dialogue_start_time > self.clarify_timeout):
                self.dialogue = None; self.dialogue_start_time = None
                return [{"type": "state", "state": "listening"},
                        {"type": "resign", "message": "Clarification timed out.", "research": {"timeout": True}}]
            return []

        frame_input = self.decode(jpeg) if self.decode is not None else jpeg
        feat = self.ext(frame_input)
        if not feat.any():
            return []

        p = feat[POSE].reshape(33, 3)
        has_pose = bool(p.any())
        wrist_y = float(min(p[15, 1], p[16, 1])) if has_pose else None
        lh_det = bool(feat[LH].any()); rh_det = bool(feat[RH].any())
        act = frame_activity(feat, self.stream.act_y)

        diag_frame = {"fps": round(self._fps, 1), "lh": lh_det, "rh": rh_det,
                      "pose": has_pose, "activity": act, "state": self.stream.state}
        out = [{"type": "diag", "research": diag_frame}]

        for e in self.stream.push(feat):
            if e["type"] == "sign_start":
                out.append({"type": "state", "state": "signing"})
            elif e["type"] == "provisional":
                out.append({"type": "provisional", "top": self._top(e["probs"])})
            elif e["type"] == "sign_end":
                probs = e["probs"]
                diag = {**uncertainty(list(probs)), "frames": e["n_frames"],
                        "top": self._top(probs, 5), "fps": round(self._fps, 1)}
                out.append({"type": "state", "state": "interpreting"})
                self.dialogue = Dialogue(list(map(float, probs)), self.rec.classes)
                act_res = self.dialogue.start()
                if act_res["action"] == "ask":
                    self.dialogue_start_time = now
                else:
                    self.dialogue_start_time = None
                out += self._events_from_action(act_res, probs, diag)
        return out

    def on_answer(self, value):
        if self.dialogue is None:
            return []
        diag = {"top": [{"concept": c, "p": float(p)}
                        for c, p in sorted(self.dialogue.cand.items(), key=lambda kv: -kv[1])]}
        act = self.dialogue.answer(value)
        if act["action"] == "ask":
            self.dialogue_start_time = time.time()
        else:
            self.dialogue = None; self.dialogue_start_time = None
        probs_arr = np.zeros(len(self.rec.classes))
        return self._events_from_action(act, probs_arr, diag)

    def reset(self):
        self.dialogue = None; self.dialogue_start_time = None
        self.stream.reset(); self.dm.reset()
