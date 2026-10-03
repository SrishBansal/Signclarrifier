"""Tests that cover the old app-core logic, now re-expressed against core/."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from core.ontology import get_ontology
from core.clarify import Dialogue, uncertainty, H

ONT = get_ontology()
CLASSES = [d["sign_id"] for d in ONT.concepts.values() if d.get("sign_id")]


# ── Language / realizer ────────────────────────────────────────────────────────

def test_every_concept_has_label_in_three_languages():
    for cid, d in ONT.concepts.items():
        for lc in ("en", "hi", "ta"):
            assert lc in d.get("label", {}), f"{cid} missing label[{lc!r}]"


def test_realizer_produces_non_empty_text():
    from core.semantics import NaturalLanguageRealizer
    from core.models import SemanticState, Item, Intent
    state = SemanticState(intent=Intent.QUESTION,
                          items=[Item(concept="SHIRT")],
                          current_focus_referent="SHIRT")
    for lang in ("English", "Hindi", "Tamil"):
        r = NaturalLanguageRealizer.realize(state, lang, ONT)
        assert r.get("text"), f"empty text for {lang}"


# ── Clarify / EIG ─────────────────────────────────────────────────────────────

def test_entropy_uniform():
    p = [0.25, 0.25, 0.25, 0.25]
    assert abs(H(p) - 2.0) < 0.01


def test_dialogue_start_asks_or_resolves():
    if len(CLASSES) < 2:
        return
    probs = [0.45, 0.40] + [0.15 / max(1, len(CLASSES)-2)] * (len(CLASSES)-2)
    d = Dialogue(probs, CLASSES)
    r = d.start()
    assert r["action"] in ("ask", "resolved")


def test_uncertainty_fields():
    u = uncertainty([0.6, 0.2, 0.2])
    assert "entropy" in u and "margin" in u and "top1" in u


# ── Sign session (stub) ────────────────────────────────────────────────────────

def test_session_reset_clears_state():
    """SignSession.reset() should clear dialogue and stream."""
    from core.dialogue import DialogueManager

    class FakeRec:
        classes = CLASSES or ["shirt"]
        def probs(self, x): return np.ones(len(self.classes)) / len(self.classes)

    class FakeExt:
        def __call__(self, f): return np.zeros(225, dtype=np.float32)

    from app.session import SignSession
    dm = DialogueManager(ONT)
    sess = SignSession(FakeRec(), FakeExt(), dm=dm)
    sess.reset()
    assert sess.dialogue is None


# ── NLU smoke ────────────────────────────────────────────────────────────────

def test_nlu_parse_greet():
    from core.nlu import NLUParser
    from core.models import Intent
    nlu = NLUParser(ONT)
    state = nlu.parse("hello namaste", "English")
    assert state.intent == Intent.GREET


def test_nlu_parse_price_query():
    from core.nlu import NLUParser
    from core.models import Intent
    nlu = NLUParser(ONT)
    state = nlu.parse("how much is this shirt", "English")
    assert state.intent in (Intent.QUESTION, Intent.REQUEST), state.intent
