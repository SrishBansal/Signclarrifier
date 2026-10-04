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
    active_codes = [entry["code"] for entry in ONT.languages() if not entry.get("nlu_input_only")]
    for cid, d in ONT.concepts.items():
        for lc in active_codes:
            assert lc in d.get("label", {}), f"{cid} missing label[{lc!r}]"


def test_realizer_produces_non_empty_text():
    from core.semantics import NaturalLanguageRealizer
    from core.models import SemanticState, Item, Intent
    state = SemanticState(intent=Intent.QUESTION,
                          items=[Item(concept="SHIRT")],
                          current_focus_referent="SHIRT")
    for lang in (entry["name"] for entry in ONT.languages() if not entry.get("nlu_input_only")):
        r = NaturalLanguageRealizer.realize(state, lang, ONT)
        assert r.get("text"), f"empty text for {lang}"


def test_customer_realizer_uses_direction_b_templates():
    """Direction B is spoken by the customer, not the shopkeeper."""
    from core.semantics import NaturalLanguageRealizer
    from core.models import SemanticState, Intent
    state = SemanticState(intent=Intent.GREET)
    assert NaturalLanguageRealizer.realize(state, "English", ONT)["text"] != "Hello."
    for language in (entry["name"] for entry in ONT.languages() if not entry.get("nlu_input_only")):
        assert NaturalLanguageRealizer.realize(state, language, ONT, perspective="customer")["text"]


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


def test_no_useful_question_resolves_high_top1_as_best_guess():
    cfg = {"top_k": 3, "commit_p": 0.85, "commit_margin": 0.40,
           "resign_p": 0.15, "min_gain": 2.0, "max_turns": 2,
           "best_guess_p": 0.40}
    result = Dialogue([0.83, 0.10, 0.07], ["shirt", "pen", "red"], cfg).start()
    assert result == {"action": "resolved", "concept": "shirt", "how": "best_guess"}


def test_no_useful_question_resigns_for_flat_distribution():
    cfg = {"top_k": 3, "commit_p": 0.85, "commit_margin": 0.40,
           "resign_p": 0.15, "min_gain": 2.0, "max_turns": 2,
           "best_guess_p": 0.40}
    assert Dialogue([1 / 3, 1 / 3, 1 / 3], ["shirt", "pen", "red"], cfg).start()["action"] == "resign"


def test_near_candidates_ask_pair_question():
    result = Dialogue([0.45, 0.35, 0.20], ["shirt", "pen", "red"]).start()
    assert result["action"] == "ask"
    assert result["question"]["id"] == "pair"


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


def test_session_emits_diag_when_pose_is_missing():
    """The always-visible camera HUD must keep updating on blank frames."""
    from app.session import SignSession

    class FakeRec:
        classes = CLASSES or ["shirt"]
        def probs(self, x): return np.ones(len(self.classes)) / len(self.classes)

    class FakeExt:
        last = {"aspect": None, "y_scale": 1.0}
        def __call__(self, f): return np.zeros(225, dtype=np.float32)

    events = SignSession(FakeRec(), FakeExt()).on_frame(b"unused")
    assert events[0]["type"] == "diag"
    assert events[0]["research"]["pose"] is False


def test_session_uses_ontology_language_metadata_for_direction_b():
    """Every active ontology language can drive resolved text and its TTS tag."""
    from app.session import SignSession

    class FakeRec:
        classes = ["shirt"]
        def probs(self, x): return np.array([1.0])

    class FakeExt:
        def __call__(self, f): return np.zeros(225, dtype=np.float32)

    for language in (entry for entry in ONT.languages() if not entry.get("nlu_input_only")):
        session = SignSession(FakeRec(), FakeExt(), lang=language["code"])
        event = session._events_from_action(
            {"action": "resolved", "concept": "shirt", "how": "confident"},
            np.array([1.0]), {})[0]
        assert event["lang"] == language["code"]
        assert event["tag"] == language["speech_tag"]
        assert event["text"]


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
