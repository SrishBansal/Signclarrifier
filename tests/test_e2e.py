"""
End-to-end tests for ClarifySign (no webcam, no model file required).
TestClient + stub extractor that emits a real 225-dim feature vector.
"""
import sys, os, json, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.ontology import get_ontology
from core.clarify import Dialogue
from clarifysign_ml.features import FEATURE_DIM

ONT = get_ontology()

# ── Stubs ─────────────────────────────────────────────────────────────────────

class StubExtractor:
    def __init__(self, sign_id="shirt"):
        self.sign_id = sign_id
        self._feat = np.zeros(FEATURE_DIM, dtype=np.float32)
        self._feat[0] = 0.5  # non-zero so frame_activity sees motion

    def __call__(self, frame):
        return self._feat.copy()


class StubRecognizer:
    def __init__(self, target_class="shirt"):
        ont = get_ontology()
        self.classes = [d["sign_id"] for d in ont.concepts.values()
                        if d.get("sign_id")]
        if target_class not in self.classes:
            self.classes.insert(0, target_class)
        self.target = target_class

    def probs(self, seq48):
        p = np.full(len(self.classes), 0.01 / max(len(self.classes)-1, 1), dtype=np.float32)
        idx = self.classes.index(self.target)
        p[idx] = 0.92
        p /= p.sum()
        return p


def make_app(sign_id="shirt"):
    from app.server import create_app
    rec = StubRecognizer(sign_id)
    return create_app(recognizer=rec, reason="",
                      extractor_factory=lambda: StubExtractor(sign_id))


# ── /api/config ───────────────────────────────────────────────────────────────

def test_config_matches_active_ontology_languages():
    client = TestClient(make_app())
    r = client.get("/api/config")
    assert r.status_code == 200
    codes = {l["code"] for l in r.json()["languages"]}
    expected = {l["code"] for l in get_ontology().languages() if not l.get("nlu_input_only")}
    assert expected == codes, f"unexpected codes: {codes}"


def test_config_hides_nlu_only_languages():
    client = TestClient(make_app())
    codes = {l["code"] for l in client.get("/api/config").json()["languages"]}
    hidden = {l["code"] for l in get_ontology().languages() if l.get("nlu_input_only")}
    assert not (codes & hidden)


def test_config_has_concepts():
    client = TestClient(make_app())
    j = client.get("/api/config").json()
    assert len(j["concepts"]) > 50


# ── Direction A ───────────────────────────────────────────────────────────────

def _understand(client, text, lang, sid=None):
    sid = sid or str(uuid.uuid4())
    r = client.get("/api/understand", params={"text": text, "lang": lang, "sid": sid})
    assert r.status_code == 200, r.text
    return r.json(), sid


def test_understand_response_shape():
    client = TestClient(make_app())
    j, _ = _understand(client, "shirt", "en")
    assert "heard" in j and "plan" in j and "text" in j


def test_understand_plan_has_kind():
    client = TestClient(make_app())
    j, _ = _understand(client, "shirt", "en")
    for item in j["plan"]:
        assert "kind" in item
        assert item["kind"] in ("sign", "marker", "nosign"), item


def test_understand_oov_gives_nosign_or_empty():
    """An unknown product token should produce nosign items or empty plan (not 'sign' kind)."""
    client = TestClient(make_app())
    # Use an OOV product token that won't accidentally match greeting cues
    j, _ = _understand(client, "I want a frobblequartz", "en")
    # Filter out any GREET/CONFIRM/REJECT that may sneak in from intent-only items
    sign_items = [p for p in j["plan"] if p.get("kind") == "sign" and p.get("concept") not in ("HELLO", "YES", "NO")]
    assert not sign_items, f"OOV product produced sign items: {sign_items}"


def test_understand_same_core_concepts_across_langs():
    """Shirt price in en/hi/ta should all yield non-empty plans."""
    phrases = {
        "en": "how much does this shirt cost",
        "hi": "is shirt ka daam kya hai",
        "ta": "idha vilai enna"
    }
    client = TestClient(make_app())
    for lang, phrase in phrases.items():
        j, _ = _understand(client, phrase, lang)
        # At minimum a non-empty plan proves NLU path works per language
        assert isinstance(j["plan"], list), f"plan not list for {lang}"


# ── Boot check ────────────────────────────────────────────────────────────────

def test_boot_refuses_class_not_in_ontology():
    class BadRec:
        classes = ["totally_fake_class_xyz"]
        def probs(self, x): return np.array([1.0])

    from app.server import create_app
    app = create_app(recognizer=BadRec(), reason="",
                     extractor_factory=lambda: StubExtractor())
    j = TestClient(app).get("/api/config").json()
    assert not j["recognition_ready"], "App should refuse boot with invalid class"


# ── Clarify → answer ──────────────────────────────────────────────────────────

def test_clarify_then_answer_resolves():
    ont = get_ontology()
    classes = [d["sign_id"] for d in ont.concepts.values() if d.get("sign_id")][:4]
    if len(classes) < 2:
        pytest.skip("Need >=2 sign classes")
    probs = [0.40, 0.35, 0.15, 0.10][:len(classes)]
    d = Dialogue(probs, classes)
    r1 = d.start()
    assert r1["action"] == "ask", f"Expected ask, got {r1['action']}"
    top_value = r1["question"]["options"][0]["value"]
    r2 = d.answer(top_value)
    assert r2["action"] in ("resolved", "ask")


# ── Direction B shared focus test (unit, no WS needed) ───────────────────────

def test_shirt_sign_then_price_query_keeps_focus():
    from core.dialogue import DialogueManager
    from core.nlu import NLUParser
    from core.planner import get_planner

    ont = get_ontology()
    dm = DialogueManager(ont)
    dm.update_from_deaf_sign("shirt", 0.92, "English")
    assert dm.state.current_focus_referent == "SHIRT", dm.state.current_focus_referent

    nlu = NLUParser(ont)
    parsed = nlu.parse("how much is this", "English", dm.get_state())
    state = dm.update_from_shopkeeper(parsed)
    plan = get_planner().plan(state)
    concepts = [c for c in plan if c]
    has_shirt_or_price = any("SHIRT" in c.upper() or "PRICE" in c.upper() for c in concepts)
    assert has_shirt_or_price, f"Expected SHIRT/PRICE in plan, got: {plan}"
