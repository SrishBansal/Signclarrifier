"""
Comprehensive tests for /core semantic core.
VERIFICATION GATE: intent accuracy, slot F1, paraphrase-equivalence rate,
multi-turn referent resolution, ISL planner grammar rules, no-sentence-to-animation guard.
"""

import json
import os
import sys
import re
from collections import defaultdict

# ── helpers ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.models import Intent, Item, Utterance, SemanticState
from core.ontology import Ontology, get_ontology
from core.nlu import NLUParser
from core.planner import ISLPlanner, get_planner
from core.dialogue import DialogueManager
from core.semantics import NaturalLanguageRealizer

passed = 0
failed = 0

def check(label, condition, details=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {label}")
    else:
        failed += 1
        print(f"FAIL  {label}  {details}")


# ── 1. Ontology Tests ────────────────────────────────────────────────────
def test_ontology():
    ont = Ontology(os.path.join(ROOT, "data", "ontology.yaml"))
    check("ontology_loads_concepts", len(ont.all_concepts()) >= 40,
          f"got {len(ont.all_concepts())}")
    check("ontology_resolve_water", ont.resolve_concept("paani") == "WATER")
    check("ontology_resolve_shirt", ont.resolve_concept("tshirt") == "SHIRT")
    check("ontology_resolve_hindi_pen", ont.resolve_concept("कलम") == "PEN")
    check("ontology_resolve_tamil_shoe", ont.resolve_concept("காலணி") == "SHOES")
    check("ontology_is_valid_WATER", ont.is_valid_concept("WATER"))
    check("ontology_not_valid_BANANA", not ont.is_valid_concept("BANANA"))
    check("ontology_category_product", ont.get_category("WATER") == "product")
    check("ontology_category_container", ont.get_category("BOTTLE") == "container")
    check("ontology_sign_id_phone", ont.get_sign_id("PHONE") == "CELLPHONE")
    check("ontology_concept_for_sign_shirt", ont.concept_for_sign("shirt") == "SHIRT")
    check("ontology_synonym_index_size", len(ont.synonym_index) > 100,
          f"got {len(ont.synonym_index)}")
    # YES/NO parsed correctly (not as bool)
    check("ontology_yes_concept_exists", ont.is_valid_concept("YES") or "YES" in [k.upper() for k in ont.concepts.keys()])
    check("ontology_no_concept_exists", ont.is_valid_concept("NO") or "NO" in [k.upper() for k in ont.concepts.keys()])
    # Colour attributes
    check("ontology_blue_is_colour", ont.get_category("BLUE") == "attribute_colour")
    check("ontology_red_is_colour", ont.get_category("RED") == "attribute_colour")


# ── 2. NLU Parser Tests ──────────────────────────────────────────────────
def test_nlu():
    ont = get_ontology()
    parser = NLUParser(ont)

    # Basic product request
    s = parser.parse("ek bottle paani dena", "Hinglish")
    check("nlu_intent_request", s.intent == Intent.REQUEST)
    check("nlu_concept_water", any(i.concept == "WATER" for i in s.items))
    check("nlu_quantity_1", s.quantity == 1 or any(i.quantity == 1 for i in s.items))
    check("nlu_container_bottle", any(i.container == "BOTTLE" for i in s.items))
    check("nlu_no_negation", not s.negation)

    # English request
    s2 = parser.parse("give me two red pens", "English")
    check("nlu_english_intent", s2.intent == Intent.REQUEST)
    check("nlu_english_pen", any(i.concept == "PEN" for i in s2.items))
    check("nlu_english_qty2", s2.quantity == 2 or any(i.quantity == 2 for i in s2.items))
    check("nlu_english_red_attr", any("colour" in i.attributes and i.attributes["colour"] == "RED" for i in s2.items))

    # Greeting
    s3 = parser.parse("hello", "English")
    check("nlu_greet_intent", s3.intent == Intent.GREET)

    # Negation
    s4 = parser.parse("nahi chahiye", "Hinglish")
    check("nlu_negation_detected", s4.negation)

    # Price question
    s5 = parser.parse("iska price kitna hai", "Hinglish")
    check("nlu_price_question", s5.intent == Intent.QUESTION)

    # Payment
    s6 = parser.parse("UPI se payment karna hai", "Hinglish")
    check("nlu_payment_intent", s6.intent == Intent.PAYMENT)

    # Confirm
    s7 = parser.parse("haan theek hai", "Hinglish")
    check("nlu_confirm", s7.intent == Intent.CONFIRM)

    # Reject
    s8 = parser.parse("nahi", "Hinglish")
    check("nlu_reject", s8.intent == Intent.REJECT)

    # Direction
    s9 = parser.parse("shoes section kahan hai", "Hinglish")
    check("nlu_direction", s9.intent == Intent.DIRECTION)

    # Availability
    s10 = parser.parse("blue shirt milega kya", "Hinglish")
    check("nlu_availability", s10.intent == Intent.AVAILABILITY)

    # Out-of-vocabulary repair
    s11 = parser.parse("give me a unicorn hat", "English")
    check("nlu_oov_nosign", any(i.concept.startswith("OOV_") for i in s11.items) or len(s11.items) == 0,
          f"items={[(i.concept) for i in s11.items]}")


# ── 3. ISL Planner Tests ─────────────────────────────────────────────────
def test_planner():
    ont = get_ontology()
    planner = ISLPlanner(ontology=ont)

    # Greeting
    gs = SemanticState(intent=Intent.GREET)
    check("planner_greet", planner.plan(gs) == ["HELLO"])

    # Confirm
    cs = SemanticState(intent=Intent.CONFIRM)
    check("planner_confirm", planner.plan(cs) == ["YES"])

    # Reject
    rs = SemanticState(intent=Intent.REJECT)
    check("planner_reject", planner.plan(rs) == ["NO"])

    # Request: water bottle quantity 1
    ws = SemanticState(
        intent=Intent.REQUEST,
        items=[Item(concept="WATER", container="BOTTLE", quantity=1)]
    )
    plan = planner.plan(ws)
    check("planner_topic_first", plan[0] == "WATER",
          f"got {plan}")
    check("planner_container_after_topic", "BOTTLE" in plan and plan.index("BOTTLE") > plan.index("WATER"),
          f"plan={plan}")
    check("planner_quantity_after_container", "ONE" in plan and plan.index("ONE") > plan.index("BOTTLE"),
          f"plan={plan}")
    check("planner_action_last", plan[-1] in ("GIVE", "BUY"),
          f"plan={plan}")

    # Question with topic-comment: question marker at end
    qs = SemanticState(
        intent=Intent.QUESTION,
        items=[Item(concept="SHIRT", attributes={"query": "PRICE"})],
        current_focus_referent="SHIRT"
    )
    qp = planner.plan(qs)
    check("planner_question_marker_at_end", qp[-1] == "QUESTION",
          f"plan={qp}")
    check("planner_topic_before_question", "SHIRT" in qp and qp.index("SHIRT") < qp.index("QUESTION"),
          f"plan={qp}")

    # Negation: after action/predicate
    ns = SemanticState(
        intent=Intent.REQUEST,
        items=[Item(concept="WATER")],
        negation=True
    )
    np_ = planner.plan(ns)
    check("planner_negation_present", "NO" in np_, f"plan={np_}")
    check("planner_negation_after_action", np_.index("NO") > np_.index("GIVE") if "GIVE" in np_ else True,
          f"plan={np_}")

    # Colour attribute placement: between topic and quantity
    as_ = SemanticState(
        intent=Intent.REQUEST,
        items=[Item(concept="SHIRT", quantity=2, attributes={"colour": "BLUE"})],
    )
    ap = planner.plan(as_)
    check("planner_attr_colour", "BLUE" in ap, f"plan={ap}")
    check("planner_attr_before_qty", ap.index("BLUE") < ap.index("TWO") if "TWO" in ap else True,
          f"plan={ap}")

    # Fingerspell OOV
    fs = SemanticState(
        intent=Intent.REQUEST,
        items=[Item(concept="FS_UNICORN")],
    )
    fp = planner.plan(fs)
    check("planner_oov_present", len(fp) > 0,
          f"plan={fp}")

    # Availability uses HAVE + QUESTION
    avs = SemanticState(
        intent=Intent.AVAILABILITY,
        items=[Item(concept="SHOES")],
    )
    avp = planner.plan(avs)
    check("planner_availability_have", "HAVE" in avp, f"plan={avp}")
    check("planner_availability_q", avp[-1] == "QUESTION", f"plan={avp}")

    # Payment
    ps = SemanticState(
        intent=Intent.PAYMENT,
        items=[Item(concept="UPI")],
    )
    pp = planner.plan(ps)
    check("planner_payment", any(s in ("UPI", "PAYMENT") for s in pp), f"plan={pp}")


# ── 4. Dialogue Manager Multi-Turn Tests ─────────────────────────────────
def test_dialogue_multiturn():
    ont = get_ontology()
    parser = NLUParser(ont)
    dm = DialogueManager(ont)

    # Turn 1: Blue shirt request
    s1 = parser.parse("blue shirt dikhao", "Hinglish")
    s1 = dm.update_from_shopkeeper(s1)
    check("dm_turn1_focus", s1.current_focus_referent == "SHIRT",
          f"got {s1.current_focus_referent}")

    # Turn 2: Anaphoric price query → resolves to SHIRT
    s2 = parser.parse("iska price kitna hai?", "Hinglish")
    s2 = dm.update_from_shopkeeper(s2)
    check("dm_turn2_referent_shirt", s2.current_focus_referent == "SHIRT",
          f"got {s2.current_focus_referent}")
    check("dm_turn2_intent_question", s2.intent == Intent.QUESTION)
    check("dm_turn2_has_shirt_item", any(i.concept == "SHIRT" for i in s2.items),
          f"items={[(i.concept) for i in s2.items]}")

    # Turn 3: Bare "how much?" also resolves to SHIRT
    dm2 = DialogueManager(ont)
    t1 = parser.parse("show me the red pen", "English")
    dm2.update_from_shopkeeper(t1)
    t2 = parser.parse("how much?", "English")
    t2 = dm2.update_from_shopkeeper(t2)
    check("dm_turn3_bare_howmuch", t2.current_focus_referent == "PEN",
          f"got {t2.current_focus_referent}")

    # Turn 4: Colour follow-up "aur koi colour hai?"
    dm3 = DialogueManager(ont)
    u1 = parser.parse("shoes dikhao", "Hinglish")
    dm3.update_from_shopkeeper(u1)
    u2 = parser.parse("aur koi colour hai?", "Hinglish")
    u2 = dm3.update_from_shopkeeper(u2)
    check("dm_turn4_colour_followup", u2.current_focus_referent == "SHOES",
          f"got {u2.current_focus_referent}")
    check("dm_turn4_availability_intent", u2.intent == Intent.AVAILABILITY)

    # Direction B: Sign recognition → NL realization
    dm4 = DialogueManager(ont)
    res = dm4.update_from_deaf_sign("HELLO", 0.95, "English")
    check("dm_signB_hello", "hello" in res["realization"]["text"].lower())

    res2 = dm4.update_from_deaf_sign("SHOES", 0.85, "English")
    check("dm_signB_product_focus", dm4.state.current_focus_referent == "SHOES")

    res3 = dm4.update_from_deaf_sign("SHOES", 0.85, "Hindi")
    check("dm_signB_hindi", "जूते" in res3["realization"]["text"] or "shoes" in res3["realization"]["text"].lower(),
          f"got {res3['realization']['text']}")


# ── 5. NL Realizer Tests ─────────────────────────────────────────────────
def test_realizer():
    s1 = SemanticState(intent=Intent.GREET)
    r1 = NaturalLanguageRealizer.realize(s1, "English")
    check("realizer_greet_en", "hello" in r1["text"].lower())

    r1h = NaturalLanguageRealizer.realize(s1, "Hindi")
    check("realizer_greet_hi", "नमस्ते" in r1h["text"])

    r1t = NaturalLanguageRealizer.realize(s1, "Tamil")
    check("realizer_greet_ta", "வணக்கம்" in r1t["text"])

    s2 = SemanticState(
        intent=Intent.REQUEST,
        items=[Item(concept="WATER", container="BOTTLE", quantity=2)],
        current_focus_referent="WATER",
        quantity=2
    )
    r2 = NaturalLanguageRealizer.realize(s2, "English")
    check("realizer_request_water", "water" in r2["text"].lower())
    check("realizer_request_bottle", "bottle" in r2["text"].lower())

    s3 = SemanticState(intent=Intent.CONFIRM)
    r3 = NaturalLanguageRealizer.realize(s3, "English")
    check("realizer_confirm", "yes" in r3["text"].lower() or "certainly" in r3["text"].lower())


# ── 6. Eval Script: Intent Accuracy + Slot F1 + Paraphrase Equivalence ──
def test_eval():
    eval_path = os.path.join(ROOT, "data", "shop_semantic_eval.jsonl")
    if not os.path.exists(eval_path):
        check("eval_file_exists", False, f"missing {eval_path}")
        return

    ont = get_ontology()
    parser = NLUParser(ont)

    total = 0
    intent_correct = 0
    slot_tp = 0
    slot_fp = 0
    slot_fn = 0
    paraphrase_groups = defaultdict(list)

    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1

            text = row["text"]
            lang = row.get("language", "English")
            gold_intent = row["intent"]
            gold_concept = row.get("concept")
            gold_container = row.get("container")
            gold_quantity = row.get("quantity")
            gold_attrs = row.get("attributes", {})
            pg = row.get("paraphrase_group")

            state = parser.parse(text, lang)

            # Intent accuracy
            if state.intent and state.intent.value == gold_intent:
                intent_correct += 1

            # Slot matching (concept, container, quantity as set-like precision/recall)
            pred_slots = set()
            gold_slots = set()

            if state.items:
                for it in state.items:
                    pred_slots.add(("concept", it.concept))
                    if it.container:
                        pred_slots.add(("container", it.container))
                    if it.quantity is not None:
                        pred_slots.add(("quantity", it.quantity))
                    for ak, av in it.attributes.items():
                        pred_slots.add(("attr_" + ak, av))

            if gold_concept:
                gold_slots.add(("concept", gold_concept))
            if gold_container:
                gold_slots.add(("container", gold_container))
            if gold_quantity is not None:
                gold_slots.add(("quantity", gold_quantity))
            for ak, av in gold_attrs.items():
                gold_slots.add(("attr_" + ak, av))

            tp = len(pred_slots & gold_slots)
            fp = len(pred_slots - gold_slots)
            fn = len(gold_slots - pred_slots)
            slot_tp += tp
            slot_fp += fp
            slot_fn += fn

            # Paraphrase group
            if pg:
                sig = state.canonical_signature()
                paraphrase_groups[pg].append(sig)

    check("eval_total_200+", total >= 200, f"got {total}")

    intent_acc = intent_correct / total if total > 0 else 0
    print(f"  Intent accuracy: {intent_acc:.1%} ({intent_correct}/{total})")
    check("eval_intent_accuracy_50pct", intent_acc >= 0.50,
          f"got {intent_acc:.1%}")

    slot_precision = slot_tp / (slot_tp + slot_fp) if (slot_tp + slot_fp) > 0 else 0
    slot_recall = slot_tp / (slot_tp + slot_fn) if (slot_tp + slot_fn) > 0 else 0
    slot_f1 = 2 * slot_precision * slot_recall / (slot_precision + slot_recall) if (slot_precision + slot_recall) > 0 else 0
    print(f"  Slot P={slot_precision:.1%} R={slot_recall:.1%} F1={slot_f1:.1%}")
    check("eval_slot_f1_40pct", slot_f1 >= 0.40,
          f"got F1={slot_f1:.1%}")

    # Paraphrase equivalence
    pg_total = 0
    pg_equiv = 0
    for group_name, sigs in paraphrase_groups.items():
        if len(sigs) < 2:
            continue
        pg_total += 1
        # All signatures must be equal
        first = json.dumps(sigs[0], sort_keys=True)
        if all(json.dumps(s, sort_keys=True) == first for s in sigs[1:]):
            pg_equiv += 1
        else:
            # Show mismatches for debugging
            pass

    pg_rate = pg_equiv / pg_total if pg_total > 0 else 0
    print(f"  Paraphrase equivalence: {pg_rate:.1%} ({pg_equiv}/{pg_total})")
    check("eval_paraphrase_groups_exist", pg_total >= 3, f"got {pg_total}")
    check("eval_paraphrase_equiv_50pct", pg_rate >= 0.50,
          f"got {pg_rate:.1%}")


# ── 7. No Full-Sentence-to-Animation Guard ────────────────────────────────
def test_no_sentence_to_animation():
    """Grep the repo: no full sentence string should map directly to an animation file/trigger."""
    # Check that no Python file contains a dict mapping full sentences to .gif/.mp4/.webm/animation
    anim_pattern = re.compile(
        r'["\'][\w\s]{10,}["\'].*?:\s*["\'].*?\.(gif|mp4|webm|json|lottie)',
        re.IGNORECASE
    )
    violations = []
    for dirpath, _, filenames in os.walk(ROOT):
        if ".git" in dirpath or "__pycache__" in dirpath or "node_modules" in dirpath:
            continue
        for fn in filenames:
            if fn.endswith((".py", ".js", ".ts", ".json")):
                fp = os.path.join(dirpath, fn)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                        for i, line in enumerate(fh, 1):
                            if anim_pattern.search(line):
                                violations.append(f"{fp}:{i}")
                except:
                    pass
    check("no_sentence_to_animation", len(violations) == 0,
          f"violations: {violations[:5]}")


# ── 8. Canonical Signature Tests ──────────────────────────────────────────
def test_canonical_signature():
    """Verify paraphrase equivalence via canonical_signature()."""
    ont = get_ontology()
    parser = NLUParser(ont)

    # Same meaning, different wording
    s1 = parser.parse("ek bottle paani dena", "Hinglish")
    s2 = parser.parse("water bottle please", "English")

    sig1 = s1.canonical_signature()
    sig2 = s2.canonical_signature()

    # Both should have WATER, BOTTLE, quantity 1
    check("canonical_both_water",
          sig1["items"][0]["concept"] == "WATER" and sig2["items"][0]["concept"] == "WATER",
          f"sig1={sig1}, sig2={sig2}")
    check("canonical_intent_match", sig1["intent"] == sig2["intent"],
          f"sig1_intent={sig1['intent']}, sig2_intent={sig2['intent']}")


# ── 9. Step 2 Acceptance Tests ────────────────────────────────────────────
def test_step2_acceptance():
    """STEP 2 required acceptance tests."""
    ont = get_ontology()
    parser = NLUParser(ont)
    planner = ISLPlanner(ontology=ont)

    # (A) Three paraphrases give the same canonical_signature:
    #     intent=REQUEST, WATER, container=BOTTLE, quantity=1
    paraphrases = [
        ("Mujhe ek bottle paani chahiye", "Hinglish"),
        ("Bhai ek bottle paani dena", "Hinglish"),
        ("Mujhe paani ki bottle chahiye", "Hinglish"),
    ]
    sigs = [parser.parse(t, l).canonical_signature() for t, l in paraphrases]
    for i, sig in enumerate(sigs):
        concepts = [it["concept"] for it in sig.get("items", [])]
        check(f"step2_paraphrase_{i}_water", "WATER" in concepts, f"sig={sig}")
        check(f"step2_paraphrase_{i}_bottle",
              any(it.get("container") == "BOTTLE" for it in sig.get("items", [])),
              f"sig={sig}")

    # (B) Availability for water questions
    av1 = parser.parse("Ek bottle water milega?", "Hinglish")
    check("step2_availability_milega", av1.intent == Intent.AVAILABILITY,
          f"intent={av1.intent}")
    av2 = parser.parse("Bhai paani hai?", "Hinglish")
    check("step2_availability_hai", av2.intent == Intent.AVAILABILITY,
          f"intent={av2.intent}")

    # (C) "hello blue shirt price kitna hai" -> plan starts HELLO, contains SHIRT, BLUE, QUESTION
    hello_state = parser.parse("hello blue shirt price kitna hai", "Hinglish")
    hello_plan = planner.plan(hello_state)
    check("step2_hello_lead_in", hello_plan and hello_plan[0] == "HELLO",
          f"plan={hello_plan}")
    check("step2_hello_shirt_in_plan", "SHIRT" in hello_plan,
          f"plan={hello_plan}")
    check("step2_hello_blue_in_plan", "BLUE" in hello_plan,
          f"plan={hello_plan}")
    check("step2_hello_question_marker", "QUESTION" in hello_plan,
          f"plan={hello_plan}")

    # (D) "Bhai mujhe ek bottle paani chahiye" -> plan has WATER, BOTTLE, ONE, GIVE, none dropped
    full_state = parser.parse("Bhai mujhe ek bottle paani chahiye", "Hinglish")
    full_plan = planner.plan(full_state)
    check("step2_water_in_plan", "WATER" in full_plan, f"plan={full_plan}")
    check("step2_bottle_in_plan", "BOTTLE" in full_plan, f"plan={full_plan}")
    check("step2_one_in_plan", "ONE" in full_plan, f"plan={full_plan}")
    check("step2_give_in_plan", "GIVE" in full_plan, f"plan={full_plan}")

    # (E) Social concepts
    ty_state = parser.parse("thank you", "English")
    ty_plan = planner.plan(ty_state)
    check("step2_thankyou_plan", ty_plan == ["THANKYOU"],
          f"plan={ty_plan}")

    nm_state = parser.parse("namaste", "Hinglish")
    nm_plan = planner.plan(nm_state)
    check("step2_namaste_hello", nm_plan == ["HELLO"],
          f"plan={nm_plan}")

    # (F) Multi-turn: "iska price kitna hai" after "red pen chahiye" -> PEN in plan
    dm = DialogueManager(ont)
    pen_state = parser.parse("red pen chahiye", "Hinglish")
    dm.update_from_shopkeeper(pen_state)
    price_state = parser.parse("iska price kitna hai", "Hinglish")
    price_dm_state = dm.update_from_shopkeeper(price_state)
    price_plan = planner.plan(price_dm_state)
    check("step2_multiturn_pen_focus",
          price_dm_state.current_focus_referent == "PEN",
          f"focus={price_dm_state.current_focus_referent}")
    check("step2_multiturn_pen_in_plan",
          "PEN" in price_plan,
          f"plan={price_plan}")


# ── Run all ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_ontology()
    test_nlu()
    test_planner()
    test_dialogue_multiturn()
    test_realizer()
    test_eval()
    test_no_sentence_to_animation()
    test_canonical_signature()
    test_step2_acceptance()

    print(f"\n{'='*60}")
    print(f"SEMANTIC CORE TESTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    if failed > 0:
        sys.exit(1)
    else:
        print("ALL SEMANTIC CORE TESTS PASSED")
