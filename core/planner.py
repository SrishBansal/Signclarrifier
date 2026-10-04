"""
ISL Sequence Planner for ClarifySign.
Implements Rule 1 & Rule 2: Strictly decoupled from natural language.
Converts SemanticState into an ordered list of CONCEPT IDs based on ISL grammar rules.
Has NO web or framework dependencies.

IMPORTANT: This planner returns CONCEPT IDs throughout. The server resolves sign_ids
and sets kind/gloss from the ontology. No entry is ever dropped.
"""

import os
from typing import List, Optional, Dict, Any
import yaml
from .models import SemanticState, Intent
from .ontology import get_ontology, Ontology


class ISLPlanner:
    """Deterministic grammar planner for Indian Sign Language concept sequences.

    Returns CONCEPT IDs only - the server resolves sign_ids via the ontology.
    Greeting intent adds HELLO as lead-in, then continues with remaining content.
    No entry is ever dropped; concepts without signs get kind='nosign' in the server.
    """

    def __init__(self, rules_path: Optional[str] = None, ontology: Optional[Ontology] = None):
        if rules_path is None:
            rules_path = os.path.join(os.path.dirname(__file__), "..", "data", "grammar_rules.yaml")
        self.rules_path = os.path.abspath(rules_path)
        self.ontology = ontology or get_ontology()
        self.rules: Dict[str, Any] = {}
        self.load_rules()

    def load_rules(self):
        if not os.path.exists(self.rules_path):
            raise FileNotFoundError(f"Grammar rules file not found at {self.rules_path}")
        with open(self.rules_path, "r", encoding="utf-8") as f:
            self.rules = yaml.safe_load(f) or {}

    def _qty_concept(self, q: int) -> Optional[str]:
        """Map integer quantity to a quantity CONCEPT ID via grammar_rules.yaml."""
        qty_word = self.rules.get("quantity_words", {}).get(q)
        if qty_word:
            return str(qty_word).upper()
        return None

    def _action_concept(self, raw_text: str) -> str:
        """Return the action CONCEPT ID for REQUEST intents."""
        if "buy" in (raw_text or "").lower():
            return "BUY"
        return "GIVE"

    def plan(self, state: SemanticState) -> List[str]:
        """Translate a SemanticState into an ordered list of CONCEPT IDs.

        Rules:
        - Returns CONCEPT IDs only, never sign_ids.
        - Greeting intent adds HELLO as lead-in, then continues planning remaining content.
        - Social concepts (THANKYOU) in items produce their concept directly.
        - No entry is ever dropped; concepts without signs keep kind='nosign' in the server.
        - Ordering: [GREETING] [TOPIC] [CONTAINER] [ATTRS] [QTY] [ACTION] [NEGATION] [QUESTION]
        """
        sequence: List[str] = []

        if not state:
            return sequence

        intent = state.intent or Intent.UNKNOWN

        # Pure social intents (CONFIRM / REJECT)
        if intent == Intent.CONFIRM:
            return ["YES"]
        if intent == Intent.REJECT:
            return ["NO"]

        # Detect greeting lead-in and separate social vs content items
        has_greeting_intent = (intent == Intent.GREET)
        social_items: List[str] = []
        content_items = []
        for it in state.items:
            cid = it.concept.upper()
            cat = self.ontology.get_category(cid) if not cid.startswith("OOV_") else None
            if cat == "social":
                social_items.append(cid)
            else:
                content_items.append(it)

        greeting_prefix: List[str] = []
        if has_greeting_intent or "HELLO" in social_items:
            greeting_prefix = ["HELLO"]

        # Pure THANKYOU (no other content)
        if "THANKYOU" in social_items and not content_items:
            return ["THANKYOU"]

        # Pure GREET with no content
        if has_greeting_intent and not content_items and all(s == "HELLO" for s in social_items):
            return ["HELLO"]
        if has_greeting_intent and not content_items and not social_items:
            return ["HELLO"]

        # Collect plan components using CONCEPT IDs
        topic_concepts: List[str] = []
        container_concepts: List[str] = []
        attr_concepts: List[str] = []
        quantity_concepts: List[str] = []
        action_concepts: List[str] = []
        negation_concepts: List[str] = []
        question_concepts: List[str] = []

        # Items & Referents (Topic)
        items_to_plan = content_items if content_items else state.items
        if items_to_plan:
            for item in items_to_plan:
                cid = item.concept.upper()
                if cid.startswith("OOV_") or cid.startswith("FS_"):
                    topic_concepts.append(cid)
                elif cid not in ("THIS", "ITEM"):
                    topic_concepts.append(cid)
                elif state.current_focus_referent:
                    topic_concepts.append(state.current_focus_referent.upper())

                # Container
                if item.container:
                    container_concepts.append(item.container.upper())

                # Attributes (Colour, Size, State)
                if item.attributes:
                    for k in ("colour", "size", "state"):
                        val = item.attributes.get(k)
                        if val:
                            attr_concepts.append(str(val).upper())

                # Quantity -> concept ID via grammar_rules.yaml
                if item.quantity is not None:
                    qc = self._qty_concept(item.quantity)
                    if qc:
                        quantity_concepts.append(qc)

        elif state.current_focus_referent:
            topic_concepts.append(state.current_focus_referent.upper())

        # Intent-driven Actions & Predicates
        if intent == Intent.QUESTION:
            if any(it.concept in ("PRICE", "COST") for it in (items_to_plan or [])) or \
               "price" in (state.active_attributes or {}):
                action_concepts.append("PRICE")
            question_concepts.append("QUESTION")

        elif intent == Intent.AVAILABILITY:
            action_concepts.append("HAVE")
            question_concepts.append("QUESTION")

        elif intent == Intent.PAYMENT:
            pay_concept = None
            for it in (items_to_plan or []):
                cat = self.ontology.get_category(it.concept) if self.ontology.is_valid_concept(it.concept) else None
                if cat == "commercial":
                    pay_concept = it.concept.upper()
                    break
            action_concepts.append(pay_concept if pay_concept else "PAYMENT")
            if "?" in (state.raw_text or "") or state.polarity == "neutral":
                question_concepts.append("QUESTION")

        elif intent == Intent.DIRECTION:
            action_concepts.append("WHERE")
            question_concepts.append("QUESTION")

        elif intent in (Intent.REQUEST, Intent.GREET) and content_items:
            action_concepts.append(self._action_concept(state.raw_text))

        # Negation handling
        if state.negation:
            negation_concepts.append("NO")

        # Assemble strictly according to ISL Topic-Comment ordering rules:
        # [GREETING] [TOPIC_ENTITY] [CONTAINER] [ATTRIBUTES] [QUANTITY]
        # [ACTION_PREDICATE] [NEGATION] [QUESTION_MARKER]
        sequence = (
            greeting_prefix +
            topic_concepts +
            container_concepts +
            attr_concepts +
            quantity_concepts +
            action_concepts +
            negation_concepts +
            question_concepts
        )

        # De-duplicate consecutive identical signs while preserving order
        deduped: List[str] = []
        for s in sequence:
            if not deduped or deduped[-1] != s:
                deduped.append(s)

        return deduped if deduped else ["HELLO"]


# Global singleton instance
_planner_instance: Optional[ISLPlanner] = None


def get_planner() -> ISLPlanner:
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = ISLPlanner()
    return _planner_instance
