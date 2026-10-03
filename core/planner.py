"""
ISL Sequence Planner for ClarifySign.
Implements Rule 1 & Rule 2: Strictly decoupled from natural language.
Converts SemanticState into an ordered list of concept IDs based on ISL grammar rules.
Has NO web or framework dependencies.
"""

import os
from typing import List, Optional, Dict, Any
import yaml
from .models import SemanticState, Intent
from .ontology import get_ontology, Ontology


class ISLPlanner:
    """Deterministic grammar planner for Indian Sign Language concept sequences."""

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

    def plan(self, state: SemanticState) -> List[str]:
        """Translates a SemanticState into an ordered list of ISL concept IDs."""
        sequence: List[str] = []

        if not state:
            return sequence

        intent = state.intent or Intent.UNKNOWN

        # 1. Greetings (GREET intent or social lead)
        if intent == Intent.GREET:
            return ["HELLO"]
        if intent == Intent.CONFIRM:
            return ["YES"]
        if intent == Intent.REJECT:
            return ["NO"]

        # Collect components
        greeting_signs = []
        topic_signs = []
        container_signs = []
        attr_signs = []
        quantity_signs = []
        action_signs = []
        negation_signs = []
        question_signs = []

        # Items & Referents (Topic)
        if state.items:
            for item in state.items:
                cid = item.concept.upper()
                if cid.startswith("FS_"):
                    # Out of vocabulary: expand to fingerspell letters
                    word = cid[3:]
                    topic_signs.extend(self.ontology.fingerspell(word))
                elif cid != "THIS" and cid != "ITEM":
                    topic_signs.append(self.ontology.get_sign_id(cid))
                elif state.current_focus_referent:
                    ref_cid = state.current_focus_referent.upper()
                    topic_signs.append(self.ontology.get_sign_id(ref_cid))

                # Container
                if item.container:
                    cont_id = item.container.upper()
                    container_signs.append(self.ontology.get_sign_id(cont_id))

                # Attributes (Colour, Size, State)
                if item.attributes:
                    for k in ("colour", "size", "state"):
                        val = item.attributes.get(k)
                        if val:
                            attr_signs.append(self.ontology.get_sign_id(str(val).upper()))

                # Quantity (In ISL: Topic + Container + Quantity)
                if item.quantity is not None:
                    q_val = item.quantity
                    qty_word = self.rules.get("quantity_words", {}).get(q_val, str(q_val))
                    quantity_signs.append(self.ontology.get_sign_id(qty_word))

        elif state.current_focus_referent:
            topic_signs.append(self.ontology.get_sign_id(state.current_focus_referent.upper()))

        # Intent-driven Actions & Predicates
        if intent == Intent.QUESTION:
            # Check if asking for price/cost
            if any(it.concept in ("PRICE", "COST") for it in state.items) or "price" in state.active_attributes:
                action_signs.append("COST")
            question_signs.append("QUESTION")

        elif intent == Intent.AVAILABILITY:
            action_signs.append("HAVE")
            question_signs.append("QUESTION")

        elif intent == Intent.PAYMENT:
            pay_concept = None
            for it in state.items:
                if it.concept in ("UPI", "CASH", "QR", "CHANGE", "BILL"):
                    pay_concept = it.concept
                    break
            if pay_concept:
                action_signs.append(self.ontology.get_sign_id(pay_concept))
            else:
                action_signs.append("PAYMENT")
            
            if "?" in (state.raw_text or "") or state.polarity == "neutral":
                question_signs.append("QUESTION")

        elif intent == Intent.DIRECTION:
            action_signs.append("WHERE")
            question_signs.append("QUESTION")

        elif intent == Intent.REQUEST:
            action_signs.append("BUY" if "buy" in (state.raw_text or "").lower() else "GIVE")

        # Negation handling: Placed after the predicate/action or at end of topic
        if state.negation:
            negation_signs.append("NO")

        # Assemble strictly according to ISL Topic-Comment ordering rules:
        # [GREETING] -> [TOPIC_ENTITY] -> [CONTAINER] -> [ATTRIBUTES] -> [QUANTITY] -> [ACTION_PREDICATE] -> [NEGATION] -> [QUESTION_MARKER]
        sequence = (
            greeting_signs +
            topic_signs +
            container_signs +
            attr_signs +
            quantity_signs +
            action_signs +
            negation_signs +
            question_signs
        )

        # De-duplicate consecutive identical signs while preserving order
        deduped = []
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
