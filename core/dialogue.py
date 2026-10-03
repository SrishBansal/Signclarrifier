"""
Dialogue State and Multi-Turn Conversation Manager for ClarifySign.
Maintains dialogue history, current-focus referent, anaphora resolution, and bidirectional turns.
Has NO web or framework dependencies.
"""

from typing import Dict, Any, Optional, List
from .models import SemanticState, Intent, Item
from .ontology import get_ontology, Ontology


class DialogueManager:
    """Manages multi-turn dialogue context, referent resolution, and anaphora."""

    def __init__(self, ontology: Optional[Ontology] = None):
        self.ontology = ontology or get_ontology()
        self.state = SemanticState()

    def get_state(self) -> SemanticState:
        return self.state

    def reset(self) -> SemanticState:
        self.state = SemanticState(
            intent=None,
            items=[],
            current_focus_referent=None,
            active_attributes={},
            quantity=None,
            negation=False,
            polarity="positive",
            context="active_shop_counter",
            dialogue_history=[]
        )
        return self.state

    def update_from_shopkeeper(self, parsed: SemanticState) -> SemanticState:
        """Processes Shopkeeper input (Direction A) and updates multi-turn state."""
        raw_text_lower = (parsed.raw_text or "").lower()

        # Check for referent resolution / anaphora:
        # e.g., "iska price kitna hai", "how much", "aur koi colour hai", "ek aur de do"
        is_anaphoric = False
        if parsed.items:
            first_item = parsed.items[0]
            if first_item.concept in ("THIS", "ITEM") or first_item.concept.startswith("FS_THIS"):
                is_anaphoric = True
        elif not parsed.items and parsed.intent in (Intent.QUESTION, Intent.AVAILABILITY, Intent.REQUEST):
            is_anaphoric = True

        # Price inquiry on referent: "how much?", "iska price kitna hai?"
        if any(w in raw_text_lower for w in ("how much", "kitna", "kitne", "price", "cost", "daam", "kimat")):
            if self.state.current_focus_referent:
                is_anaphoric = True
                parsed.intent = Intent.QUESTION
                # Ensure the product is attached
                if not any(it.concept == self.state.current_focus_referent for it in parsed.items):
                    parsed.items = [Item(concept=self.state.current_focus_referent, attributes={"query": "PRICE"})]
                else:
                    for it in parsed.items:
                        it.attributes["query"] = "PRICE"

        # Attribute follow-up: "aur koi colour hai?", "any other color?"
        if any(w in raw_text_lower for w in ("colour", "color", "rang", "size")):
            if self.state.current_focus_referent:
                is_anaphoric = True
                parsed.intent = Intent.AVAILABILITY
                target_attr = "colour" if any(w in raw_text_lower for w in ("colour", "color", "rang")) else "size"
                if not any(it.concept == self.state.current_focus_referent for it in parsed.items):
                    parsed.items = [Item(concept=self.state.current_focus_referent, attributes={"query": target_attr})]

        # Quantity follow-up: "aur ek dena", "ek aur de do", "do de do"
        if ("aur" in raw_text_lower or "more" in raw_text_lower or "another" in raw_text_lower) and parsed.quantity:
            if self.state.current_focus_referent:
                is_anaphoric = True
                curr_qty = self.state.quantity or 1
                new_qty = curr_qty + parsed.quantity
                parsed.quantity = new_qty
                parsed.items = [Item(concept=self.state.current_focus_referent, quantity=new_qty)]

        # If a concrete product is mentioned, update the focus referent
        concrete_product = None
        for it in parsed.items:
            cid = it.concept.upper()
            if self.ontology.get_category(cid) == "product":
                concrete_product = cid
                break

        if concrete_product:
            self.state.current_focus_referent = concrete_product
        elif is_anaphoric and self.state.current_focus_referent:
            # Maintain the current referent
            pass

        # Update active state fields
        self.state.intent = parsed.intent
        self.state.items = parsed.items
        self.state.negation = parsed.negation
        self.state.polarity = parsed.polarity
        self.state.raw_text = parsed.raw_text
        self.state.language = parsed.language
        if parsed.quantity is not None:
            self.state.quantity = parsed.quantity
        if parsed.active_attributes:
            self.state.active_attributes.update(parsed.active_attributes)

        # Record history turn
        self.state.dialogue_history.append({
            "speaker": "shopkeeper",
            "text": parsed.raw_text,
            "intent": parsed.intent.value if parsed.intent else None,
            "referent": self.state.current_focus_referent,
            "items": [it.canonical_dict() for it in self.state.items]
        })

        return self.state

    def update_from_deaf_sign(self, predicted_sign: str, confidence: float, language: str = "English") -> Dict[str, Any]:
        """Direction B: Updates state from customer's recognized sign and realizes speech."""
        sign_upper = predicted_sign.upper()

        if self.ontology.get_category(sign_upper) == "product":
            self.state.current_focus_referent = sign_upper
            self.state.items = [Item(concept=sign_upper)]
            self.state.intent = Intent.REQUEST
        elif sign_upper in ("PRICE", "COST"):
            self.state.intent = Intent.QUESTION
            if self.state.current_focus_referent:
                self.state.items = [Item(concept=self.state.current_focus_referent, attributes={"query": "PRICE"})]
        elif sign_upper in ("HELLO", "GREETING"):
            self.state.intent = Intent.GREET
        elif sign_upper in ("GOOD", "YES"):
            self.state.intent = Intent.CONFIRM
        elif sign_upper in ("NO",):
            self.state.intent = Intent.REJECT
        elif sign_upper in ("UPI", "CASH", "PAYMENT"):
            self.state.intent = Intent.PAYMENT

        # Natural language realization
        from .semantics import NaturalLanguageRealizer
        realization = NaturalLanguageRealizer.realize(self.state, language=language)

        self.state.dialogue_history.append({
            "speaker": "customer_isl",
            "sign": sign_upper,
            "confidence": confidence,
            "intent": self.state.intent.value if self.state.intent else None,
            "referent": self.state.current_focus_referent
        })

        return {
            "realization": realization,
            "semantic_state": self.state.model_dump()
        }

    def resolve_clarification(self, chosen_label: str, language: str = "English") -> Dict[str, Any]:
        """Direction B: Resolves clarification choice confirming intended concept."""
        return self.update_from_deaf_sign(chosen_label, confidence=1.0, language=language)
