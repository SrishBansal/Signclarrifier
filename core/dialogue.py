"""
Dialogue State Manager for ClarifySign.
Direction B: update_from_deaf_sign uses concept_for_sign and ontology signed_effect.
No sign names in Python.
"""
from typing import Dict, Any, Optional, List
from .models import SemanticState, Intent, Item
from .ontology import get_ontology, Ontology


class DialogueManager:
    def __init__(self, ontology: Optional[Ontology] = None):
        self.ontology = ontology or get_ontology()
        self.state = SemanticState()

    def get_state(self) -> SemanticState:
        return self.state

    def reset(self) -> SemanticState:
        self.state = SemanticState(intent=None, items=[], current_focus_referent=None,
                                   active_attributes={}, quantity=None, negation=False,
                                   polarity="positive", context="active_shop_counter",
                                   dialogue_history=[])
        return self.state

    def update_from_shopkeeper(self, parsed: SemanticState) -> SemanticState:
        raw = (parsed.raw_text or "").lower()
        is_anaphoric = False
        if parsed.items and parsed.items[0].concept in ("THIS", "ITEM"):
            is_anaphoric = True
        elif not parsed.items and parsed.intent in (Intent.QUESTION, Intent.AVAILABILITY, Intent.REQUEST):
            is_anaphoric = True

        if any(w in raw for w in ("how much", "kitna", "kitne", "price", "cost", "daam", "kimat")):
            if self.state.current_focus_referent:
                is_anaphoric = True
                parsed.intent = Intent.QUESTION
                if not any(it.concept == self.state.current_focus_referent for it in parsed.items):
                    parsed.items = [Item(concept=self.state.current_focus_referent, attributes={"query": "PRICE"})]
                else:
                    for it in parsed.items: it.attributes["query"] = "PRICE"

        if any(w in raw for w in ("colour", "color", "rang", "size")):
            if self.state.current_focus_referent:
                is_anaphoric = True
                parsed.intent = Intent.AVAILABILITY
                attr = "colour" if any(w in raw for w in ("colour", "color", "rang")) else "size"
                if not any(it.concept == self.state.current_focus_referent for it in parsed.items):
                    parsed.items = [Item(concept=self.state.current_focus_referent, attributes={"query": attr})]

        if ("aur" in raw or "more" in raw or "another" in raw) and parsed.quantity:
            if self.state.current_focus_referent:
                is_anaphoric = True
                new_qty = (self.state.quantity or 1) + parsed.quantity
                parsed.quantity = new_qty
                parsed.items = [Item(concept=self.state.current_focus_referent, quantity=new_qty)]

        for it in parsed.items:
            if self.ontology.get_category(it.concept) == "product":
                self.state.current_focus_referent = it.concept
                break

        self.state.intent = parsed.intent
        self.state.items = parsed.items
        self.state.negation = parsed.negation
        self.state.polarity = parsed.polarity
        self.state.raw_text = parsed.raw_text
        self.state.language = parsed.language
        if parsed.quantity is not None: self.state.quantity = parsed.quantity
        if parsed.active_attributes: self.state.active_attributes.update(parsed.active_attributes)
        self.state.dialogue_history.append({
            "speaker": "shopkeeper", "text": parsed.raw_text,
            "intent": parsed.intent.value if parsed.intent else None,
            "referent": self.state.current_focus_referent,
            "items": [it.canonical_dict() for it in self.state.items]})
        return self.state

    def update_from_deaf_sign(self, sign_label: str, confidence: float, language: str = "English") -> Dict[str, Any]:
        """Direction B. Maps sign_label via concept_for_sign; applies data-driven effect.
        Raises ValueError if sign_label is unknown."""
        concept_id = self.ontology.concept_for_sign(sign_label)
        if concept_id is None:
            raise ValueError(f"Unknown sign label: {sign_label!r}")

        effect = (self.ontology.concepts.get(concept_id, {}).get("signed_effect") or "product")

        if effect == "product":
            self.state.current_focus_referent = concept_id
            self.state.items = [Item(concept=concept_id)]
            self.state.intent = Intent.REQUEST
        elif effect == "attribute":
            cat = self.ontology.get_category(concept_id) or ""
            if "colour" in cat:
                self.state.active_attributes["colour"] = concept_id
            elif "size" in cat:
                self.state.active_attributes["size"] = concept_id
            else:
                self.state.active_attributes["state"] = concept_id
            if self.state.current_focus_referent:
                focus = self.state.current_focus_referent
                self.state.items = [Item(concept=focus, attributes=dict(self.state.active_attributes))]
                self.state.intent = Intent.REQUEST   # never leave a stale GREET/PAYMENT intent behind
            else:
                # Bare attribute ("red", "cheap") with no product yet: nothing to compose,
                # so clear stale state and let the caller show the sign's own label.
                self.state.items = []
                self.state.intent = None
        elif effect == "price_query":
            self.state.intent = Intent.QUESTION
            focus = self.state.current_focus_referent
            if focus:
                self.state.items = [Item(concept=focus, attributes={**self.state.active_attributes, "query": "PRICE"})]
        elif effect == "greeting":
            self.state.intent = Intent.GREET
        elif effect == "confirm":
            self.state.intent = Intent.CONFIRM
        elif effect == "reject":
            self.state.intent = Intent.REJECT
        elif effect == "payment":
            self.state.intent = Intent.PAYMENT
            self.state.items = [Item(concept=concept_id)]

        from .semantics import NaturalLanguageRealizer
        realization = NaturalLanguageRealizer.realize(self.state, language=language)
        self.state.dialogue_history.append({
            "speaker": "customer_isl", "sign": sign_label, "concept": concept_id,
            "confidence": confidence, "effect": effect,
            "intent": self.state.intent.value if self.state.intent else None,
            "referent": self.state.current_focus_referent})
        return {"realization": realization, "semantic_state": self.state.model_dump()}

    def resolve_clarification(self, chosen_label: str, language: str = "English") -> Dict[str, Any]:
        return self.update_from_deaf_sign(chosen_label, confidence=1.0, language=language)