"""
Natural Language Understanding (NLU) Engine for ClarifySign.
Implements structured-output parsing with ontology validation and offline rule/fuzzy fallback.
Has NO web or framework dependencies.
"""

import re
from typing import Dict, Any, Optional, List, Callable
from .models import Intent, Item, Utterance, SemanticState
from .ontology import get_ontology, Ontology


class NLUParser:
    """Hybrid NLU parser combining rule-based semantic extraction with ontology constraints."""

    def __init__(self, ontology: Optional[Ontology] = None, llm_callable: Optional[Callable] = None):
        self.ontology = ontology or get_ontology()
        self.llm_callable = llm_callable

    def parse(self, text: str, language: str = "English", current_state: Optional[SemanticState] = None) -> SemanticState:
        """Main parsing entrypoint: returns a validated SemanticState."""
        if not text or not text.strip():
            return SemanticState(intent=Intent.UNKNOWN, raw_text=text, language=language)

        # 1. If LLM hook is provided, attempt structured output
        if self.llm_callable is not None:
            try:
                llm_output = self._call_llm(text, language)
                state = self._validate_and_repair_llm_output(llm_output, text, language, current_state)
                if state is not None:
                    return state
            except Exception as e:
                # Graceful offline fallback
                pass

        # 2. Robust offline rule/fuzzy engine
        utterance = self.parse_utterance(text, language)
        return self._utterance_to_state(utterance, current_state)

    def parse_utterance(self, text: str, language: str = "English") -> Utterance:
        """Extracts intent, items, negation, and referents from raw text."""
        norm_text = self.ontology.normalize_text(text)
        tokens = norm_text.split()

        # Negation check
        negation = self._detect_negation(tokens)
        polarity = "negative" if negation else "positive"

        # Intent detection
        intent = self._detect_intent(norm_text, tokens, negation)

        # Referent detection (anaphoric / demonstrative)
        referent = self._detect_referent(tokens)

        # Extract quantities
        quantity = self._extract_quantity(norm_text, tokens)

        # Extract attributes (colours, sizes, states)
        attributes = self._extract_attributes(norm_text, tokens)

        # Extract containers
        container = self._extract_container(tokens)

        # Extract products / domain concepts
        products = self._extract_products(tokens)

        items: List[Item] = []
        if products:
            for p in products:
                items.append(Item(
                    concept=p,
                    container=container,
                    quantity=quantity,
                    attributes=attributes
                ))
        elif container:
            items.append(Item(
                concept=container,
                container=None,
                quantity=quantity,
                attributes=attributes
            ))
        elif attributes or quantity:
            # Item with only attributes/quantity (referent-based)
            items.append(Item(
                concept=referent or "ITEM",
                container=container,
                quantity=quantity,
                attributes=attributes
            ))

        # Validate / repair items against ontology
        valid_items = []
        for it in items:
            repaired_item = self._validate_or_repair_item(it)
            if repaired_item:
                valid_items.append(repaired_item)

        return Utterance(
            text=text,
            language=language,
            intent=intent,
            items=valid_items,
            negation=negation,
            polarity=polarity,
            referent=referent,
            confidence=0.95
        )

    def _detect_negation(self, tokens: List[str]) -> bool:
        neg_words = {"nahi", "nahin", "na", "mat", "not", "no", "never", "illai", "vendaam", "don't", "dont", "नहीं", "मत", "இல்லை"}
        return any(t in neg_words for t in tokens)

    def _detect_referent(self, tokens: List[str]) -> Optional[str]:
        ref_words = {"ye", "yeh", "this", "it", "iska", "iski", "iske", "that", "wo", "woh", "uske", "idhu", "adhu", "antha", "இது", "அது"}
        for t in tokens:
            if t in ref_words:
                return "THIS"
        return None

    def _extract_quantity(self, norm_text: str, tokens: List[str]) -> Optional[int]:
        # Numeric digits
        digits = re.findall(r"\b\d+\b", norm_text)
        if digits:
            try:
                return int(digits[0])
            except ValueError:
                pass
        
        # Word mapping
        qty_map = {
            "ek": 1, "one": 1, "oru": 1, "onnu": 1, "एक": 1, "ஒன்று": 1,
            "do": 2, "two": 2, "rendu": 2, "दो": 2, "இரண்டு": 2,
            "teen": 3, "three": 3, "moonu": 3, "तीन": 3, "மூன்று": 3,
            "chaar": 4, "char": 4, "four": 4, "naalu": 4, "चार": 4, "நான்கு": 4,
            "paanch": 5, "panch": 5, "five": 5, "anju": 5, "पांच": 5, "ஐந்து": 5,
        }
        for t in tokens:
            if t in qty_map:
                return qty_map[t]
        return None

    def _extract_container(self, tokens: List[str]) -> Optional[str]:
        for t in tokens:
            cid = self.ontology.resolve_concept(t)
            if cid and self.ontology.get_category(cid) == "container":
                return cid
        return None

    def _extract_attributes(self, norm_text: str, tokens: List[str]) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        for t in tokens:
            cid = self.ontology.resolve_concept(t)
            if cid:
                cat = self.ontology.get_category(cid)
                if cat == "attribute_colour":
                    attrs["colour"] = cid
                elif cat == "attribute_size":
                    attrs["size"] = cid
                elif cat == "attribute_state":
                    attrs["state"] = cid
        return attrs

    def _extract_products(self, tokens: List[str]) -> List[str]:
        products = []
        # Multi-word checks first
        for i in range(len(tokens)):
            for j in range(len(tokens), i, -1):
                phrase = " ".join(tokens[i:j])
                cid = self.ontology.resolve_concept(phrase)
                if cid and self.ontology.get_category(cid) == "product":
                    if cid not in products:
                        products.append(cid)
        return products

    def _detect_intent(self, norm_text: str, tokens: List[str], negation: bool) -> Intent:
        # Greetings (token + multi-word)
        greet_cues = {"hello", "hi", "hey", "namaste", "namaskar", "pranam", "vanakkam", "வணக்கம்"}
        if any(t in greet_cues for t in tokens):
            return Intent.GREET
        greet_phrases = {"good morning", "good afternoon", "good evening", "shubh prabhat"}
        if any(phrase in norm_text for phrase in greet_phrases):
            return Intent.GREET
        # Ontology-resolved greeting
        resolved = self.ontology.resolve_concept(norm_text)
        if resolved == "HELLO":
            return Intent.GREET

        # Confirmations
        confirm_cues = {"haan", "ha", "yes", "theek", "sahi", "ok", "okay", "sure", "correct", "aam", "sari", "fine"}
        if any(t in confirm_cues for t in tokens) and not negation and len(tokens) <= 3:
            return Intent.CONFIRM

        # Rejections
        if negation and any(t in {"nahi", "no", "cancel", "never", "vendaam", "mat"} for t in tokens) and len(tokens) <= 3:
            return Intent.REJECT

        # Payment
        pay_cues = {"upi", "qr", "scanner", "scan", "cash", "gpay", "phonepe", "paytm", "paise", "payment", "chhutte", "khulle", "change", "bill", "pay", "பணம்"}
        if any(t in pay_cues for t in tokens):
            return Intent.PAYMENT
        pay_phrases = {"செலுத்த", "பணம் செலுத்த"}
        if any(phrase in norm_text for phrase in pay_phrases):
            return Intent.PAYMENT

        # Price / Cost Question
        cost_cues = {"kitna", "kitne", "price", "cost", "daam", "kimat", "rate", "how much", "evvalavu", "விலை"}
        if any(t in cost_cues for t in tokens) or ("how" in tokens and "much" in tokens):
            return Intent.QUESTION

        # Availability
        avail_cues = {"milega", "available", "stock", "hai kya", "irukka", "இருக்கிறதா"}
        if any(cue in norm_text for cue in avail_cues):
            return Intent.AVAILABILITY

        # Direction / Location Question
        dir_cues = {"kahan", "kidhar", "where", "rasta", "counter", "exit", "engu", "engae"}
        if any(t in dir_cues for t in tokens):
            return Intent.DIRECTION

        # Request / Buy
        req_cues = {"chahiye", "dena", "dedo", "de do", "dijiye", "pack", "want", "give", "buy", "kodu", "thanga", "need", "bottle paani", "please", "show", "dikhao", "dikha", "கொடுங்கள்", "kodungal", "வேண்டும்"}
        if any(cue in norm_text for cue in req_cues):
            return Intent.REQUEST

        # General Question if question mark or question words
        if "?" in norm_text or any(t in {"kya", "what", "which", "kaunsa", "enna"} for t in tokens):
            return Intent.QUESTION

        # If product/items present, default to REQUEST or INFORM
        return Intent.REQUEST if ("chahiye" in norm_text or "de" in tokens or "give" in tokens) else Intent.INFORM

    def _validate_or_repair_item(self, item: Item) -> Optional[Item]:
        """Ensures concept is in ontology; repairs or falls back to fingerspelling."""
        cid = item.concept.upper()
        if self.ontology.is_valid_concept(cid):
            return item
        
        # Try resolving via synonym index
        resolved = self.ontology.resolve_concept(cid)
        if resolved:
            item.concept = resolved
            return item
        
        # Out-of-vocabulary item: mark for fingerspell fallback
        item.concept = f"FS_{cid}"
        return item

    def _utterance_to_state(self, utterance: Utterance, current_state: Optional[SemanticState] = None) -> SemanticState:
        """Constructs and updates SemanticState with multi-turn context."""
        state = SemanticState(
            intent=utterance.intent,
            items=utterance.items,
            negation=utterance.negation,
            polarity=utterance.polarity,
            confidence=utterance.confidence,
            raw_text=utterance.text,
            language=utterance.language
        )

        # Inherit or resolve current referent
        if utterance.items and utterance.items[0].concept not in ("THIS", "ITEM") and not utterance.items[0].concept.startswith("FS_"):
            state.current_focus_referent = utterance.items[0].concept
        elif utterance.referent or (utterance.items and utterance.items[0].concept in ("THIS", "ITEM")):
            if current_state and current_state.current_focus_referent:
                state.current_focus_referent = current_state.current_focus_referent
                # Attach referent to item
                if state.items:
                    state.items[0].concept = current_state.current_focus_referent

        # Inherit quantity or attributes
        if state.items:
            state.quantity = state.items[0].quantity
            state.active_attributes = state.items[0].attributes
        elif current_state:
            state.current_focus_referent = current_state.current_focus_referent

        # Update dialogue history
        if current_state:
            state.dialogue_history = list(current_state.dialogue_history)
        
        state.dialogue_history.append({
            "speaker": "shopkeeper",
            "text": utterance.text,
            "intent": utterance.intent.value,
            "referent": state.current_focus_referent,
            "items": [it.canonical_dict() for it in state.items]
        })

        return state

    def _call_llm(self, text: str, language: str) -> Dict[str, Any]:
        """Schema-constrained LLM output invocation."""
        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": [i.value for i in Intent]},
                "concept": {"type": "string"},
                "container": {"type": "string"},
                "quantity": {"type": "integer"},
                "attributes": {"type": "object"},
                "negation": {"type": "boolean"},
                "referent": {"type": "string"}
            },
            "required": ["intent"]
        }
        return self.llm_callable(text, schema)

    def _validate_and_repair_llm_output(self, llm_dict: Dict[str, Any], raw_text: str, language: str, current_state: Optional[SemanticState]) -> Optional[SemanticState]:
        """Validates LLM output against the ontology."""
        intent_str = llm_dict.get("intent", "UNKNOWN").upper()
        intent = Intent(intent_str) if intent_str in Intent.__members__ else Intent.UNKNOWN

        concept = llm_dict.get("concept")
        if concept:
            resolved = self.ontology.resolve_concept(concept)
            concept = resolved if resolved else f"FS_{concept.upper()}"

        container = llm_dict.get("container")
        if container:
            container = self.ontology.resolve_concept(container) or container.upper()

        items = []
        if concept:
            items.append(Item(
                concept=concept,
                container=container,
                quantity=llm_dict.get("quantity"),
                attributes=llm_dict.get("attributes", {})
            ))

        utt = Utterance(
            text=raw_text,
            language=language,
            intent=intent,
            items=items,
            negation=bool(llm_dict.get("negation", False)),
            referent=llm_dict.get("referent"),
            confidence=0.98
        )
        return self._utterance_to_state(utt, current_state)
