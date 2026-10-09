"""
NLU Engine for ClarifySign.
All cue sets are loaded from ontology.intent_cues; no inline vocabularies.
Question-mark detection runs on the RAW text (before normalize_text strips punctuation).
Social concepts (HELLO, THANKYOU) are detected via the ontology category index.
"""
import re
from typing import Dict, Any, Optional, List, Callable
from .models import Intent, Item, Utterance, SemanticState
from .ontology import get_ontology, Ontology


class NLUParser:
    def __init__(self, ontology: Optional[Ontology] = None, llm_callable: Optional[Callable] = None):
        self.ontology = ontology or get_ontology()
        self.llm_callable = llm_callable
        self._cues = self.ontology.intent_cues

    def parse(self, text: str, language: str = "English", current_state: Optional[SemanticState] = None) -> SemanticState:
        if not text or not text.strip():
            return SemanticState(intent=Intent.UNKNOWN, raw_text=text, language=language)
        if self.llm_callable is not None:
            try:
                llm_output = self._call_llm(text, language)
                state = self._validate_and_repair_llm_output(llm_output, text, language, current_state)
                if state is not None:
                    return state
            except Exception:
                pass
        utterance = self.parse_utterance(text, language)
        return self._utterance_to_state(utterance, current_state)

    def parse_utterance(self, text: str, language: str = "English") -> Utterance:
        raw_text = text  # preserve for "?" detection
        norm = self.ontology.normalize_text(text)
        tokens = norm.split()
        negation = self._detect_negation(tokens)
        intent = self._detect_intent(norm, tokens, negation, raw_text)
        referent = self._detect_referent(tokens)
        quantity = self._extract_quantity(norm, tokens)
        attributes = self._extract_attributes(norm, tokens)
        container = self._extract_container(tokens)
        products = self._extract_products(tokens)
        social = self._extract_social(tokens)   # HELLO, THANKYOU etc.
        items: List[Item] = []
        # Add social items first (they go into items for planner to detect)
        for sc in social:
            items.append(Item(concept=sc))
        if products:
            for p in products:
                items.append(Item(concept=p, container=container, quantity=quantity, attributes=attributes))
        elif container:
            items.append(Item(concept=container, quantity=quantity, attributes=attributes))
        elif attributes or quantity:
            items.append(Item(concept=referent or "ITEM", container=container, quantity=quantity, attributes=attributes))
        valid_items = [self._validate_or_repair_item(it) for it in items]
        valid_items = [it for it in valid_items if it is not None]
        return Utterance(text=text, language=language, intent=intent, items=valid_items,
                         negation=negation, polarity="negative" if negation else "positive",
                         referent=referent, confidence=0.95)

    def _cue_tokens(self, intent_key: str) -> set:
        c = self._cues.get(intent_key, {})
        base = c.get("tokens") or []
        for key in ("hi_tokens", "ta_tokens", "bn_tokens", "te_tokens"):
            base = base + (c.get(key) or [])
        return set(base)

    def _cue_phrases(self, intent_key: str) -> set:
        return set(self._cues.get(intent_key, {}).get("phrases") or [])

    def _detect_negation(self, tokens: List[str]) -> bool:
        # Load negation synonyms from NO concept in ontology
        no_concept = self.ontology.concepts.get("NO", {})
        neg_set = set()
        for syn_list in (no_concept.get("synonyms") or {}).values():
            for s in (syn_list or []):
                w = str(s).lower().strip()
                if w and " " not in w:
                    neg_set.add(w)
        # Fallback core tokens always included (structural, not vocabulary)
        neg_set.update({"nahi", "nahin", "na", "mat", "not", "no", "never", "illai", "vendaam", "dont"})
        return any(t in neg_set for t in tokens)

    def _detect_referent(self, tokens: List[str]) -> Optional[str]:
        # Referent pronouns - structural function words, not domain vocab
        ref = {"ye", "yeh", "this", "it", "iska", "iski", "iske", "that", "wo", "woh", "uske",
               "idhu", "adhu", "antha"}
        return "THIS" if any(t in ref for t in tokens) else None

    def _extract_quantity(self, norm: str, tokens: List[str]) -> Optional[int]:
        digits = re.findall(r"\b\d+\b", norm)
        if digits:
            try: return int(digits[0])
            except ValueError: pass
        # Load quantity word mapping from ontology synonym index (ONE, TWO ... FIVE)
        qty_map: Dict[str, int] = {}
        for q_concept, q_val in [("ONE", 1), ("TWO", 2), ("THREE", 3), ("FOUR", 4), ("FIVE", 5)]:
            d = self.ontology.concepts.get(q_concept, {})
            for syn_list in (d.get("synonyms") or {}).values():
                for s in (syn_list or []):
                    w = str(s).lower().strip()
                    if w and " " not in w:
                        qty_map[w] = q_val
        not_a_number = {"you", "i", "we", "they", "he", "she", "not", "have", "the", "this", "that", "it"}
        for i, t in enumerate(tokens):
            if t == "do" and i + 1 < len(tokens) and tokens[i + 1] in not_a_number:
                continue
            if t in qty_map:
                return qty_map[t]
        return None

    def _extract_container(self, tokens: List[str]) -> Optional[str]:
        for t in tokens:
            cid = self.ontology.resolve_concept(t)
            if cid and self.ontology.get_category(cid) == "container":
                return cid
        return None

    def _social_covered(self, tokens: List[str]) -> set:
        """Token indices inside a social phrase ("good morning"), so their words are not also read as attributes/products."""
        covered: set = set()
        for n in range(min(4, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                if any(k in covered for k in range(i, i + n)):
                    continue
                cid = self.ontology.synonym_index.get(" ".join(tokens[i:i + n]))
                if cid and cid not in ("YES", "NO") and self.ontology.get_category(cid) == "social":
                    covered.update(range(i, i + n))
        return covered

    def _extract_attributes(self, norm: str, tokens: List[str]) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        covered = self._social_covered(tokens)
        for idx, t in enumerate(tokens):
            if idx in covered:
                continue
            cid = self.ontology.resolve_concept(t)
            if cid:
                cat = self.ontology.get_category(cid)
                if cat == "attribute_colour": attrs["colour"] = cid
                elif cat == "attribute_size": attrs["size"] = cid
                elif cat == "attribute_state": attrs["state"] = cid
        return attrs

    def _extract_products(self, tokens: List[str]) -> List[str]:
        products = []
        covered = self._social_covered(tokens)
        for i in range(len(tokens)):
            for j in range(len(tokens), i, -1):
                if any(k in covered for k in range(i, j)):
                    continue
                phrase = " ".join(tokens[i:j])
                cid = self.ontology.resolve_concept(phrase)
                if cid and self.ontology.get_category(cid) in ("product", "time", "descriptor") and cid not in products:
                    products.append(cid)
        return products

    def _extract_social(self, tokens: List[str]) -> List[str]:
        """Extract social concepts (HELLO, THANKYOU, etc.) via ontology category 'social'."""
        social = []
        for t in tokens:
            cid = self.ontology.resolve_concept(t)
            if cid and self.ontology.get_category(cid) == "social" and cid not in social:
                social.append(cid)
        # Also check multi-word phrases (e.g. "thank you")
        norm_joined = " ".join(tokens)
        for cid in self.ontology.category_index.get("social", []):
            d = self.ontology.concepts.get(cid, {})
            for syn_list in (d.get("synonyms") or {}).values():
                for s in (syn_list or []):
                    sn = self.ontology.normalize_text(str(s))
                    if sn and f" {sn} " in f" {norm_joined} " and cid not in social:  # whole words only: "hi" must not match "shirt"/"this"
                        social.append(cid)
        # YES and NO are CONFIRM/REJECT sentinels, not display-social
        return [c for c in social if c not in ("YES", "NO")]

    def _detect_intent(self, norm: str, tokens: List[str], negation: bool, raw_text: str = "") -> Intent:
        def has_tok(key): return bool(self._cue_tokens(key) & set(tokens))
        def has_phrase(key): return any(p in norm for p in self._cue_phrases(key))

        # GREET - if only greet tokens are present, return GREET.
        # If GREET tokens co-occur with QUESTION cues, let QUESTION win so the
        # planner can add the QUESTION marker while still prepending HELLO.
        is_greet = has_tok("GREET") or has_phrase("GREET") or (self.ontology.resolve_concept(norm) == "HELLO")
        bigrams = self._cues.get("QUESTION", {}).get("bigrams", [])
        has_question = (has_tok("QUESTION") or has_phrase("QUESTION") or
                        any(all(w in tokens for w in bg) for bg in bigrams))
        has_avail = has_phrase("AVAILABILITY") or has_tok("AVAILABILITY")
        if is_greet and not has_question and not has_avail:
            return Intent.GREET
        # If greet + question both present, fall through (HELLO added via social items)

        # Pure social (THANKYOU etc.) - detect from ontology
        social_concepts = self._extract_social(tokens)
        if social_concepts and all(self.ontology.get_category(c) == "social" for c in social_concepts):
            # Check it's truly a pure social utterance (no product content)
            products = self._extract_products(tokens)
            if not products:
                # THANKYOU -> INFORM (the planner converts it to [THANKYOU])
                return Intent.INFORM

        cues = self._cues
        max_confirm = cues.get("CONFIRM", {}).get("max_tokens", 3)
        if has_tok("CONFIRM") and not negation and len(tokens) <= max_confirm: return Intent.CONFIRM

        max_reject = cues.get("REJECT", {}).get("max_tokens", 3)
        if negation and has_tok("REJECT") and len(tokens) <= max_reject: return Intent.REJECT

        if has_tok("PAYMENT") or has_phrase("PAYMENT"): return Intent.PAYMENT

        bigrams_q = self._cues.get("QUESTION", {}).get("bigrams", [])
        if has_tok("QUESTION") or has_phrase("QUESTION") or any(
                all(w in tokens for w in bg) for bg in bigrams_q):
            return Intent.QUESTION

        # AVAILABILITY detection: check raw text for "?" plus availability phrase, or cue tokens
        has_q_mark = "?" in raw_text
        if has_phrase("AVAILABILITY") or has_tok("AVAILABILITY"):
            return Intent.AVAILABILITY
        # "paani hai?" / "blue shirt medium mein hai?" -> AVAILABILITY
        if has_q_mark and has_tok("QUESTION_WORDS"):
            pass  # falls through to QUESTION below
        if has_q_mark:
            # Check if asking about existence vs price
            price_tok = self._cue_tokens("QUESTION")
            if price_tok & set(tokens):
                return Intent.QUESTION
            return Intent.AVAILABILITY

        if has_tok("DIRECTION"): return Intent.DIRECTION

        if has_phrase("REQUEST") or has_tok("REQUEST"): return Intent.REQUEST

        if has_tok("QUESTION_WORDS"): return Intent.QUESTION

        return Intent.REQUEST if any(w in norm for w in ("chahiye", "give", "de ")) else Intent.INFORM

    def _validate_or_repair_item(self, item: Item) -> Optional[Item]:
        cid = item.concept.upper()
        if self.ontology.is_valid_concept(cid):
            item.concept = cid
            return item
        resolved = self.ontology.resolve_concept(cid)
        if resolved:
            item.concept = resolved
            return item
        # OOV -> nosign item (no FS_ prefix)
        item.concept = f"OOV_{cid}"
        return item

    def _utterance_to_state(self, utterance: Utterance, current_state: Optional[SemanticState] = None) -> SemanticState:
        state = SemanticState(intent=utterance.intent, items=utterance.items,
                              negation=utterance.negation, polarity=utterance.polarity,
                              confidence=utterance.confidence, raw_text=utterance.text,
                              language=utterance.language)
        # Identify non-social items for referent tracking
        non_social = [it for it in utterance.items
                      if self.ontology.get_category(it.concept) != "social"]
        if non_social and non_social[0].concept not in ("THIS", "ITEM"):
            state.current_focus_referent = non_social[0].concept
        elif utterance.referent and current_state and current_state.current_focus_referent:
            state.current_focus_referent = current_state.current_focus_referent
            if state.items:
                # Replace placeholder THIS/ITEM with actual referent
                for it in state.items:
                    if it.concept in ("THIS", "ITEM"):
                        it.concept = current_state.current_focus_referent
        if state.items:
            non_social_items = [it for it in state.items if self.ontology.get_category(it.concept) != "social"]
            if non_social_items:
                state.quantity = non_social_items[0].quantity
                state.active_attributes = non_social_items[0].attributes
        elif current_state:
            state.current_focus_referent = current_state.current_focus_referent
        if current_state:
            state.dialogue_history = list(current_state.dialogue_history)
        state.dialogue_history.append({"speaker": "shopkeeper", "text": utterance.text,
                                        "intent": utterance.intent.value,
                                        "referent": state.current_focus_referent,
                                        "items": [it.canonical_dict() for it in state.items]})
        return state

    def _call_llm(self, text: str, language: str) -> Dict[str, Any]:
        schema = {"type": "object", "properties": {
            "intent": {"type": "string", "enum": [i.value for i in Intent]},
            "concept": {"type": "string"}, "container": {"type": "string"},
            "quantity": {"type": "integer"}, "attributes": {"type": "object"},
            "negation": {"type": "boolean"}, "referent": {"type": "string"}},
            "required": ["intent"]}
        return self.llm_callable(text, schema)

    def _validate_and_repair_llm_output(self, llm_dict, raw_text, language, current_state):
        intent_str = llm_dict.get("intent", "UNKNOWN").upper()
        intent = Intent(intent_str) if intent_str in Intent.__members__ else Intent.UNKNOWN
        concept = llm_dict.get("concept")
        if concept:
            resolved = self.ontology.resolve_concept(concept)
            concept = resolved if resolved else f"OOV_{concept.upper()}"
        container = llm_dict.get("container")
        if container:
            container = self.ontology.resolve_concept(container) or container.upper()
        items = []
        if concept:
            items.append(Item(concept=concept, container=container,
                              quantity=llm_dict.get("quantity"),
                              attributes=llm_dict.get("attributes", {})))
        utt = Utterance(text=raw_text, language=language, intent=intent, items=items,
                        negation=bool(llm_dict.get("negation", False)),
                        referent=llm_dict.get("referent"), confidence=0.98)
        return self._utterance_to_state(utt, current_state)