
"""NL realizer. All labels and templates come from ontology.yaml; no inline dicts.
Translations are machine-authored drafts: have a native speaker review before claiming support."""
from typing import Dict, Any
from .ontology import get_ontology, Ontology
from .models import SemanticState, Intent


class NaturalLanguageRealizer:
    @staticmethod
    def realize(state: SemanticState, language: str = "English",
                ontology: Ontology = None, perspective: str = "shopkeeper") -> Dict[str, Any]:
        """Realize a SemanticState as natural language text.

        Args:
            perspective: "shopkeeper" (Direction A, default) uses templates;
                         "customer" (Direction B) uses customer_templates.
        """
        ont = ontology or get_ontology()
        lang_map = {entry["name"].lower(): entry["code"] for entry in ont.languages()}
        lc = lang_map.get(language.lower(), language.lower())

        if perspective == "customer":
            tmpls = ont.customer_templates.get(lc, ont.customer_templates.get("en", {}))
        else:
            tmpls = ont.templates.get(lc, ont.templates.get("en", {}))

        def label(cid):
            try: return ont.label(cid, lc)
            except KeyError:
                try: return ont.label(cid, "en")
                except KeyError: return cid.lower()

        intent = state.intent
        items = state.items or []

        if intent == Intent.GREET:
            text = tmpls.get("GREET", "Hello.")
        elif intent == Intent.CONFIRM:
            text = tmpls.get("CONFIRM", "Yes.")
        elif intent == Intent.REJECT:
            text = tmpls.get("REJECT", "No.")
        elif intent == Intent.PAYMENT:
            method = label(items[0].concept) if items else "payment"
            text = tmpls.get("PAYMENT", "Payment: {method}.").format(method=method)
        elif intent == Intent.QUESTION:
            product = label(state.current_focus_referent or (items[0].concept if items else "item"))
            text = tmpls.get("PRICE_Q", "How much is this {product}?").format(product=product)
        elif intent == Intent.AVAILABILITY:
            product = label(items[0].concept) if items else label(state.current_focus_referent or "item")
            text = tmpls.get("AVAIL_Q", "Is {product} available?").format(product=product)
        elif intent == Intent.REQUEST and items:
            it = items[0]
            product = label(it.concept)
            qty = it.quantity or ""
            container = label(it.container) if it.container else ""
            attrs = it.attributes or {}
            if "colour" in attrs:
                product = label(attrs["colour"]) + " " + product
            if "size" in attrs:
                product = label(attrs["size"]) + " " + product
            text = tmpls.get("REQUEST", "Give {quantity} {container} {product}.").format(
                quantity=qty, container=container, product=product).strip()
            text = " ".join(text.split())
        else:
            ref = state.current_focus_referent
            product = label(ref) if ref else ""
            text = tmpls.get("DEFAULT", "{product}").format(product=product)

        return {"text": text, "language": language, "intent": intent.value if intent else None}
