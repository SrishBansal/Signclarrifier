"""
Semantic Interface for ClarifySign Core.
Provides decoupled access to NLUParser, SemanticState, Intent, and NaturalLanguageRealizer.
Has NO web or framework dependencies.
"""

from typing import Dict, Any, Optional, List
from .models import Intent, Item, Utterance, SemanticState
from .nlu import NLUParser
from .ontology import get_ontology, Ontology


class NaturalLanguageRealizer:
    """Generates natural language realizations for Shopkeeper TTS from SemanticState."""

    REALIZATION_MAP = {
        "English": {
            "GREET": "Hello! How can I help you?",
            "CONFIRM": "Yes, certainly.",
            "REJECT": "No, not available.",
            "PRICE_Q": "How much is this {product}?",
            "AVAIL_Q": "Is {product} available?",
            "REQUEST": "Please give {quantity} {container} {product}.",
            "PAYMENT": "I want to pay using {method}.",
            "DEFAULT": "{product}"
        },
        "Hindi": {
            "GREET": "नमस्ते! मैं आपकी क्या मदद कर सकता हूँ?",
            "CONFIRM": "हाँ, बिल्कुल।",
            "REJECT": "नहीं, उपलब्ध नहीं है।",
            "PRICE_Q": "इस {product} का दाम कितना है?",
            "AVAIL_Q": "क्या {product} मिलेगा?",
            "REQUEST": "कृपया {quantity} {container} {product} दीजिए।",
            "PAYMENT": "{method} से भुगतान करना है।",
            "DEFAULT": "{product}"
        },
        "Tamil": {
            "GREET": "வணக்கம்! உங்களுக்கு என்ன வேண்டும்?",
            "CONFIRM": "ஆம், நிச்சயமாக.",
            "REJECT": "இல்லை, கிடைக்காது.",
            "PRICE_Q": "இந்த {product} விலை என்ன?",
            "AVAIL_Q": "{product} இருக்கிறதா?",
            "REQUEST": "தயவுசெய்து {quantity} {container} {product} கொடுங்கள்.",
            "PAYMENT": "{method} மூலம் பணம் செலுத்த வேண்டும்.",
            "DEFAULT": "{product}"
        }
    }

    CONCEPT_LABELS = {
        "English": {
            "WATER": "water", "BOTTLE": "bottle", "SHIRT": "shirt", "PEN": "pen",
            "MILK": "milk", "BREAD": "bread", "SOAP": "soap", "SHOES": "shoes",
            "CELLPHONE": "phone", "PHONE": "phone", "TEA": "tea", "COFFEE": "coffee",
            "RICE": "rice", "SUGAR": "sugar", "EGG": "egg", "BISCUIT": "biscuit",
            "BOOK": "book", "MEDICINE": "medicine", "UPI": "UPI", "CASH": "cash",
            "QR": "QR scanner", "ONE": "one", "TWO": "two", "BLUE": "blue", "RED": "red"
        },
        "Hindi": {
            "WATER": "पानी", "BOTTLE": "बोतल", "SHIRT": "शर्ट", "PEN": "पेन",
            "MILK": "दूध", "BREAD": "ब्रेड", "SOAP": "साबुन", "SHOES": "जूते",
            "CELLPHONE": "फोन", "PHONE": "फोन", "TEA": "चाय", "COFFEE": "कॉफी",
            "RICE": "चावल", "SUGAR": "चीनी", "EGG": "अंडा", "BISCUIT": "बिस्कुट",
            "BOOK": "किताब", "MEDICINE": "दवाई", "UPI": "यूपीआई", "CASH": "नकद",
            "QR": "क्यूआर कोड", "ONE": "एक", "TWO": "दो", "BLUE": "नीला", "RED": "लाल"
        },
        "Tamil": {
            "WATER": "தண்ணீர்", "BOTTLE": "பாட்டில்", "SHIRT": "சட்டை", "PEN": "பேனா",
            "MILK": "பால்", "BREAD": "ரொட்டி", "SOAP": "சோப்", "SHOES": "காலணி",
            "CELLPHONE": "போன்", "PHONE": "போன்", "TEA": "டீ", "COFFEE": "காபி",
            "RICE": "அரிசி", "SUGAR": "சர்க்கரை", "EGG": "முட்டை", "BISCUIT": "பிஸ்கட்",
            "BOOK": "புத்தகம்", "MEDICINE": "மருந்து", "UPI": "யுபிஐ", "CASH": "ரொக்கம்",
            "QR": "ஸ்கேனர்", "ONE": "ஒரு", "TWO": "இரண்டு", "BLUE": "நீலம்", "RED": "சிவப்பு"
        }
    }

    @classmethod
    def realize(cls, state: SemanticState, language: str = "English") -> Dict[str, Any]:
        """Realizes speech string for Shopkeeper TTS."""
        lang_key = language if language in cls.REALIZATION_MAP else "English"
        templates = cls.REALIZATION_MAP[lang_key]
        labels = cls.CONCEPT_LABELS.get(lang_key, cls.CONCEPT_LABELS["English"])

        intent = state.intent or Intent.REQUEST
        product = state.current_focus_referent or "item"
        if state.items and state.items[0].concept not in ("THIS", "ITEM"):
            product = state.items[0].concept
        product_label = labels.get(product.upper(), product.lower())

        container = ""
        quantity = str(state.quantity or "")
        if state.items and state.items[0].container:
            c = state.items[0].container.upper()
            container = labels.get(c, c.lower())

        if intent == Intent.GREET:
            text = templates["GREET"]
        elif intent == Intent.CONFIRM:
            text = templates["CONFIRM"]
        elif intent == Intent.REJECT:
            text = templates["REJECT"]
        elif intent == Intent.QUESTION and (product in ("PRICE", "COST") or "price" in state.active_attributes or any(it.attributes.get("query") == "PRICE" for it in state.items)):
            text = templates["PRICE_Q"].format(product=product_label)
        elif intent == Intent.AVAILABILITY:
            text = templates["AVAIL_Q"].format(product=product_label)
        elif intent == Intent.PAYMENT:
            pay_method = labels.get("UPI", "UPI")
            for it in state.items:
                if it.concept in ("CASH", "UPI", "QR"):
                    pay_method = labels.get(it.concept, it.concept)
                    break
            text = templates["PAYMENT"].format(method=pay_method)
        elif intent == Intent.REQUEST:
            qty_part = quantity if quantity else ("1" if lang_key == "English" else "")
            text = templates["REQUEST"].format(quantity=qty_part, container=container, product=product_label)
            text = " ".join(text.split())
        else:
            text = templates["DEFAULT"].format(product=product_label)

        return {
            "text": text,
            "language": lang_key,
            "intent": intent.value if intent else None,
            "product": product
        }


__all__ = ["NLUParser", "SemanticState", "NaturalLanguageRealizer", "Intent", "Item", "Utterance"]
