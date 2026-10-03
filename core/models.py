"""
Pydantic data models for ClarifySign Semantic Core.
Decoupled from web frameworks and transports.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class Intent(str, Enum):
    REQUEST = "REQUEST"
    QUESTION = "QUESTION"
    CONFIRM = "CONFIRM"
    REJECT = "REJECT"
    GREET = "GREET"
    INFORM = "INFORM"
    PAYMENT = "PAYMENT"
    AVAILABILITY = "AVAILABILITY"
    DIRECTION = "DIRECTION"
    UNKNOWN = "UNKNOWN"


class Item(BaseModel):
    concept: str = Field(..., description="Ontology concept ID, e.g. WATER, SHIRT")
    container: Optional[str] = Field(None, description="Container concept, e.g. BOTTLE, PACKET")
    quantity: Optional[int] = Field(None, description="Item quantity, e.g. 1, 2")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Attributes like colour, size, price")

    def canonical_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept.upper(),
            "container": self.container.upper() if self.container else None,
            "quantity": self.quantity,
            "attributes": {k.lower(): (v.upper() if isinstance(v, str) else v) for k, v in sorted(self.attributes.items())}
        }


class Utterance(BaseModel):
    text: str = Field(..., description="Raw input text")
    language: str = Field("English", description="Input language: English, Hindi, Hinglish, Tamil")
    intent: Intent = Field(Intent.UNKNOWN, description="Dialogue intent")
    items: List[Item] = Field(default_factory=list, description="Extracted domain items")
    negation: bool = Field(False, description="Whether utterance contains negation")
    polarity: str = Field("positive", description="Polarity: positive, negative, neutral")
    referent: Optional[str] = Field(None, description="Explicit referent or pronoun indicator")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="NLU parsing confidence")


class SemanticState(BaseModel):
    intent: Optional[Intent] = Field(None, description="Current dialogue intent")
    items: List[Item] = Field(default_factory=list, description="Active items in focus")
    current_focus_referent: Optional[str] = Field(None, description="Active referent concept in dialogue memory")
    active_attributes: Dict[str, Any] = Field(default_factory=dict, description="Active attributes in focus")
    quantity: Optional[int] = Field(None, description="Focused quantity")
    negation: bool = Field(False, description="Whether negation applies")
    polarity: str = Field("positive", description="Polarity of current state")
    context: str = Field("active_shop_counter", description="Domain dialogue context")
    dialogue_history: List[Dict[str, Any]] = Field(default_factory=list, description="Multi-turn history")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Overall confidence score")
    raw_text: Optional[str] = Field(None, description="Last parsed text")
    language: str = Field("English", description="Active conversation language")

    def canonical_signature(self) -> Dict[str, Any]:
        """Returns a canonical dictionary used to verify paraphrase equivalence."""
        canonical_items = [it.canonical_dict() for it in self.items]
        canonical_items.sort(key=lambda x: (x["concept"], str(x["container"]), str(x["quantity"])))
        return {
            "intent": self.intent.value if self.intent else None,
            "items": canonical_items,
            "referent": self.current_focus_referent.upper() if self.current_focus_referent else None,
            "negation": self.negation,
            "attributes": {k.lower(): (v.upper() if isinstance(v, str) else v) for k, v in sorted(self.active_attributes.items())}
        }
