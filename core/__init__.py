"""
ClarifySign Core.
Decoupled modules:
- sign_library: keyframe sequence library and build tools
- renderer: avatar renderer and playback API
- models, ontology, nlu, planner, dialogue, semantics (lazy loaded to prevent unintended coupling)
"""

# Common light models
from .models import Intent, Item, Utterance, SemanticState

__all__ = [
    "Intent",
    "Item",
    "Utterance",
    "SemanticState",
    "Ontology",
    "get_ontology",
    "NLUParser",
    "ISLPlanner",
    "get_planner",
    "DialogueManager",
    "NaturalLanguageRealizer",
    "SignLibrary",
    "AvatarRenderer",
]


def __getattr__(name: str):
    if name in ("Ontology", "get_ontology"):
        from .ontology import Ontology, get_ontology
        return Ontology if name == "Ontology" else get_ontology
    if name == "NLUParser":
        from .nlu import NLUParser
        return NLUParser
    if name in ("ISLPlanner", "get_planner"):
        from .planner import ISLPlanner, get_planner
        return ISLPlanner if name == "ISLPlanner" else get_planner
    if name == "DialogueManager":
        from .dialogue import DialogueManager
        return DialogueManager
    if name == "NaturalLanguageRealizer":
        from .semantics import NaturalLanguageRealizer
        return NaturalLanguageRealizer
    if name in ("SignLibrary", "SignTimeline"):
        from .sign_library import SignLibrary, SignTimeline
        return SignLibrary if name == "SignLibrary" else SignTimeline
    if name == "AvatarRenderer":
        from .renderer import AvatarRenderer
        return AvatarRenderer
    raise AttributeError(f"module 'core' has no attribute '{name}'")

