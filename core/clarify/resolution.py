# Resolution utilities for clarification

"""Utilities that drive a clarification dialogue using the core modules.

The original monolithic implementation lived in ``app/clarify.py``.  The
functionality is now split across ``candidates``, ``policy`` (which contains the
``Dialogue`` class), and ``planner``.  This module provides a thin convenience
wrapper that orchestrates the dialogue loop so that callers only need to pass a
probability vector and a list of class names.

Typical usage::

    from core.clarify.resolution import resolve_dialogue
    result = resolve_dialogue(probs, classes)
    # ``result`` is a dict with the final action, e.g. ``{"action": "resolved",
    # "concept": "hello", "how": "clarified"}``.
"""

from typing import List, Dict, Any

from .policy import Dialogue
from .config import load_config


def resolve_dialogue(probs: List[float], classes: List[str], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run a full clarification episode.

    * ``probs`` – raw recogniser probabilities for each class (must sum to 1).
    * ``classes`` – ordered list of class identifiers matching ``probs``.
    * ``config`` – optional override of the YAML configuration; if omitted the
      default configuration is loaded via :func:`load_config`.

    The function creates a :class:`~core.clarify.policy.Dialogue` instance, asks
    the first question (if any) and then repeatedly calls ``answer`` with a mock
    answer ``"<skip>"`` until the dialogue resolves.  In the real system the
    answer would come from the user interface; for unit‑testing we provide a
    deterministic fallback that always selects the first option.

    The return value mirrors the structure used throughout the codebase – a
    dictionary containing at least an ``action`` key.
    """
    cfg = config or load_config()
    dlg = Dialogue(probs, classes, cfg)
    state = dlg.start()
    # If the first step is a question we simulate an answer using the first
    # option.  This keeps the function side‑effect‑free for testing.
    while state.get("action") == "ask":
        # Pick the first option value (the real UI would provide the user's
        # response).  ``state["question"]["options"]`` is a list of dicts with a
        # ``value`` key.
        first_opt = state["question"]["options"][0]["value"]
        state = dlg.answer(first_opt)
    return state

__all__ = ["resolve_dialogue"]
