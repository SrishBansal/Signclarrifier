# Clarify core package public API
from .candidates import candidates, uncertainty, H
from .policy import Dialogue, update_posterior
from .planner import plan, info_gain, _question_models, _likelihood, context
from .config import load_config

__all__ = [
    "candidates", "uncertainty", "H",
    "Dialogue", "update_posterior",
    "plan", "info_gain", "_question_models", "_likelihood", "context",
    "load_config",
]
