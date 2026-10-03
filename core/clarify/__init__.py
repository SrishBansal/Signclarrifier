# Clarify core package public API

from .candidates import candidates, uncertainty, H
from .calibration import calibrate_probabilities, compute_ece
from .policy import Dialogue
from .planner import plan, info_gain, _question_models, _likelihood, context
from .resolution import update_posterior, resolve_dialogue

__all__ = [
    'candidates',
    'uncertainty',
    'H',
    'calibrate_probabilities',
    'compute_ece',
    'Dialogue',
    'plan',
    'info_gain',
    '_question_models',
    '_likelihood',
    'context',
    'update_posterior',
    'resolve_dialogue',
]
