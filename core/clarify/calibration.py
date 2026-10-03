# Calibration utilities for clarification

import math
import numpy as np
from typing import List, Tuple, Dict


def temperature_scale(probs: List[float], temperature: float) -> List[float]:
    """Apply temperature scaling to a probability distribution.
    ``probs`` should sum to 1. Returns a new list of probabilities.
    """
    probs = np.array(probs, dtype=float)
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    exp_logits = np.exp(logits - np.max(logits))
    return (exp_logits / exp_logits.sum()).tolist()


def calibrate_probabilities(probs: List[float], heldout: List[Tuple[List[float], int]]) -> Tuple[List[float], float]:
    """Fit a temperature on a held‑out set.

    ``heldout`` is a list of ``(prob_vector, true_index)`` pairs.
    Returns the calibrated probability vector for ``probs`` and the fitted temperature.
    """
    best_temp = 1.0
    best_nll = math.inf
    for t in np.linspace(0.5, 2.0, 16):
        nll = 0.0
        for p_vec, true_idx in heldout:
            p_cal = temperature_scale(p_vec, t)
            nll -= math.log(max(p_cal[true_idx], 1e-12))
        if nll < best_nll:
            best_nll = nll
            best_temp = t
    calibrated = temperature_scale(probs, best_temp)
    return calibrated, best_temp


def compute_ece(probs: List[float], true_idx: int, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) for a single prediction.
    ``probs`` is a probability vector, ``true_idx`` the index of the true class.
    """
    confidence = max(probs)
    pred = int(np.argmax(probs))
    bin_idx = int(confidence * n_bins)
    if bin_idx == n_bins:
        bin_idx -= 1
    accuracy = 1.0 if pred == true_idx else 0.0
    return abs(confidence - accuracy)

__all__ = ["temperature_scale", "calibrate_probabilities", "compute_ece"]
