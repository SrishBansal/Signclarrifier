# Candidates utilities for clarification

import math
from typing import List, Tuple, Dict, Any


def H(p: List[float]) -> float:
    """Entropy (in bits) of a probability distribution."""
    return -sum(x * math.log2(x) for x in p if x > 0)


def candidates(probs: List[float], classes: List[str], k: int = None) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    """Return top‑k candidate meanings.

    Returns a dict mapping class -> normalized probability (mass of top‑k only) and
    a list of (class, original probability) pairs for the top‑k.
    """
    k = k or 6
    idx = sorted(range(len(classes)), key=lambda i: -probs[i])[:k]
    mass = sum(probs[i] for i in idx) or 1.0
    cand = {classes[i]: probs[i] / mass for i in idx}
    ranked = [(classes[i], probs[i]) for i in idx]
    return cand, ranked


def uncertainty(probs: List[float]) -> Dict[str, float]:
    """Return basic uncertainty metrics used by the policy."""
    s = sorted(probs, reverse=True)
    n = len(probs)
    top1 = s[0]
    margin = top1 - (s[1] if n > 1 else 0.0)
    ent = H(probs)
    ent_norm = ent / math.log2(n) if n > 1 else 0.0
    return {"top1": top1, "margin": margin, "entropy": ent, "entropy_norm": ent_norm}
