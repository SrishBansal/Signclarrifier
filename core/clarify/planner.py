# Planner utilities for clarification

import math
from typing import List, Dict, Any, Tuple

from .candidates import H
from .config import load_config

# Load config for default parameters (may be overridden)
CFG = load_config()


def _likelihood(options: List[str], truth_option: str, reliability: float) -> Dict[str, float]:
    """P(answer=a | truth) for each option a.
    Reliability is the probability of giving the correct answer.
    """
    m = len(options)
    if m == 1:
        return {options[0]: 1.0}
    return {a: (reliability if a == truth_option else (1 - reliability) / max(m - 1, 1)) for a in options}


def _question_models(cand: Dict[str, float], asked: List[str]) -> Dict[str, Tuple[str, List[str], Any, Dict[str, str]]]:
    """Return {qid: (text, options, truth_fn, labels)}.
    Uses CONCEPTS, KIND_EN, EN from semantics for English labels.
    """
    from .semantics import CONCEPTS, KIND_EN, EN
    qs: Dict[str, Tuple[str, List[str], Any, Dict[str, str]]] = {}
    kinds = sorted({CONCEPTS[c] for c in cand})
    if "kind" not in asked and len(kinds) > 1:
        qs["kind"] = ("What kind of word was it?", kinds, lambda c: CONCEPTS[c], {k: KIND_EN[k] for k in kinds})
    top = sorted(cand, key=lambda c: -cand[c])
    if len(top) >= 2 and "pair" not in asked:
        a, b = top[0], top[1]
        opts = [a, b, "neither"]
        qs["pair"] = (f"Did you mean '{EN[a]}' or '{EN[b]}'?", opts,
                       lambda c, a=a, b=b: c if c in (a, b) else "neither",
                       {a: EN[a], b: EN[b], "neither": "Neither"})
    return qs


def context(qid: str, state: Any = None) -> float:
    """Placeholder context hook – returns 0.0 for now."""
    return 0.0


def info_gain(cand: Dict[str, float], options: List[str], truth_fn: Any, reliability: float) -> float:
    """Expected information gain for a question.
    Implements the KL‑based reduction in entropy.
    """
    prior = list(cand.values())
    pa: Dict[str, float] = {a: 0.0 for a in options}
    post: Dict[str, Dict[str, float]] = {a: {} for a in options}
    for c, p in cand.items():
        lik = _likelihood(options, truth_fn(c), reliability)
        for a in options:
            pa[a] += p * lik[a]
            post[a][c] = p * lik[a]
    exp_h = 0.0
    for a in options:
        if pa[a] > 0:
            exp_h += pa[a] * H([v / pa[a] for v in post[a].values()])
    return H(prior) - exp_h


def plan(cand: Dict[str, float], asked: List[str], cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """Select the best clarification question according to EIG minus cost.
    Returns a dict with fields id, text, options, labels, ig, utility, truth.
    """
    cfg = cfg or CFG
    best: Dict[str, Any] = None
    for qid, (text, opts, truth, labels) in _question_models(cand, asked).items():
        ig = info_gain(cand, opts, truth, cfg.get("reliability", 0.95))
        answerability = 1.0 if len(opts) <= 3 else 0.7
        utility = ig + cfg.get("alpha", 0.0) * context(qid, None) + cfg.get("beta", 0.2) * answerability - cfg.get("lam", 0.1) * len(opts)
        if best is None or utility > best["utility"]:
            best = {
                "id": qid,
                "text": text,
                "options": opts,
                "labels": labels,
                "ig": ig,
                "utility": utility,
                "truth": truth,
            }
    return best

