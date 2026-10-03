
# Planner utilities for clarification - reads labels from ontology, no .semantics import
import math
from typing import List, Dict, Any, Tuple
from .candidates import H
from .config import load_config

CFG = load_config()


def _likelihood(options: List[str], truth_option: str, reliability: float) -> Dict[str, float]:
    m = len(options)
    if m == 1:
        return {options[0]: 1.0}
    return {a: (reliability if a == truth_option else (1 - reliability) / max(m - 1, 1)) for a in options}


def _question_models(cand: Dict[str, float], asked: List[str], ontology=None) -> Dict[str, Tuple]:
    if ontology is None:
        from ..ontology import get_ontology
        ontology = get_ontology()

    qs: Dict[str, Tuple] = {}

    # Kind question: group candidates by their kind
    kinds = sorted({ontology.kind(c) for c in cand})
    if "kind" not in asked and len(kinds) > 1:
        kind_labels = {k: k.replace("nosign", "other").replace("marker", "grammar marker") for k in kinds}
        qs["kind"] = ("What kind of word was it?", kinds, lambda c: ontology.kind(c), kind_labels)

    # Pair question: top-2 direct comparison
    top = sorted(cand, key=lambda c: -cand[c])
    if len(top) >= 2 and "pair" not in asked:
        a, b = top[0], top[1]
        opts = [a, b, "neither"]
        try: la = ontology.label(a, "en")
        except KeyError: la = a
        try: lb = ontology.label(b, "en")
        except KeyError: lb = b
        qs["pair"] = (
            f"Did you mean '{la}' or '{lb}'?", opts,
            lambda c, a=a, b=b: c if c in (a, b) else "neither",
            {a: la, b: lb, "neither": "Neither"})
    return qs


def context(qid: str, state: Any = None) -> float:
    return 0.0


def info_gain(cand: Dict[str, float], options: List[str], truth_fn: Any, reliability: float) -> float:
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


def plan(cand: Dict[str, float], asked: List[str], cfg: Dict[str, Any] = None, ontology=None) -> Dict[str, Any]:
    cfg = cfg or CFG
    best: Dict[str, Any] = None
    for qid, (text, opts, truth, labels) in _question_models(cand, asked, ontology).items():
        ig = info_gain(cand, opts, truth, cfg.get("reliability", 0.95))
        answerability = 1.0 if len(opts) <= 3 else 0.7
        utility = (ig + cfg.get("alpha", 0.0) * context(qid, None)
                   + cfg.get("beta", 0.2) * answerability - cfg.get("lam", 0.1) * len(opts))
        if best is None or utility > best["utility"]:
            best = {"id": qid, "text": text, "options": opts, "labels": labels,
                    "ig": ig, "utility": utility, "truth": truth}
    return best
