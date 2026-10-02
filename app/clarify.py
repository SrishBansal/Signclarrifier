"""Uncertainty + Expected-Information-Gain clarification.

U(q) = IG(q) + alpha*Context(q) + beta*Answerability(q) - lambda*InteractionCost(q)
IG(q) = mutual information (bits) between the true concept and the user's (noisy) answer, over the candidate set.
Context(q) is 0.0 for now: no dialogue-history model exists yet. The hook is `context()`.
Thresholds are starting values, not tuned. Tune on validation data."""
import math
from .semantics import CONCEPTS, KIND_EN, EN

CFG = dict(top_k=6, commit_p=0.85, commit_margin=0.40, resign_p=0.15, resolve_p=0.80,
           reliability=0.95, alpha=0.0, beta=0.2, lam=0.1, min_gain=0.05, max_turns=2)


def H(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def candidates(probs, classes, k=None):
    k = k or CFG["top_k"]
    idx = sorted(range(len(classes)), key=lambda i: -probs[i])[:k]
    mass = sum(probs[i] for i in idx) or 1.0
    return {classes[i]: probs[i] / mass for i in idx}, [(classes[i], probs[i]) for i in idx]


def uncertainty(probs):
    s = sorted(probs, reverse=True)
    n = len(probs)
    return {"top1": s[0], "margin": s[0] - (s[1] if n > 1 else 0.0),
            "entropy": H(probs), "entropy_norm": H(probs) / math.log2(n) if n > 1 else 0.0}


def _likelihood(options, truth_option, r):
    """P(answer=a | truth) for each option a: r on the right option, the rest spread evenly."""
    m = len(options)
    if m == 1:
        return {options[0]: 1.0}
    return {a: (r if a == truth_option else (1 - r) / max(m - 1, 1)) for a in options}


def _question_models(cand, asked):
    """Return {qid: (text, options[list of values], truth_fn(concept)->value, labels{value:label})}."""
    qs = {}
    kinds = sorted({CONCEPTS[c] for c in cand})
    if "kind" not in asked and len(kinds) > 1:
        qs["kind"] = ("What kind of word was it?", kinds, lambda c: CONCEPTS[c], {k: KIND_EN[k] for k in kinds})
    top = sorted(cand, key=lambda c: -cand[c])
    if len(top) >= 2 and "pair" not in asked:
        a, b = top[0], top[1]
        opts = [a, b, "neither"]
        qs["pair"] = (f"Did you mean '{EN[a]}' or '{EN[b]}'?", opts,
                      lambda c, a=a, b=b: c if c in (a, b) else "neither", {a: EN[a], b: EN[b], "neither": "Neither"})
    return qs


def context(qid, state):
    return 0.0


def info_gain(cand, options, truth_fn, r):
    prior = list(cand.values())
    pa = {a: 0.0 for a in options}
    post = {a: {} for a in options}
    for c, p in cand.items():
        lik = _likelihood(options, truth_fn(c), r)
        for a in options:
            pa[a] += p * lik[a]
            post[a][c] = p * lik[a]
    exp_h = 0.0
    for a in options:
        if pa[a] > 0:
            exp_h += pa[a] * H([v / pa[a] for v in post[a].values()])
    return H(prior) - exp_h


def plan(cand, asked, cfg=CFG):
    best = None
    for qid, (text, opts, truth, labels) in _question_models(cand, asked).items():
        ig = info_gain(cand, opts, truth, cfg["reliability"])
        answer = 1.0 if len(opts) <= 3 else 0.7
        u = ig + cfg["alpha"] * context(qid, None) + cfg["beta"] * answer - cfg["lam"] * len(opts)
        if best is None or u > best["utility"]:
            best = {"id": qid, "text": text, "options": opts, "labels": labels, "ig": ig, "utility": u, "truth": truth}
    return best


class Dialogue:
    """One clarification episode: from an uncertain recognition to a resolved concept."""

    def __init__(self, probs, classes, cfg=CFG):
        self.cfg, self.classes, self.asked, self.turn = cfg, classes, [], 0
        self.cand, self.top = candidates(probs, classes, cfg["top_k"])
        self.unc = uncertainty(list(probs))
        self.q = None

    def start(self):
        u, p1 = self.unc, self.unc["top1"]
        best = max(self.cand, key=self.cand.get)
        if p1 >= self.cfg["commit_p"] and u["margin"] >= self.cfg["commit_margin"]:
            return {"action": "resolved", "concept": best, "how": "confident"}
        if p1 < self.cfg["resign_p"]:
            return {"action": "resign"}
        return self._ask()

    def _ask(self):
        q = plan(self.cand, self.asked, self.cfg)
        if q is None or q["ig"] < self.cfg["min_gain"] or self.turn >= self.cfg["max_turns"]:
            best = max(self.cand, key=self.cand.get)
            return {"action": "resolved", "concept": best, "how": "best_guess"} if self.turn else {"action": "resign"}
        self.q = q; self.asked.append(q["id"]); self.turn += 1
        return {"action": "ask", "question": {"id": q["id"], "text": q["text"],
                "options": [{"value": o, "label": q["labels"][o]} for o in q["options"]]}, "ig": q["ig"], "utility": q["utility"]}

    def answer(self, value):
        q = self.q
        if q is None or value not in q["options"]:
            raise ValueError("invalid answer")
        post = {c: p * _likelihood(q["options"], q["truth"](c), self.cfg["reliability"])[value] for c, p in self.cand.items()}
        z = sum(post.values()) or 1.0
        self.cand = {c: v / z for c, v in post.items()}
        best = max(self.cand, key=self.cand.get)
        if self.cand[best] >= self.cfg["resolve_p"]:
            return {"action": "resolved", "concept": best, "how": "clarified", "p": self.cand[best]}
        return self._ask()
