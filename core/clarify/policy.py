# Policy (decision) module for clarification

import math
from typing import List, Dict, Any

from .candidates import candidates, uncertainty, H
from .config import load_config
from .planner import plan, _likelihood

# Load default configuration
CFG = load_config()


class Dialogue:
    """One clarification episode: from uncertain recognition to resolved concept.

    Mirrors the original implementation from `app/clarify.py` but relies on the
    core utilities and configuration file.
    """

    def __init__(self, probs: List[float], classes: List[str], cfg: Dict[str, Any] = None):
        self.cfg = cfg or CFG
        self.classes = classes
        self.asked: List[str] = []
        self.turn = 0
        self.cand, self.top = candidates(probs, classes, self.cfg.get("top_k"))
        self.unc = uncertainty(list(probs))
        self.q = None

    def start(self) -> Dict[str, Any]:
        u, p1 = self.unc, self.unc["top1"]
        best = max(self.cand, key=self.cand.get)
        if p1 >= self.cfg.get("commit_p", 0.85) and u["margin"] >= self.cfg.get("commit_margin", 0.4):
            return {"action": "resolved", "concept": best, "how": "confident"}
        if p1 < self.cfg.get("resign_p", 0.15):
            return {"action": "resign"}
        return self._ask()

    def _ask(self) -> Dict[str, Any]:
        q = plan(self.cand, self.asked, self.cfg)
        if q is None or q["ig"] < self.cfg.get("min_gain", 0.05) or self.turn >= self.cfg.get("max_turns", 2):
            best = max(self.cand, key=self.cand.get)
            if self.turn:
                return {"action": "resolved", "concept": best, "how": "best_guess"}
            else:
                return {"action": "resign"}
        self.q = q
        self.asked.append(q["id"])
        self.turn += 1
        return {
            "action": "ask",
            "question": {
                "id": q["id"],
                "text": q["text"],
                "options": [{"value": o, "label": q["labels"][o]} for o in q["options"]],
            },
            "ig": q["ig"],
            "utility": q["utility"],
        }

    def answer(self, value: str) -> Dict[str, Any]:
        q = self.q
        if q is None or value not in q["options"]:
            raise ValueError("invalid answer")
        # Posterior update using the same likelihood as in planner
        post = {
            c: p * _likelihood(q["options"], q["truth"](c), self.cfg.get("reliability", 0.95))[value]
            for c, p in self.cand.items()
        }
        z = sum(post.values()) or 1.0
        self.cand = {c: v / z for c, v in post.items()}
        best = max(self.cand, key=self.cand.get)
        if self.cand[best] >= self.cfg.get("resolve_p", 0.8):
            return {"action": "resolved", "concept": best, "how": "clarified", "p": self.cand[best]}
        return self._ask()

__all__ = ["Dialogue", "CFG"]
