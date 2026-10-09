#!/usr/bin/env python3
"""Read-only end-to-end check using the candidate ontology + candidate signs.
python3 scripts/e2e_candidate.py            (default phrases)
python3 scripts/e2e_candidate.py "tall shirt" "good night"
"""
import sys
sys.path.insert(0, ".")
from core.ontology import Ontology
from core.nlu import NLUParser
from core.planner import ISLPlanner
from core.sign_library.library import SignLibrary

ont = Ontology("data/ontology_candidate.yaml")
nlu = NLUParser(ont)
planner = ISLPlanner(ontology=ont)
lib = SignLibrary(source="models/signs_include_candidate.json")

DEFAULT = ["hello", "good morning", "good night", "how are you", "hello how are you",
           "thank you", "tomorrow", "monday morning", "tall", "tall shirt", "old shirt",
           "cheap shirt", "clean clothes", "how much is the soft shirt", "are you happy",
           "is it wet", "come on friday", "do you have a pocket", "xylophone"]

def matched(tokens):
    found, covered = [], set()
    i = 0
    while i < len(tokens):
        for n in (3, 2, 1):
            g = " ".join(tokens[i:i + n])
            if len(tokens[i:i + n]) == n and g in ont.synonym_index:
                found.append(ont.synonym_index[g]); covered.update(range(i, i + n)); i += n; break
        else:
            i += 1
    return found, [t for k, t in enumerate(tokens) if k not in covered]

for phrase in (sys.argv[1:] or DEFAULT):
    print(f"\n=== {phrase!r}")
    try:
        state = nlu.parse(phrase, "English")
        plan = planner.plan(state)
    except Exception as e:
        print("  ERROR:", type(e).__name__, e); continue
    toks = ont.normalize_text(phrase).split()
    expected, unmapped = matched(toks)
    print("  intent:", getattr(state.intent, "value", state.intent))
    print("  items :", [(it.concept, dict(it.attributes or {})) for it in state.items])
    parts = []
    for c in plan:
        sid = ont.get_sign_id(c)
        st = ("OK" if sid and lib.has_sign(sid) else "NOSIGN" if not sid and ont.is_valid_concept(c)
              else "MISSING-FROM-LIBRARY" if sid else "OOV")
        parts.append(f"{c}[{st}]")
    print("  plan  :", " ".join(parts) or "(empty)")
    dropped = [c for c in dict.fromkeys(expected) if ont.get_sign_id(c) and c not in plan]
    if dropped: print("  DROPPED (recognised but not planned):", dropped)
    if unmapped: print("  unmapped words:", unmapped)
    if plan == ["HELLO"] and "HELLO" not in expected: print("  *** FALLBACK TO HELLO ***")
