
"""Vocabulary integrity gate for data/ontology.yaml."""
import ast, os, sys, pathlib
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.ontology import Ontology

INC = {l.strip().lower() for l in (ROOT/"data"/"include_classes.txt").read_text().splitlines() if l.strip()}
REQUIRED_LANGS = ["en", "hi", "ta"]

# Files whose large literal collections are structural, not inline vocab
ALLOWLIST = {
    "core/__init__.py", "core/clarify/__init__.py", "core/sign_library/__init__.py",
    "app/clarify.py", "app/server.py", "app/semantics.py", "core/models.py",
    "core/clarify/planner.py", "core/clarify/candidates.py",
    "core/dialogue.py", "tests/test_vocabulary.py",
    # structural token sets / routing tables – not domain vocab
    "core/nlu.py", "app/session.py",
}

def _py_files():
    for d in ["core", "app"]:
        for p in (ROOT/d).rglob("*.py"):
            if "__pycache__" not in str(p):
                yield p

def _is_domain(fp):
    return str(fp.relative_to(ROOT)) not in ALLOWLIST

def test_sign_ids_unique_and_in_include():
    ont = Ontology(str(ROOT/"data"/"ontology.yaml"))
    seen = {}
    for cid, d in ont.concepts.items():
        sid = d.get("sign_id")
        if sid is None:
            continue
        sid_l = str(sid).lower()
        assert sid_l in INC, f"{cid}: sign_id {sid!r} not in include_classes.txt"
        assert sid_l not in seen, f"{cid}: sign_id {sid!r} duplicated (also in {seen[sid_l]})"
        seen[sid_l] = cid

def test_every_concept_labelled():
    ont = Ontology(str(ROOT/"data"/"ontology.yaml"))
    for cid, d in ont.concepts.items():
        lbls = d.get("label", {})
        for lc in REQUIRED_LANGS:
            assert lc in lbls and lbls[lc], f"{cid}: missing label[{lc!r}]"

def test_counts():
    ont = Ontology(str(ROOT/"data"/"ontology.yaml"))
    with_sign = sum(1 for d in ont.concepts.values() if d.get("sign_id"))
    without = len(ont.concepts) - with_sign
    print(f"  Concepts total={len(ont.concepts)} with_sign_id={with_sign} without={without}")
    assert len(ont.concepts) > 50

def test_no_non_ascii_string_literals_in_source():
    """No non-ASCII string literals in domain Python files (all text lives in ontology.yaml)."""
    violations = []
    for fp in _py_files():
        if not _is_domain(fp):
            continue
        src = fp.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(ord(c) > 127 for c in node.value):
                    rel = fp.relative_to(ROOT)
                    violations.append(f"{rel}:{node.lineno}: {node.value[:40]!r}")
    assert not violations, "Non-ASCII string literals in source:\n" + "\n".join(violations[:10])

def test_no_large_inline_string_collections():
    """No set/list/dict literals with >=5 string constants in domain Python files.
    For ast.Dict, only the VALUES are checked (not keys), since metadata dicts
    with string keys (like sign metadata) are structural, not vocabulary collections.
    """
    violations = []
    for fp in _py_files():
        if not _is_domain(fp):
            continue
        src = fp.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Set, ast.List)):
                str_elts = [e for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                if len(str_elts) >= 5:
                    rel = fp.relative_to(ROOT)
                    violations.append(f"{rel}:{node.lineno}: {len(str_elts)} strings")
            elif isinstance(node, ast.Dict):
                # Only check VALUES, not keys — metadata dicts with string keys are structural
                str_vals = [v for v in node.values if v is not None and isinstance(v, ast.Constant) and isinstance(v.value, str)]
                if len(str_vals) >= 5:
                    rel = fp.relative_to(ROOT)
                    violations.append(f"{rel}:{node.lineno}: {len(str_vals)} string values")
    assert not violations, "Large inline string literals in domain source:\n" + "\n".join(violations[:15])

