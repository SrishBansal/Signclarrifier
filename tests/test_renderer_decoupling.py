"""Architectural Verification Gate:
Proves that sign_library and app/static/renderer.js import no NLU or planner code.
"""
import ast
import os
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FORBIDDEN_MODULES = {
    "core.nlu",
    "core.planner",
    "core.dialogue",
    "nlu",
    "planner",
    "dialogue",
}

FORBIDDEN_SYMBOLS = {
    "NLUParser",
    "ISLPlanner",
    "DialogueManager",
    "SemanticState",
}


def get_imports_from_file(filepath: str):
    """Parses Python file with AST and extracts all imported module names and symbols."""
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    imports = []
    from_imports = []
    imported_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                imported_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            from_imports.append(mod)
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)

    return imports, from_imports, imported_names


def test_sign_library_has_no_nlu_planner_imports():
    """Verify all files under core/sign_library/ have zero imports of NLU / planner code."""
    lib_dir = os.path.join(ROOT, "core", "sign_library")
    py_files = glob.glob(os.path.join(lib_dir, "**", "*.py"), recursive=True)
    assert len(py_files) > 0, "core/sign_library must contain Python modules"

    for py_file in py_files:
        imports, from_imports, names = get_imports_from_file(py_file)
        all_modules = set(imports + from_imports)
        for forbidden in FORBIDDEN_MODULES:
            for mod in all_modules:
                assert not (mod == forbidden or mod.startswith(forbidden + ".")), (
                    f"VIOLATION: {os.path.basename(py_file)} imports forbidden module '{mod}'"
                )
        for sym in FORBIDDEN_SYMBOLS:
            assert sym not in names, (
                f"VIOLATION: {os.path.basename(py_file)} imports forbidden symbol '{sym}'"
            )


def test_js_renderer_decoupling():
    """Verify app/static/renderer.js does not contain imports/references to NLU or planner."""
    js_path = os.path.join(ROOT, "app", "static", "renderer.js")
    assert os.path.exists(js_path), "app/static/renderer.js must exist"

    with open(js_path, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_js = ["core/nlu", "core/planner", "NLUParser", "ISLPlanner", "SemanticState"]
    for word in forbidden_js:
        assert word not in content, f"VIOLATION: renderer.js references '{word}'"
