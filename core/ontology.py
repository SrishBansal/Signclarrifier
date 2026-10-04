
"""
Concept Ontology Loader for ClarifySign.
Single source of truth: data/ontology.yaml.
Fails loudly at load if sign_ids duplicate, are not in include_classes.txt,
or any concept lacks a label in any declared non-NLU-input-only language.
"""
import os, re, unicodedata
from typing import Dict, Any, Optional, List
import yaml


class Ontology:
    def __init__(self, yaml_path: Optional[str] = None):
        if yaml_path is None:
            yaml_path = os.path.join(os.path.dirname(__file__), "..", "data", "ontology.yaml")
        self.yaml_path = os.path.abspath(yaml_path)
        self._raw: Dict[str, Any] = {}
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.synonym_index: Dict[str, str] = {}
        self.category_index: Dict[str, List[str]] = {}
        self._sign_to_concept: Dict[str, str] = {}   # reverse: sign_id -> concept_id
        self._lang_meta: List[Dict[str, Any]] = []
        self.intent_cues: Dict[str, Any] = {}
        self.templates: Dict[str, Dict[str, str]] = {}
        self.customer_templates: Dict[str, Dict[str, str]] = {}
        self.load()

    # ── load & validate ───────────────────────────────────────────────────────
    def load(self):
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._raw = data
        self._lang_meta = data.get("languages", [])
        self.intent_cues = data.get("intent_cues", {})
        self.templates = data.get("templates", {})
        self.customer_templates = data.get("customer_templates", {})


        # Load include_classes for validation
        inc_path = os.path.join(os.path.dirname(self.yaml_path), "include_classes.txt")
        include_classes: set = set()
        if os.path.exists(inc_path):
            include_classes = {l.strip().lower() for l in open(inc_path) if l.strip()}

        # Non-NLU-input-only lang codes that require labels
        required_langs = [l["code"] for l in self._lang_meta if not l.get("nlu_input_only")]

        self.concepts.clear(); self.synonym_index.clear()
        self.category_index.clear(); self._sign_to_concept.clear()

        seen_sign_ids: Dict[str, str] = {}  # sign_id -> concept_id (duplicate detection)
        errors: List[str] = []

        for raw_cid, details in data.get("concepts", {}).items():
            cid = str(raw_cid).upper()
            if not isinstance(details, dict):
                errors.append(f"Concept {cid}: bad value type"); continue

            # sign_id validation
            sign_id = details.get("sign_id")
            if sign_id is not None:
                sid = str(sign_id).lower()
                if include_classes and sid not in include_classes:
                    errors.append(f"Concept {cid}: sign_id '{sid}' not in include_classes.txt")
                elif sid in seen_sign_ids:
                    errors.append(f"Concept {cid}: sign_id '{sid}' already used by {seen_sign_ids[sid]}")
                else:
                    seen_sign_ids[sid] = cid
                    self._sign_to_concept[sid] = cid

            # label validation
            labels = details.get("label", {})
            for lc in required_langs:
                if lc not in labels or not labels[lc]:
                    errors.append(f"Concept {cid}: missing label for language '{lc}'")

            # store
            self.concepts[cid] = details
            cat = details.get("category", "entity")
            self.category_index.setdefault(cat, []).append(cid)

            # synonym index
            self.synonym_index[cid.lower()] = cid
            for lang, syn_list in details.get("synonyms", {}).items():
                for s in (syn_list or []):
                    norm = self._norm(str(s))
                    if norm:
                        self.synonym_index[norm] = cid

        if errors:
            raise ValueError("Ontology load errors:\n" + "\n".join(errors))

    # ── public API ────────────────────────────────────────────────────────────
    @staticmethod
    def _norm(text: str) -> str:
        t = text.lower().strip()
        t = "".join(ch if (ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M")) else " " for ch in t)
        return re.sub(r"\s+", " ", t).strip()

    # keep old name for callers
    normalize_text = _norm.__func__ if hasattr(_norm, "__func__") else _norm

    @staticmethod
    def normalize_text(text: str) -> str:
        t = text.lower().strip()
        t = "".join(ch if (ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M")) else " " for ch in t)
        return re.sub(r"\s+", " ", t).strip()

    def is_valid_concept(self, concept_id: str) -> bool:
        return str(concept_id).upper() in self.concepts

    def get_category(self, concept_id: str) -> Optional[str]:
        d = self.concepts.get(str(concept_id).upper())
        return d.get("category") if d else None

    def get_sign_id(self, concept_id: str) -> Optional[str]:
        d = self.concepts.get(str(concept_id).upper())
        return d.get("sign_id") if d else None

    def concept_for_sign(self, sign_label: str) -> Optional[str]:
        """Reverse lookup: sign_id (INCLUDE class) -> concept_id. Returns None if unknown."""
        return self._sign_to_concept.get(str(sign_label).lower())

    def label(self, concept_id: str, lang: str) -> str:
        """Display label for concept in lang. Raises KeyError if missing."""
        d = self.concepts.get(str(concept_id).upper(), {})
        lbls = d.get("label", {})
        if lang not in lbls:
            raise KeyError(f"No label for concept {concept_id} in lang {lang}")
        return lbls[lang]

    def kind(self, concept_id: str) -> str:
        """Return kind: 'sign' | 'nosign' | 'marker'."""
        d = self.concepts.get(str(concept_id).upper(), {})
        cat = d.get("category", "")
        if cat == "grammar":
            return "marker"
        if d.get("sign_id") is None:
            return "nosign"
        return "sign"

    def languages(self) -> List[Dict[str, Any]]:
        return list(self._lang_meta)

    def all_concepts(self) -> List[str]:
        return sorted(self.concepts.keys())

    def resolve_concept(self, term: str) -> Optional[str]:
        norm = self.normalize_text(term)
        if not norm:
            return None
        if norm.upper() in self.concepts:
            return norm.upper()
        if norm in self.synonym_index:
            return self.synonym_index[norm]
        for w in norm.split():
            if w in self.synonym_index:
                return self.synonym_index[w]
        for w in norm.split():
            for suffix in ("es", "s"):
                if w.endswith(suffix) and len(w) > len(suffix) + 1:
                    stem = w[:-len(suffix)]
                    if stem in self.synonym_index:
                        return self.synonym_index[stem]
        return None


_ontology_instance: Optional[Ontology] = None

def get_ontology() -> Ontology:
    global _ontology_instance
    if _ontology_instance is None:
        _ontology_instance = Ontology()
    return _ontology_instance
