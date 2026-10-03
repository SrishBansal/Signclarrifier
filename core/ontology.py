"""
Concept Ontology Loader and Resolver for ClarifySign.
Loads data/ontology.yaml and provides concept lookup, validation, and sign mappings.
Has NO web or framework dependencies.
"""

import os
import re
from typing import Dict, Any, Optional, List, Tuple
import yaml


class Ontology:
    """Singleton-friendly domain ontology manager."""

    def __init__(self, yaml_path: Optional[str] = None):
        if yaml_path is None:
            yaml_path = os.path.join(os.path.dirname(__file__), "..", "data", "ontology.yaml")
        self.yaml_path = os.path.abspath(yaml_path)
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.synonym_index: Dict[str, str] = {}  # normalized synonym -> concept_id
        self.category_index: Dict[str, List[str]] = {}
        self.load()

    def load(self):
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"Ontology file not found at {self.yaml_path}")
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        raw_concepts = data.get("concepts", {})
        self.concepts = raw_concepts
        self.synonym_index.clear()
        self.category_index.clear()

        for concept_id, details in raw_concepts.items():
            # Safety: YAML may parse YES/NO/TRUE/FALSE as booleans
            if not isinstance(concept_id, str):
                concept_id = str(concept_id).upper()
            cid = concept_id.upper()
            cat = details.get("category", "entity")
            if cat not in self.category_index:
                self.category_index[cat] = []
            self.category_index[cat].append(cid)

            # Self-reference
            self.synonym_index[cid.lower()] = cid

            synonyms = details.get("synonyms", {})
            for lang, syn_list in synonyms.items():
                for s in syn_list:
                    norm_s = self.normalize_text(s)
                    if norm_s:
                        self.synonym_index[norm_s] = cid

    @staticmethod
    def normalize_text(text: str) -> str:
        t = text.lower().strip()
        t = re.sub(r"[^\w\s\u0900-\u097F\u0B80-\u0BFF]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def is_valid_concept(self, concept_id: str) -> bool:
        return concept_id.upper() in self.concepts

    def get_category(self, concept_id: str) -> Optional[str]:
        details = self.concepts.get(concept_id.upper())
        return details.get("category") if details else None

    def get_sign_id(self, concept_id: str) -> str:
        details = self.concepts.get(concept_id.upper())
        if details:
            return details.get("sign_id", concept_id.upper())
        return f"FS_{concept_id.upper()}"

    def fingerspell(self, word: str) -> List[str]:
        """Fallback to fingerspelling character tokens."""
        clean = re.sub(r"[^A-Za-z0-9]", "", word.upper())
        return [f"FS_{c}" for c in clean] if clean else [f"FS_{word.upper()}"]

    def all_concepts(self) -> List[str]:
        return sorted(list(self.concepts.keys()))

    def resolve_concept(self, term: str) -> Optional[str]:
        """Resolves word or phrase to concept ID using exact match then synonym index."""
        norm = self.normalize_text(term)
        if not norm:
            return None
        
        # Direct hit
        if norm.upper() in self.concepts:
            return norm.upper()
        if norm in self.synonym_index:
            return self.synonym_index[norm]

        # Token search if multi-word or compound
        words = norm.split()
        for w in words:
            if w in self.synonym_index:
                return self.synonym_index[w]

        # English plural fallback: strip trailing s/es
        for w in words:
            if w.endswith("es") and len(w) > 3:
                stem = w[:-2]
                if stem in self.synonym_index:
                    return self.synonym_index[stem]
            if w.endswith("s") and len(w) > 2:
                stem = w[:-1]
                if stem in self.synonym_index:
                    return self.synonym_index[stem]

        return None


# Global singleton instance
_ontology_instance: Optional[Ontology] = None


def get_ontology() -> Ontology:
    global _ontology_instance
    if _ontology_instance is None:
        _ontology_instance = Ontology()
    return _ontology_instance
