"""Compatibility shim. All realization logic moved to core/semantics.py."""
from core.semantics import NaturalLanguageRealizer  # noqa: re-export
__all__ = ["NaturalLanguageRealizer"]
