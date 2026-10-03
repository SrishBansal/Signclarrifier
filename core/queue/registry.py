"""Simple JSON‑based model registry for the queue system.
The registry keeps a list of model versions, their dataset source, metrics and whether
they were promoted. It lives in ``data/model_registry.json``.
"""

import json
from pathlib import Path
from typing import Dict, Any

from .config import REGISTRY_FILE


class ModelRegistry:
    """Manage a JSON file that records fine‑tuned model entries.
    Each entry is a mapping:
    ``version`` -> {"path": <dataset_dir>, "metrics": {...}, "promoted": bool}
    """

    def __init__(self) -> None:
        self.file = Path(REGISTRY_FILE)
        self.entries: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.file.is_file():
            try:
                with self.file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.file.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

    def register(self, *, version: str, path: str, metrics: Dict[str, Any], promoted: bool) -> None:
        """Add a new model entry and persist the registry."""
        self.entries[version] = {
            "path": path,
            "metrics": metrics,
            "promoted": promoted,
        }
        self._save()

    def get(self, version: str) -> Dict[str, Any]:
        return self.entries.get(version, {})

    def list_versions(self) -> list[str]:
        return sorted(self.entries.keys())

    def latest(self) -> str | None:
        versions = self.list_versions()
        return versions[-1] if versions else None

__all__ = ["ModelRegistry"]
