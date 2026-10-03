import yaml
import os
from typing import Dict

def load_config(path: str = None) -> Dict:
    """Load clarification config YAML.
    If ``path`` is None, uses the default location within the repository.
    """
    if path is None:
        default = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "clarify.yaml")
        path = os.path.abspath(default)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg
