import json
from pathlib import Path
from typing import Dict, List


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parent / "registry.json"


def load_registry(registry_path: str = None) -> List[Dict]:
    path = Path(registry_path) if registry_path else _default_registry_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("sources", [])

