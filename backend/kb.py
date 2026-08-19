"""KB registry: scans data/kbs/*/kb.yaml. Admin-authored KB creation is a
future phase; for now dropping a folder + kb.yaml under data/kbs is enough.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
KB_DATA_DIR = ROOT / "data" / "kbs"


@dataclass(frozen=True)
class KnowledgeBase:
    slug: str
    name: str
    description: str


def list_knowledge_bases() -> list[KnowledgeBase]:
    kbs = []
    if not KB_DATA_DIR.exists():
        return kbs
    for kb_dir in sorted(KB_DATA_DIR.iterdir()):
        yaml_path = kb_dir / "kb.yaml"
        if not kb_dir.is_dir() or not yaml_path.exists():
            continue
        meta = yaml.safe_load(yaml_path.read_text()) or {}
        kbs.append(
            KnowledgeBase(
                slug=kb_dir.name,
                name=meta.get("name", kb_dir.name),
                description=meta.get("description", ""),
            )
        )
    return kbs


def get_kb(slug: str) -> KnowledgeBase:
    for kb in list_knowledge_bases():
        if kb.slug == slug:
            return kb
    raise KeyError(f"Unknown knowledge base: {slug}")
