"""KB registry: scans data/kbs/*/kb.yaml, plus CRUD for the admin UI
(create/delete a KB, add/remove its source documents).
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from backend.offline_pipeline.loaders import SUPPORTED_EXTENSIONS

ROOT = Path(__file__).resolve().parent.parent
KB_DATA_DIR = ROOT / "data" / "kbs"
FAISS_INDEX_DIR = ROOT / "data" / "faiss_indexes"

_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


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


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def create_kb(name: str, description: str = "", slug: str | None = None) -> KnowledgeBase:
    slug = slug or _slugify(name)
    if not slug or not _SLUG_RE.match(slug):
        raise ValueError(f"Invalid slug: {slug!r} (lowercase letters, digits, hyphens only)")
    kb_dir = KB_DATA_DIR / slug
    if kb_dir.exists():
        raise FileExistsError(f"Knowledge base already exists: {slug}")
    kb_dir.mkdir(parents=True)
    (kb_dir / "kb.yaml").write_text(yaml.safe_dump({"name": name, "description": description}))
    return KnowledgeBase(slug=slug, name=name, description=description)


def delete_kb(slug: str) -> None:
    get_kb(slug)  # raises KeyError if unknown
    shutil.rmtree(KB_DATA_DIR / slug, ignore_errors=True)
    shutil.rmtree(FAISS_INDEX_DIR / slug, ignore_errors=True)


def list_kb_documents(slug: str) -> list[dict]:
    get_kb(slug)
    kb_dir = KB_DATA_DIR / slug
    return [
        {"filename": str(path.relative_to(kb_dir)), "size": path.stat().st_size}
        for path in sorted(kb_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def save_kb_document(slug: str, filename: str, content: bytes) -> str:
    get_kb(slug)
    safe_name = Path(filename).name
    if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {safe_name}")
    (KB_DATA_DIR / slug / safe_name).write_bytes(content)
    return safe_name


def delete_kb_document(slug: str, filename: str) -> None:
    get_kb(slug)
    path = KB_DATA_DIR / slug / Path(filename).name
    if not path.exists():
        raise FileNotFoundError(filename)
    path.unlink()
