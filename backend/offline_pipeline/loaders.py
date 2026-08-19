"""Extension -> langchain document loader dispatch for KB source files."""

from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

_LOADERS = {
    ".md": TextLoader,
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
}

SUPPORTED_EXTENSIONS = set(_LOADERS)


def load_source_file(path: Path) -> list[Document]:
    loader_cls = _LOADERS.get(path.suffix.lower())
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return loader_cls(str(path)).load()
