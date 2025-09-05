import io
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF
from docx import Document

from .config import (
    TEXT_EXTS, CODE_EXTS, PDF_EXTS, DOCX_EXTS, IMAGE_EXTS, PREVIEW_CHARS
)

def classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in PDF_EXTS:
        return "pdf"
    if ext in DOCX_EXTS:
        return "docx"
    if ext in TEXT_EXTS or ext in CODE_EXTS:
        return "text"
    return "other"

def read_text_plain(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        # last resort: binary read -> decode ignoring errors
        return path.read_bytes().decode("utf-8", errors="ignore")

def read_text_pdf(path: Path) -> str:
    try:
        doc = fitz.open(str(path))
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        return "\n".join(texts)
    except Exception:
        return ""

def read_text_docx(path: Path) -> str:
    try:
        d = Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""

def extract_text_any(path: Path) -> str:
    kind = classify(path)
    if kind == "pdf":
        return read_text_pdf(path)
    if kind == "docx":
        return read_text_docx(path)
    if kind == "text":
        return read_text_plain(path)
    return ""  # images etc. handled elsewhere

_WORD_RE = re.compile(r"\S+")

def chunk_text(text: str, max_words: int, overlap_words: int) -> List[str]:
    words = _WORD_RE.findall(text or "")
    if not words:
        return []
    chunks = []
    i = 0
    step = max(1, max_words - overlap_words)
    while i < len(words):
        chunk_words = words[i:i + max_words]
        chunks.append(" ".join(chunk_words))
        i += step
    return chunks

def make_preview(text: str) -> str:
    text = text.replace("\n", " ").strip()
    return text[:PREVIEW_CHARS]
