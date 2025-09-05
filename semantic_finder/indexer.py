import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import lancedb
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, SiglipVisionModel
from PIL import Image, ImageFile
from .config import (
    PROJECT_ROOT, DATA_DIR, INDEX_DIR,
    TEXT_EMBED_MODEL, IMAGE_EMBED_MODEL,
    CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS,
    TEXT_BATCH, IMAGE_BATCH,
    TABLE_TEXT, TABLE_IMAGE,
)
from .utils import classify, extract_text_any, chunk_text, make_preview
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
# ----- LanceDB schemas -----
class TextRow(LanceModel):
    id: str
    path: str
    chunk_id: int
    mtime: float
    mime: str
    preview: str
    embedding: Vector(768)  # EmbeddingGemma default is 768-d

class ImageRow(LanceModel):
    id: str
    path: str
    mtime: float
    width: int
    height: int
    embedding: Vector(768)  # SigLIP base model projects to 768-d

@dataclass
class Models:
    text: SentenceTransformer
    siglip_vision: SiglipVisionModel
    siglip_processor: AutoImageProcessor
    device: torch.device

def load_models(text_model_id: str, image_model_id: str) -> Models:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Loading text model: {text_model_id} on {device} …", flush=True)
    # Load model without device first, then move to device to avoid meta tensor issues
    text_model = SentenceTransformer(text_model_id)
    text_model = text_model.to(device)
    print("[✓] Text model ready.", flush=True)

    print(f"[+] Loading image model (vision-only): {image_model_id} on {device} …", flush=True)
    siglip_processor = AutoImageProcessor.from_pretrained(image_model_id)
    siglip_vision = SiglipVisionModel.from_pretrained(image_model_id).to(device).eval()
    print("[✓] Image model ready.", flush=True)

    return Models(
        text=text_model,
        siglip_vision=siglip_vision,
        siglip_processor=siglip_processor,
        device=device,
    )


# ----- LanceDB helpers -----
def open_db():
    INDEX_DIR.mkdir(exist_ok=True, parents=True)
    return lancedb.connect(INDEX_DIR.as_posix())

def get_or_create_table(db, name: str, schema):
    if name in db.table_names():
        return db.open_table(name)
    return db.create_table(name, schema=schema, mode="overwrite")

def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / denom

# ----- Text indexing -----
def embed_text_documents(models: Models, docs: List[str], is_query: bool = False) -> np.ndarray:
    # EmbeddingGemma supports encode_query / encode_document; keep embeddings normalized
    if is_query:
        embs = models.text.encode_query(docs, batch_size=TEXT_BATCH, normalize_embeddings=True)
    else:
        embs = models.text.encode_document(docs, batch_size=TEXT_BATCH, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)

def index_text(db, models: Models, files: List[Path]) -> int:
    tbl = get_or_create_table(db, TABLE_TEXT, TextRow)

    # Gather current mtimes to skip unchanged
    existing: Dict[str, float] = {}
    if tbl.count_rows() > 0:
        df = tbl.to_pandas()  # LanceDB 0.25: no columns kwarg
        if not df.empty and {"path", "mtime"}.issubset(df.columns):
            # keep max mtime per path (if multiple chunks)
            existing = df.groupby("path")["mtime"].max().to_dict()

    rows: List[Dict] = []
    added = 0
    for f in files:
        mtime = f.stat().st_mtime
        if existing.get(str(f), -1) >= mtime:
            continue  # unchanged

        text = extract_text_any(f)
        if not text.strip():
            continue

        chunks = chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        if not chunks:
            continue

        # embed as documents
        embs = embed_text_documents(models, chunks, is_query=False)
        for i, (chunk, vec) in enumerate(zip(chunks, embs)):
            rows.append({
                "id": f"{f}:{i}",
                "path": str(f),
                "chunk_id": i,
                "mtime": mtime,
                "mime": f.suffix.lower(),
                "preview": make_preview(chunk),
                "embedding": vec.tolist(),
            })
        added += len(chunks)

        # flush periodically
        if len(rows) >= 2000:
            tbl.add(pd.DataFrame(rows))
            rows.clear()

    if rows:
        tbl.add(pd.DataFrame(rows))
    return added

# ----- Image indexing -----
@torch.inference_mode()
def embed_images_siglip(models: Models, pil_images: List[Image.Image]) -> np.ndarray:
    # Processor handles resize/normalize to 224x224
    inputs = models.siglip_processor(images=pil_images, return_tensors="pt").to(models.device)
    outputs = models.siglip_vision(**inputs)
    # Use pooled CLS if available; otherwise CLS token
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        feats = outputs.pooler_output
    else:
        feats = outputs.last_hidden_state[:, 0, :]
    feats = feats.detach().cpu().float().numpy()
    return l2_normalize(feats).astype(np.float32)


def index_images(db, models: Models, files: List[Path]) -> int:
    tbl = get_or_create_table(db, TABLE_IMAGE, ImageRow)

    existing: Dict[str, float] = {}
    if tbl.count_rows() > 0:
        df = tbl.to_pandas()
        if not df.empty and {"path", "mtime"}.issubset(df.columns):
            existing = df.set_index("path")["mtime"].to_dict()

    rows: List[Dict] = []
    added = 0
    batch_imgs: List[Image.Image] = []
    batch_meta: List[Tuple[str, float, int, int]] = []

    def flush_batch():
        nonlocal rows, batch_imgs, batch_meta
        if not batch_imgs:
            return
        vecs = embed_images_siglip(models, batch_imgs)
        for (p, mtime, w, h), v in zip(batch_meta, vecs):
            rows.append({
                "id": f"{p}",
                "path": p,
                "mtime": mtime,
                "width": w,
                "height": h,
                "embedding": v.tolist(),
            })
        batch_imgs.clear()
        batch_meta.clear()
        if len(rows) >= 1000:
            tbl.add(pd.DataFrame(rows))
            rows.clear()

    for f in files:
        mtime = f.stat().st_mtime
        if existing.get(str(f), -1) >= mtime:
            continue
        try:
            img = Image.open(f).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        batch_imgs.append(img)
        batch_meta.append((str(f), mtime, w, h))
        added += 1
        if len(batch_imgs) >= IMAGE_BATCH:
            flush_batch()

    flush_batch()
    if rows:
        tbl.add(pd.DataFrame(rows))
    return added

# ----- Orchestrator -----
def crawl_files(root: Path) -> Tuple[List[Path], List[Path]]:
    text_like, images = [], []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        kind = classify(p)
        if kind in {"text", "pdf", "docx"}:
            text_like.append(p)
        elif kind == "image":
            images.append(p)
    return text_like, images

def main():
    parser = argparse.ArgumentParser(description="Index local files into LanceDB.")
    parser.add_argument("--root", type=str, default=str(DATA_DIR), help="Root folder to index")
    parser.add_argument("--rebuild", action="store_true", help="Drop & rebuild tables")
    parser.add_argument("--text-model", type=str, default=TEXT_EMBED_MODEL, help="HF ID for text embeddings")
    parser.add_argument("--image-model", type=str, default=IMAGE_EMBED_MODEL, help="HF ID for image embeddings")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    assert root.exists(), f"Root not found: {root}"

    print(f"Loading models on device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    models = load_models(args.text_model, args.image_model)

    db = open_db()
    if args.rebuild:
        # drop + recreate
        if TABLE_TEXT in db.table_names():
            db.drop_table(TABLE_TEXT)
        if TABLE_IMAGE in db.table_names():
            db.drop_table(TABLE_IMAGE)

    text_files, image_files = crawl_files(root)
    print(f"Found {len(text_files)} text-like files, {len(image_files)} images.")

    t0 = time.time()
    added_text = index_text(db, models, text_files)
    t1 = time.time()
    added_img = index_images(db, models, image_files)
    t2 = time.time()

    print(f"Indexed {added_text} text chunks in {t1 - t0:.1f}s")
    print(f"Indexed {added_img} images in {t2 - t1:.1f}s")
    print("Done.")

if __name__ == "__main__":
    main()
