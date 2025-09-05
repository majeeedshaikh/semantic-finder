import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import lancedb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, SiglipTextModel, AutoImageProcessor, SiglipVisionModel

from .config import INDEX_DIR, TABLE_TEXT, TABLE_IMAGE, TEXT_EMBED_MODEL, IMAGE_EMBED_MODEL, PREVIEW_CHARS

# -------------------- utils --------------------
def open_db():
    INDEX_DIR.mkdir(exist_ok=True, parents=True)
    return lancedb.connect(INDEX_DIR.as_posix())

def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / denom

def pick_score_col(df: pd.DataFrame) -> str:
    for c in ["_distance", "score", "_score", "vector_distance"]:
        if c in df.columns:
            return c
    return None

def cosine_from_distance(dist: np.ndarray) -> np.ndarray:
    # LanceDB returns L2 distance by default. Convert to "similarity-like" score.
    # Smaller distance => higher score. We'll invert and scale to [0, 1) approximately.
    dist = np.asarray(dist, dtype=np.float32)
    return 1.0 / (1.0 + dist)

def preview_text(s: str) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s[:PREVIEW_CHARS]

# -------------------- models --------------------
@dataclass
class Models:
    # text encoder for text chunks (E5/EmbeddingGemma/etc.)
    text_embed: SentenceTransformer
    # SigLIP text tower (for text->image)
    siglip_tok: AutoTokenizer
    siglip_text: SiglipTextModel
    # SigLIP vision tower (for image->image)
    siglip_proc: AutoImageProcessor
    siglip_vision: SiglipVisionModel
    device: torch.device

def load_models(text_model_id: str, siglip_id: str) -> Models:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"[+] Loading query text model: {text_model_id} on {device} …", flush=True)
    # Load model without device first, then move to device to avoid meta tensor issues
    txt = SentenceTransformer(text_model_id)
    txt = txt.to(device)
    print("[✓] Text model ready.", flush=True)

    print(f"[+] Loading SigLIP text & vision: {siglip_id} …", flush=True)
    tok = AutoTokenizer.from_pretrained(siglip_id)
    tmodel = SiglipTextModel.from_pretrained(siglip_id).to(device).eval()
    iproc = AutoImageProcessor.from_pretrained(siglip_id)
    vmodel = SiglipVisionModel.from_pretrained(siglip_id).to(device).eval()
    print("[✓] SigLIP ready.", flush=True)

    return Models(
        text_embed=txt,
        siglip_tok=tok,
        siglip_text=tmodel,
        siglip_proc=iproc,
        siglip_vision=vmodel,
        device=device,
    )

@torch.inference_mode()
def siglip_text_embed(models: Models, queries: List[str]) -> np.ndarray:
    toks = models.siglip_tok(
        queries, return_tensors="pt", padding=True, truncation=True, max_length=64
    ).to(models.device)
    out = models.siglip_text(**toks)
    # prefer pooled if available, else CLS
    feats = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0, :]
    v = feats.detach().cpu().float().numpy()
    return l2_normalize(v).astype(np.float32)

@torch.inference_mode()
def siglip_image_embed(models: Models, image_path: Path) -> np.ndarray:
    from PIL import Image, ImageFile
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(image_path).convert("RGB")
    inputs = models.siglip_proc(images=[img], return_tensors="pt").to(models.device)
    out = models.siglip_vision(**inputs)
    feats = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0, :]
    v = feats.detach().cpu().float().numpy()
    return l2_normalize(v).astype(np.float32)

# -------------------- search ops --------------------
def search_text_chunks(db, q_vec: np.ndarray, k: int) -> pd.DataFrame:
    if TABLE_TEXT not in db.table_names():
        return pd.DataFrame()
    tbl = db.open_table(TABLE_TEXT)
    df = tbl.search(q_vec[0]).limit(k).to_pandas()
    if df.empty:
        return df
    sc = pick_score_col(df)
    if sc:
        df["score"] = cosine_from_distance(df[sc].values)
    else:
        df["score"] = 0.0
    df["type"] = "text"
    df["display"] = df["preview"].apply(preview_text)
    return df[["type", "path", "score", "display", "mtime"]]

def search_images_with_text(db, q_vec: np.ndarray, k: int) -> pd.DataFrame:
    if TABLE_IMAGE not in db.table_names():
        return pd.DataFrame()
    tbl = db.open_table(TABLE_IMAGE)
    df = tbl.search(q_vec[0]).limit(k).to_pandas()
    if df.empty:
        return df
    sc = pick_score_col(df)
    if sc:
        df["score"] = cosine_from_distance(df[sc].values)
    else:
        df["score"] = 0.0
    df["type"] = "image"
    df["display"] = "image"
    return df[["type", "path", "score", "display", "mtime"]]

def search_images_with_image(db, img_vec: np.ndarray, k: int) -> pd.DataFrame:
    if TABLE_IMAGE not in db.table_names():
        return pd.DataFrame()
    tbl = db.open_table(TABLE_IMAGE)
    df = tbl.search(img_vec[0]).limit(k).to_pandas()
    if df.empty:
        return df
    sc = pick_score_col(df)
    if sc:
        df["score"] = cosine_from_distance(df[sc].values)
    else:
        df["score"] = 0.0
    df["type"] = "image"
    df["display"] = "image"
    return df[["type", "path", "score", "display", "mtime"]]

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Search LanceDB index (text, images).")
    ap.add_argument("--q", type=str, default=None, help="Text query")
    ap.add_argument("--image", type=str, default=None, help="Path to a query image")
    ap.add_argument("--topk", type=int, default=10, help="Top-K per modality")
    ap.add_argument("--text-model", type=str, default=TEXT_EMBED_MODEL, help="HF id for text embedder")
    ap.add_argument("--image-model", type=str, default=IMAGE_EMBED_MODEL, help="HF id for SigLIP")
    ap.add_argument("--open", action="store_true", help="Open top result in Finder")
    args = ap.parse_args()

    if not args.q and not args.image:
        print("Provide --q 'your query' or --image /path/to/img")
        return

    db = open_db()
    models = load_models(args.text_model, args.image_model)

    results = []

    # text -> text
    if args.q:
        q_emb = np.asarray(models.text_embed.encode_query([args.q], normalize_embeddings=True), dtype=np.float32)
        results.append(search_text_chunks(db, q_emb, args.topk))

        # text -> image (SigLIP text tower)
        t2i = siglip_text_embed(models, [args.q])
        results.append(search_images_with_text(db, t2i, args.topk))

    # image -> image
    if args.image:
        ivec = siglip_image_embed(models, Path(args.image))
        results.append(search_images_with_image(db, ivec, args.topk))

    if not results:
        print("No results.")
        return

    merged = pd.concat([df for df in results if df is not None and not df.empty], ignore_index=True)
    if merged.empty:
        print("No results.")
        return

    merged = merged.sort_values(by="score", ascending=False).reset_index(drop=True)

    # pretty print
    print("\n=== RESULTS ===")
    for i, row in merged.head(args.topk * 2).iterrows():
        print(f"[{i+1:02d}] ({row['type']}) score={row['score']:.3f}  {row['path']}")
        if row["type"] == "text":
            print(f"     {row['display']}")

    if args.open:
        top_path = merged.iloc[0]["path"]
        print(f"\nOpening top result: {top_path}")
        os.system(f'open "{top_path}"')

if __name__ == "__main__":
    main()
