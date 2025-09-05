# semantic_finder/ui.py
import os
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import INDEX_DIR, TEXT_EMBED_MODEL, IMAGE_EMBED_MODEL
from .search import (
    load_models as load_search_models,
    siglip_text_embed,
    search_text_chunks,
    search_images_with_text,
    open_db,
)

app = FastAPI(title="Semantic Finder")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- model & DB cache ----------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DB = open_db()

# Allow overrides without editing code
TEXT_ID = os.getenv("SF_TEXT_MODEL", TEXT_EMBED_MODEL)
IMAGE_ID = os.getenv("SF_IMAGE_MODEL", IMAGE_EMBED_MODEL)

# Use a smaller, faster model for development if specified
FAST_MODE = os.getenv("SF_FAST_MODE", "false").lower() == "true"
if FAST_MODE:
    # Use a model with same dimensions as the original to avoid database issues
    TEXT_ID = "sentence-transformers/all-mpnet-base-v2"  # 768-d, faster than Gemma
    print("[!] Fast mode enabled - using faster 768-d model for development")

MODELS = None
MODELS_LOADING = False
MODELS_LOAD_ERROR = None

def get_models():
    global MODELS, MODELS_LOADING, MODELS_LOAD_ERROR
    
    if MODELS is not None:
        return MODELS
    
    if MODELS_LOADING:
        # Return a simple error response while loading
        raise Exception("Models are still loading, please try again in a moment")
    
    if MODELS_LOAD_ERROR is not None:
        raise MODELS_LOAD_ERROR
    
    try:
        MODELS_LOADING = True
        print(f"[+] Loading models (this may take a while for first time)...")
        MODELS = load_search_models(TEXT_ID, IMAGE_ID)
        print(f"[✓] Models loaded successfully!")
        return MODELS
    except Exception as e:
        MODELS_LOAD_ERROR = e
        print(f"[✗] Failed to load models: {e}")
        raise e
    finally:
        MODELS_LOADING = False



def merge_and_rank(dfs: List[pd.DataFrame], limit: int = 20) -> List[dict]:
    parts = [d for d in dfs if d is not None and not d.empty]
    if not parts:
        return []
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("score", ascending=False).head(limit).reset_index(drop=True)
    out = []
    for _, r in df.iterrows():
        out.append({
            "type": r["type"],
            "path": r["path"],
            "score": float(r["score"]),
            "display": r.get("display", ""),
            "mtime": float(r.get("mtime", 0)),
        })
    return out

@app.get("/health")
def health():
    global MODELS, MODELS_LOADING, MODELS_LOAD_ERROR
    status = {
        "ok": True, 
        "device": str(DEVICE), 
        "tables": DB.table_names(),
        "models_loaded": MODELS is not None,
        "models_loading": MODELS_LOADING,
        "fast_mode": FAST_MODE,
        "text_model": TEXT_ID
    }
    if MODELS_LOAD_ERROR:
        status["models_error"] = str(MODELS_LOAD_ERROR)
    return status

@app.get("/search")
def search(q: str = Query(..., min_length=1), topk: int = 10):
    try:
        # text -> text
        models = get_models()
        q_emb = np.asarray(models.text_embed.encode_query([q], normalize_embeddings=True), dtype=np.float32)
        tdf = search_text_chunks(DB, q_emb, topk)
        t2i = siglip_text_embed(models, [q])
        idf = search_images_with_text(DB, t2i, topk)
        merged = merge_and_rank([tdf, idf], limit=topk*2)
        return JSONResponse(merged)
    except Exception as e:
        error_msg = str(e)
        if "still loading" in error_msg:
            return JSONResponse({"error": "Models are loading, please wait a moment and try again"}, status_code=503)
        else:
            return JSONResponse({"error": f"Search failed: {error_msg}"}, status_code=500)

@app.post("/open")
def open_path(payload: dict = Body(...)):
    p = payload.get("path")
    if not p or not Path(p).exists():
        return JSONResponse({"ok": False, "error": "Path not found"}, status_code=400)
    # Reveal in Finder
    os.system(f'open -R "{p}"')
    return {"ok": True}

# Minimal Spotlight-like HTML UI
HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Semantic Finder</title>
<style>
html,body{margin:0;padding:0;background:#0b0b0c;color:#eaeaea;font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial}
.container{max-width:900px;margin:5vh auto;padding:24px}
.box{background:#151517;border:1px solid #2a2a2e;border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,.35)}
.header{padding:16px 20px;border-bottom:1px solid #242428}
.input{width:100%;background:#0f0f12;color:#eaeaea;border:1px solid #2b2b30;border-radius:12px;padding:14px 16px;font-size:18px;outline:none}
.input:focus{border-color:#3f7cff;box-shadow:0 0 0 3px rgba(63,124,255,.25)}
.list{max-height:60vh;overflow:auto;padding:10px 0}
.item{display:flex;gap:16px;align-items:flex-start;padding:14px 20px;border-top:1px solid #242428}
.item:hover{background:#111114}
.type{font-size:12px;background:#26262b;padding:2px 8px;border-radius:999px;margin-top:4px;color:#9aa0a6}
.path{font-size:14px;color:#c9c9d1;word-break:break-all}
.preview{font-size:13px;color:#9aa0a6;margin-top:4px}
.score{margin-left:auto;color:#8ab4ff;font-variant-numeric:tabular-nums}
.btn{background:#2b2b30;color:#eaeaea;border:1px solid #3a3a40;padding:6px 10px;border-radius:8px;cursor:pointer}
.btn:hover{background:#333338}
.footer{display:flex;gap:10px;justify-content:flex-end;padding:12px 16px;border-top:1px solid #242428}
kbd{background:#1b1b1f;border:1px solid #35353a;border-bottom-width:2px;border-radius:6px;padding:2px 6px;color:#c9c9d1;font-size:12px}
</style>
</head>
<body>
<div class="container">
  <div class="box">
    <div class="header">
      <input id="q" class="input" placeholder="Type to search (e.g. ‘green apple’, ‘kalman filter pdf’)"/>
    </div>
    <div id="list" class="list"></div>
    <div class="footer">
      <button class="btn" onclick="revealTop()">Reveal Top <kbd>⏎</kbd></button>
    </div>
  </div>
</div>
<script>
const q = document.getElementById('q');
const list = document.getElementById('list');

async function doSearch() {
  const qs = q.value.trim();
  if (!qs) { list.innerHTML = ''; return; }
  const resp = await fetch('/search?' + new URLSearchParams({q: qs, topk: 10}));
  const data = await resp.json();
  list.innerHTML = data.map((r, i) => `
    <div class="item">
      <div class="type">${r.type}</div>
      <div class="main">
        <div class="path">${r.path}</div>
        ${r.type === 'text' && r.display ? `<div class="preview">${r.display}</div>` : ''}
      </div>
      <div class="score">${r.score.toFixed(3)}</div>
      <div><button class="btn" onclick='reveal(${JSON.stringify(r.path)})'>Reveal</button></div>
    </div>
  `).join('');
}
async function revealTop(){ 
  const first = list.querySelector('.item .btn');
  if(first){ first.click(); }
}
async function reveal(pathStr){
  await fetch('/open', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({path: pathStr})});
}
q.addEventListener('input', () => { doSearch(); });
q.addEventListener('keydown', (e) => { if(e.key === 'Enter') revealTop(); });
q.focus();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML
