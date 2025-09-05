# 🚀 Fast Mode for Semantic Finder

## Problem Solved
The original `google/embeddinggemma-300m` model is 1.21GB and takes a very long time to download and load, making development and testing slow.

## Solution: Fast Mode
Fast mode uses a smaller, faster model (`sentence-transformers/all-mpnet-base-v2`) that:
- ✅ Is much faster to download and load
- ✅ Has the same 768-dimensional embeddings (compatible with existing database)
- ✅ Still provides good semantic search quality
- ✅ Perfect for development and testing

## How to Use

### Option 1: Use the Fast Mode Script
```bash
python start_fast.py
```

### Option 2: Set Environment Variable
```bash
export SF_FAST_MODE=true
uvicorn semantic_finder.ui:app --reload
```

### Option 3: Preload Models (One-time)
```bash
python preload_models.py  # Downloads and caches the full models
uvicorn semantic_finder.ui:app --reload
```

## Performance Comparison

| Mode | Model | Size | Load Time | Quality |
|------|-------|------|-----------|---------|
| **Fast Mode** | `all-mpnet-base-v2` | ~420MB | ~10-30s | Good |
| **Full Mode** | `google/embeddinggemma-300m` | ~1.21GB | ~2-5min | Excellent |

## Features
- 🔄 **Lazy Loading**: Models only load when first search is made
- 💾 **Caching**: Once loaded, models stay in memory
- ⚡ **Fast Startup**: Server starts immediately, models load on demand
- 🛡️ **Error Handling**: Graceful handling of loading states
- 📊 **Status Endpoint**: Check model loading status via `/health`

## API Endpoints
- `GET /health` - Check server and model status
- `GET /search?q=query&topk=10` - Search (loads models if needed)
- `POST /open` - Open files in Finder

## Development Workflow
1. Use **Fast Mode** for development and testing
2. Use **Full Mode** for production or when you need the best quality
3. Models are cached after first load, so subsequent searches are fast
