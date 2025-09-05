#!/usr/bin/env python3
"""
Preload and cache models to avoid loading delays during search.
This downloads and caches the models so they're ready when needed.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from semantic_finder.config import TEXT_EMBED_MODEL, IMAGE_EMBED_MODEL
from semantic_finder.search import load_models

def main():
    print("🔄 Preloading models for faster startup...")
    print(f"   Text model: {TEXT_EMBED_MODEL}")
    print(f"   Image model: {IMAGE_EMBED_MODEL}")
    print()
    
    try:
        # Load models - this will download and cache them
        models = load_models(TEXT_EMBED_MODEL, IMAGE_EMBED_MODEL)
        print("✅ Models preloaded successfully!")
        print("   - Models are now cached and ready for fast loading")
        print("   - You can now start the server with: uvicorn semantic_finder.ui:app")
        
    except Exception as e:
        print(f"❌ Failed to preload models: {e}")
        print("   - You can still use fast mode: python start_fast.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
