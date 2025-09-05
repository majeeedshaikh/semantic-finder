#!/usr/bin/env python3
"""
Start the semantic finder server in fast mode with smaller, faster models.
This is much faster for development and testing.
"""

import os
import subprocess
import sys

def main():
    # Set environment variables for fast mode
    env = os.environ.copy()
    env["SF_FAST_MODE"] = "true"
    
    print("🚀 Starting Semantic Finder in FAST MODE")
    print("   - Using smaller, faster models")
    print("   - Models will load on first search request")
    print("   - Perfect for development and testing")
    print()
    
    try:
        # Start uvicorn with the environment variables
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "semantic_finder.ui:app", 
            "--reload", 
            "--host", "127.0.0.1", 
            "--port", "8000"
        ], env=env)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    main()
