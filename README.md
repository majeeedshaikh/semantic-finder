# Semantic Finder

> **Spotlight-style local multimodal semantic search for Mac** - Find your files by meaning, not just names!

A powerful local search engine that understands the *content* of your files using AI embeddings. Search through text documents, images, PDFs, code files, and more using natural language queries.

## **Why I Built This**

I wanted to make my life easier by having a search system that actually understands what's in my files. Instead of remembering exact filenames or folder structures, I can just search for concepts like "that document about machine learning" or "the image with a sunset" and find exactly what I'm looking for.

## ✨ **Features**

- 🔍 **Semantic Search**: Find files by meaning, not just keywords
- 🖼️ **Multimodal**: Search text documents AND images with text queries
- 📱 **Cross-modal**: Find images using text descriptions ("sunset photo", "document with charts")
- �� **100% Local**: No cloud dependencies, your data stays private
- ⚡ **Fast Mode**: Quick development with smaller models
- 🎨 **Beautiful UI**: Clean, dark-themed web interface
- 📁 **File Type Support**: PDFs, Word docs, images, code files, and more
- �� **Smart Caching**: Models load once, then stay in memory
- ��️ **Privacy-focused**: Optional file anonymization for testing

## 🚀 **Quick Start**

### Option 1: Fast Mode (Recommended for Development)
```bash
# Clone the repository
git clone https://github.com/majeeedshaikh/semantic-finder.git
cd semantic-finder

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start in fast mode (uses smaller, faster models)
python start_fast.py
```

### Option 2: Full Mode (Best Quality)
```bash
# Preload models (one-time, takes a few minutes)
python preload_models.py

# Start server
uvicorn semantic_finder.ui:app --reload
```

### Option 3: Index Your Files
```bash
# Index files in the data directory
python -m semantic_finder.indexer

# Or index a custom directory
python -m semantic_finder.indexer --root /path/to/your/files
```

## 🌐 **Using the Web Interface**

1. Open your browser to `http://127.0.0.1:8000`
2. Type your search query in the search box
3. Results appear in real-time as you type
4. Click "Reveal" to open files in Finder
5. Press Enter to reveal the top result

### Example Searches
- `"machine learning algorithms"` - Find documents about ML
- `"sunset photo"` - Find images with sunsets
- `"Python code for data analysis"` - Find relevant code files
- `"PDF about climate change"` - Find specific document types

## 🛠️ **Technical Details**

### **AI Models Used**
- **Text Embeddings**: `google/embeddinggemma-300m` (768-d, multilingual)
- **Fast Mode**: `sentence-transformers/all-mpnet-base-v2` (768-d, faster)
- **Image Embeddings**: `google/siglip-base-patch16-224` (SigLIP vision model)
- **Vector Database**: LanceDB (local, fast, efficient)

### **Supported File Types**
- **Text**: `.txt`, `.md`, `.rtf`, `.csv`, `.json`, `.yaml`
- **Code**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rb`, etc.
- **Documents**: `.pdf` (PyMuPDF), `.docx` (python-docx)
- **Images**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`
- **Audio**: `.mp3` (basic support)

### **Architecture**
