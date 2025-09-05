from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"            # your files live here
INDEX_DIR = PROJECT_ROOT / ".lancedb"       # local vector store dir

# Models
TEXT_EMBED_MODEL = "google/embeddinggemma-300m"  # 768-d, multilingual
IMAGE_EMBED_MODEL = "google/siglip-base-patch16-224"

# Text chunking
CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 80

# Batch sizes
TEXT_BATCH = 64
IMAGE_BATCH = 32

# Table names
TABLE_TEXT = "text_chunks"
TABLE_IMAGE = "images"

# File type sets
TEXT_EXTS = {".txt", ".md", ".rtf", ".csv", ".json", ".yaml", ".yml"}
CODE_EXTS = {".py", ".js", ".ts", ".java", ".cpp", ".hpp", ".c", ".h", ".cs", ".php", ".go", ".rb", ".rs", ".kt", ".scala"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}   # (.doc not supported by python-docx)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Misc
PREVIEW_CHARS = 400
