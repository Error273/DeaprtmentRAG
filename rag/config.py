"""
Конфигурация RAG-системы.
Все настройки в одном месте — легко менять модель, адрес Qdrant и параметры.
"""

from pathlib import Path

# ── Пути ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks" / "chunks.json"
DOC_TEXTS_PATH = DATA_DIR / "chunks" / "doc_texts.json"

# ── Модель эмбеддингов ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384  # размер вектора для этой модели

# ── Qdrant ───────────────────────────────────────────────────────────
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "department_chunks"

# ── Чанковка ─────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Retriever ────────────────────────────────────────────────────────
TOP_K = 5  # сколько документов (страниц) возвращать при поиске
SEMANTIC_TOP_K = 15  # сколько чанков брать из семантического поиска (до дедупликации)
KEYWORD_TOP_K = 10  # сколько чанков брать из keyword-поиска
