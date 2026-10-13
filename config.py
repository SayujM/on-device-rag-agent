import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Automatic Path Resolution for Modular Imports ---
# Ensures all subdirectories in src/ can import each other seamlessly
MODULE_PATHS = [
    BASE_DIR,
    os.path.join(BASE_DIR, "src"),
    os.path.join(BASE_DIR, "src", "ingestion"),
    os.path.join(BASE_DIR, "src", "retrieval"),
    os.path.join(BASE_DIR, "src", "models"),
    os.path.join(BASE_DIR, "src", "agent"),
    os.path.join(BASE_DIR, "scripts"),
    os.path.join(BASE_DIR, "tests"),
]

for p in MODULE_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

# --- Model Configuration ---
MODEL_FILENAME = "gemma-3-4b-it-Q4_K_M.gguf"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Directory & Storage Paths ---
PDF_STAGING_DIR = os.path.join(BASE_DIR, "pdf_files", "staging")
PDF_SOURCE_DIR = os.path.join(BASE_DIR, "pdf_files", "source")
PDF_QUARANTINE_DIR = os.path.join(BASE_DIR, "pdf_files", "quarantine")
PDF_DESTINATION_BASE_DIR = os.path.join(BASE_DIR, "pdf_files", "destination")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "langgraph_checkpoints")
CHECKPOINT_DB_PATH = os.path.join(CHECKPOINT_DIR, "checkpoints.sqlite")

# --- Persistent Metadata & Data Storage ---
DATA_DIR = os.path.join(BASE_DIR, "data")
PROFILE_DB_PATH = os.path.join(DATA_DIR, "user_profiles.db")
PDF_SUMMARIES_PATH = os.path.join(DATA_DIR, "pdf_summaries.json")
USER_MEMORY_PATH = os.path.join(DATA_DIR, "user_memory.json")
