import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Configuration
MODEL_FILENAME = "gemma-3-4b-it-Q4_K_M.gguf"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Directory & Database Paths
PDF_STAGING_DIR = os.path.join(BASE_DIR, "pdf_files", "staging")
PDF_SOURCE_DIR = os.path.join(BASE_DIR, "pdf_files", "source")
PDF_QUARANTINE_DIR = os.path.join(BASE_DIR, "pdf_files", "quarantine")
PDF_DESTINATION_BASE_DIR = os.path.join(BASE_DIR, "pdf_files", "destination")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "langgraph_checkpoints")
CHECKPOINT_DB_PATH = os.path.join(CHECKPOINT_DIR, "checkpoints.sqlite")
PROFILE_DB_PATH = os.path.join(BASE_DIR, "user_profiles.db")
PDF_SUMMARIES_PATH = os.path.join(BASE_DIR, "pdf_summaries.json")
