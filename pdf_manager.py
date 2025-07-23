import os
import json
import pickle
from typing import List, Tuple, Dict

# Local module imports
from pdf_processor import process_pdf
from bm25_indexer_v4 import BM25Indexer # New import for v4 BM25 Indexer
from hybrid_retriever_v4 import HybridRetriever # New import for v4 Hybrid Retriever
from embedding_model import load_embedding_model # For chunking and embedding
from text_utils import tokenize_text # For chunking
from text_chunker import chunk_text # New import for robust chunking

""" Commenting out individual program testing code
# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_SOURCE_DIR = os.path.join(BASE_DIR, "pdf_files", "source")
PDF_DESTINATION_BASE_DIR = os.path.join(BASE_DIR, "pdf_files", "destination")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Global instance of HybridRetriever for use in process_and_index_pdf
# This will be initialized once the LLM and embedding model are loaded in rag_agent_v4.py
# and then passed to pdf_manager for use here.
_hybrid_retriever_instance: HybridRetriever = None
_embedding_model_instance = None
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_SOURCE_DIR = os.path.join(BASE_DIR, "pdf_files", "source")
PDF_DESTINATION_BASE_DIR = os.path.join(BASE_DIR, "pdf_files", "destination")

def set_global_retriever_and_embedding_model(retriever: HybridRetriever, embedding_model):
    global _hybrid_retriever_instance
    global _embedding_model_instance
    _hybrid_retriever_instance = retriever
    _embedding_model_instance = embedding_model

def get_pdf_output_dirs(pdf_name_without_ext: str) -> Dict[str, str]:
    """
    Provides standardized absolute paths for all processed artifacts related to a given PDF.
    """
    pdf_specific_output_dir = os.path.join(PDF_DESTINATION_BASE_DIR, pdf_name_without_ext)
    
    return {
        "pdf_specific_output_dir": pdf_specific_output_dir,
        "extracted_images_dir": os.path.join(pdf_specific_output_dir, "extracted_images"),
        "text_chunks_json_path": os.path.join(pdf_specific_output_dir, "text_chunks.json"),
        "bm25_index_pkl_path": os.path.join(pdf_specific_output_dir, "bm25_index.pkl"),
        "retrieval_cache_dir": os.path.join(pdf_specific_output_dir, "retrieval_cache"),
    }

def check_pdf_processed(pdf_name_without_ext: str) -> bool:
    """
    Efficiently determines if a PDF has already been processed and its artifacts saved locally.
    """
    paths = get_pdf_output_dirs(pdf_name_without_ext)
    return os.path.exists(paths["text_chunks_json_path"]) and \
           os.path.exists(paths["bm25_index_pkl_path"])

def process_and_index_pdf(pdf_path: str) -> Tuple[bool, str]:
    """
    Orchestrates the end-to-end processing and indexing of a PDF.
    """
    if _hybrid_retriever_instance is None or _embedding_model_instance is None:
        return False, "HybridRetriever or Embedding Model not initialized in pdf_manager. Call set_global_retriever_and_embedding_model first."

    pdf_name_without_ext = os.path.splitext(os.path.basename(pdf_path))[0]
    paths = get_pdf_output_dirs(pdf_name_without_ext)

    if check_pdf_processed(pdf_name_without_ext):
        print(f"PDF '{pdf_name_without_ext}' already processed. Verifying ChromaDB...")
        # --- BUG FIX: Use get_or_create_collection to handle missing DB ---
        collection = _hybrid_retriever_instance.chroma_client.get_or_create_collection(name=pdf_name_without_ext)
        if collection.count() == 0:
            print(f"Warning: ChromaDB collection '{pdf_name_without_ext}' was missing or empty. Re-ingesting from JSON.")
            with open(paths["text_chunks_json_path"], "r", encoding="utf-8") as f:
                chunks_to_reingest = json.load(f)
            _hybrid_retriever_instance.add_documents_to_chroma(chunks_to_reingest, collection_name=pdf_name_without_ext)
        # --- END BUG FIX ---
        return True, f"PDF '{pdf_name_without_ext}' already processed."

    print(f"Processing and indexing PDF: {pdf_path}...")
    try:
        # Ensure output directories exist
        os.makedirs(paths["pdf_specific_output_dir"], exist_ok=True)
        os.makedirs(paths["extracted_images_dir"], exist_ok=True)

        # 1. Process PDF to extract text and images
        extracted_texts, _ = process_pdf(pdf_path, paths["extracted_images_dir"])
        print(f"Extracted {len(extracted_texts)} pages of text from {pdf_path}")

        # 2. Chunk the text and prepare for indexing
        # Use the robust chunking logic from text_chunker.py
        chunks_with_metadata = chunk_text(extracted_texts, pdf_path)
        print(f"Created {len(chunks_with_metadata)} chunks from the PDF using text_chunker.")

        # Add pdf_name to metadata for ChromaDB filtering if needed
        for chunk in chunks_with_metadata:
            chunk["metadata"]["pdf_name"] = pdf_name_without_ext

        if not chunks_with_metadata:
            return False, f"No content extracted or chunked from {pdf_path}."

        # Save chunks to JSON for BM25 and future reference
        with open(paths["text_chunks_json_path"], "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, indent=4)
        print(f"Chunks saved to {paths["text_chunks_json_path"]}")

        # 3. Create and save BM25 Index
        bm25_indexer = BM25Indexer()
        bm25_indexer.create_and_save_bm25_index(chunks_with_metadata, paths["bm25_index_pkl_path"])

        # 4. Add documents to ChromaDB (specific collection for this PDF)
        _hybrid_retriever_instance.add_documents_to_chroma(chunks_with_metadata, collection_name=pdf_name_without_ext)
        print(f"Documents added to ChromaDB collection '{pdf_name_without_ext}'.")

        return True, f"Successfully processed and indexed {pdf_name_without_ext}."

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return False, f"Failed to process PDF {pdf_name_without_ext}: {e}"