import os
import glob
from hybrid_retriever_v4 import HybridRetriever
from embedding_model import load_embedding_model
import pdf_manager
from config import PDF_SOURCE_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME


def main():
    print("=" * 60)
    print("PRE-INDEXING SOURCE CHAPTERS (BM25 + ChromaDB)")
    print("=" * 60)

    print("Loading embedding model:", EMBEDDING_MODEL_NAME)
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

    print("Initializing Hybrid Retriever & ChromaDB...")
    retriever = HybridRetriever(db_dir=CHROMA_DB_DIR, embedding_model_name=EMBEDDING_MODEL_NAME)
    pdf_manager.set_global_retriever_and_embedding_model(retriever, embedding_model)

    pdf_files = sorted(glob.glob(os.path.join(PDF_SOURCE_DIR, "*.pdf")))
    print(f"Found {len(pdf_files)} PDF(s) to index in {PDF_SOURCE_DIR}:\n")

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        print(f"\n--- Indexing: {file_name} ---")
        success, msg = pdf_manager.process_and_index_pdf(pdf_path)
        if success:
            print(f"[PASS] {msg}")
        else:
            print(f"[FAIL] {msg}")

    print("\n" + "=" * 60)
    print("All PDFs indexed into ChromaDB & BM25 successfully!")
    print("=" * 60)

    # Seed User Profile for Sayuj
    print("\nSeeding user profile into SQLite database...")
    import json
    from profile_manager import ProfileManager
    from config import PROFILE_DB_PATH, USER_MEMORY_PATH

    if os.path.exists(USER_MEMORY_PATH):
        with open(USER_MEMORY_PATH, "r", encoding="utf-8") as f:
            user_data = json.load(f)
        pm = ProfileManager(PROFILE_DB_PATH)
        pm.save_profile("Sayuj", user_data)
        pm.close()
        print(f"[PASS] User profile for 'Sayuj' seeded into: {PROFILE_DB_PATH}")


if __name__ == "__main__":
    main()
