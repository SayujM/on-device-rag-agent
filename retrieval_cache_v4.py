import sqlite3
import json
import os
import uuid
from typing import List

# Third-party imports
import chromadb
from sentence_transformers import SentenceTransformer

class RetrievalCache:
    """
    A semantic cache for retrieval results, using ChromaDB for vector search and
    SQLite for storing results. Each PDF will have its own isolated cache directory.
    This version corrects the flawed implementation by reverting to the robust,
    ChromaDB-based logic from the original cache.
    """
    def __init__(self, cache_dir: str, embedding_model: SentenceTransformer,
                 collection_name: str = "retrieval_cache_v4", similarity_threshold: float = 0.95):
        """
        Initializes the RetrievalCache.

        Args:
            cache_dir (str): The absolute path to the directory where cache files will be stored.
            embedding_model: The embedding model instance to use for query similarity.
            collection_name (str): Name for the ChromaDB collection for cached queries.
            similarity_threshold (float): Cosine similarity threshold for cache hits.
        """
        print(f"Initializing RetrievalCache (v4 - Corrected) at: {cache_dir}")
        self.cache_dir = cache_dir
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

        os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Initialize SQLite Database for storing results
        self.sqlite_path = os.path.join(self.cache_dir, "cache_results.db")
        self._init_sqlite()

        # 2. Initialize ChromaDB for storing query embeddings
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(self.cache_dir, "chroma_cache"))
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print("RetrievalCache initialized successfully.")

    def _init_sqlite(self):
        """
        Initializes the SQLite database and creates the cache table if it doesn't exist.
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    query_id TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    retrieved_ids TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add_to_cache(self, query_text: str, retrieved_ids: List[str]):
        """
        Adds a query and its retrieved document IDs to the cache.
        """
        query_id = str(uuid.uuid4())
        query_embedding = self.embedding_model.encode([query_text]).tolist()[0]

        # 1. Add query embedding to ChromaDB
        self.chroma_collection.add(
            ids=[query_id],
            embeddings=[query_embedding],
            metadatas=[{"query_text": query_text}]
        )

        # 2. Add results to SQLite
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO cache (query_id, query_text, retrieved_ids) VALUES (?, ?, ?)",
                (query_id, query_text, json.dumps(retrieved_ids))
            )
            conn.commit()
        print(f"Added query '{query_text[:50]}...' to cache.")


    def check_cache(self, query_text: str) -> List[str] | None:
        """
        Checks if a semantically similar query exists in the cache using ChromaDB.
        Returns the cached retrieved IDs if a hit is found, otherwise None.
        """
        if self.chroma_collection.count() == 0:
            return None

        query_embedding = self.embedding_model.encode([query_text]).tolist()

        results = self.chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=1,
            include=["distances"]
        )

        if results["ids"][0] and (1 - results['distances'][0][0]) >= self.similarity_threshold:
            most_similar_id = results["ids"][0][0]
            distance = results['distances'][0][0]
            similarity = 1 - distance
            print(f"Cache HIT! Similarity: {similarity:.4f} >= {self.similarity_threshold}")

            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT retrieved_ids FROM cache WHERE query_id = ?", (most_similar_id,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                else:
                    # This case is unlikely but good to handle (embedding exists but result doesn't)
                    print(f"Cache INCONSISTENCY. Found embedding for ID {most_similar_id} but no result in SQLite.")
                    return None
        else:
            if results["ids"][0]:
                distance = results['distances'][0][0]
                similarity = 1 - distance
            return None

if __name__ == "__main__":
    # --- Standalone Test ---
    print("\nRunning RetrievalCache (v4 - Corrected) standalone test...")
    # Local import for testing
    from embedding_model import load_embedding_model
    import shutil

    test_cache_dir = "./test_cache_v4"
    test_embedding_model = load_embedding_model("all-MiniLM-L6-v2")

    # Clean up previous test cache
    if os.path.exists(test_cache_dir):
        shutil.rmtree(test_cache_dir)

    cache = RetrievalCache(test_cache_dir, test_embedding_model, similarity_threshold=0.95)

    # Test 1: Add to cache and check for exact hit
    print("\n--- Test 1: Exact Hit ---")
    query1 = "What is the capital of France?"
    ids1 = ["doc1", "doc2"]
    cache.add_to_cache(query1, ids1)
    retrieved = cache.check_cache(query1)
    assert retrieved == ids1, f"Test 1 Failed: Expected {ids1}, got {retrieved}"
    print("Test 1 Passed: Exact hit.")

    # Test 2: Check for semantic hit
    print("\n--- Test 2: Semantic Hit ---")
    query2 = "What is the capital city of France?" # Slightly different
    retrieved = cache.check_cache(query2)
    assert retrieved == ids1, f"Test 2 Failed: Expected {ids1}, got {retrieved}"
    print("Test 2 Passed: Semantic hit.")

    # Test 3: Check for miss (semantically different)
    print("\n--- Test 3: Cache Miss ---")
    query3 = "What is the highest mountain?"
    retrieved = cache.check_cache(query3)
    assert retrieved is None, f"Test 3 Failed: Expected None, got {retrieved}"
    print("Test 3 Passed: Cache miss.")

    # Test 4: Check for miss (below threshold)
    print("\n--- Test 4: Below Threshold Miss ---")
    query4 = "What is France known for?" # Related but should be different enough
    retrieved = cache.check_cache(query4)
    assert retrieved is None, f"Test 4 Failed: Expected None, got {retrieved}"
    print("Test 4 Passed: Below threshold miss.")


    # Test 5: Add another entry
    print("\n--- Test 5: Add Second Entry ---")
    query5 = "Who painted the Mona Lisa?"
    ids5 = ["doc3"]
    cache.add_to_cache(query5, ids5)
    retrieved = cache.check_cache(query5)
    assert retrieved == ids5, f"Test 5 Failed: Expected {ids5}, got {retrieved}"
    print("Test 5 Passed: Added new entry.")

    # Clean up test cache
    if os.path.exists(test_cache_dir):
        shutil.rmtree(test_cache_dir)
    print("\nStandalone test complete. Test cache removed.")