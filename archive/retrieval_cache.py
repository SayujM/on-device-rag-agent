
import os
import sqlite3
import json
import uuid

import chromadb
from sentence_transformers import SentenceTransformer

class RetrievalCache:
    """
    A semantic cache for retrieval results that stores query embeddings in ChromaDB
    and the corresponding results (chunk IDs) in a SQLite database.
    """
    def __init__(self, cache_dir: str, embedding_model: SentenceTransformer, 
                 collection_name: str = "retrieval_cache", similarity_threshold: float = 0.67):
        """
        Initializes the cache.

        Args:
            cache_dir (str): Directory to store the SQLite DB and ChromaDB files.
            embedding_model (SentenceTransformer): The loaded embedding model instance.
            collection_name (str): Name for the ChromaDB collection for cached queries.
            similarity_threshold (float): The threshold for a cache hit (0.0 to 1.0).
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

        # 1. Initialize SQLite Database for storing results
        self.sqlite_path = os.path.join(cache_dir, "cache_results.db")
        self._init_sqlite()

        # 2. Initialize ChromaDB for storing query embeddings
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(cache_dir, "chroma_cache"))
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Specify cosine similarity
        )
        print(f"Retrieval cache initialized. SQLite at: {self.sqlite_path}")

    def _init_sqlite(self):
        """
        Creates the SQLite table if it doesn't exist.
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    query_id TEXT PRIMARY KEY,
                    results_json TEXT NOT NULL
                )
            ''')
            conn.commit()

    def check_cache(self, query_text: str) -> list[str] | None:
        """
        Checks if a semantically similar query exists in the cache.

        Args:
            query_text (str): The incoming user query.

        Returns:
            list[str] | None: The cached list of chunk IDs if a hit is found, otherwise None.
        """
        if self.chroma_collection.count() == 0:
            return None # Cache is empty

        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        results = self.chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=1, # Find the single most similar query
            include=["distances"] # IDs are returned by default
        )

        # Check if any result was found and if it meets the similarity threshold
        if results["ids"][0] and (1 - results['distances'][0][0]) >= self.similarity_threshold:
            most_similar_id = results["ids"][0][0]
            print(f"Cache HIT. Found similar query (ID: {most_similar_id}). Retrieving results from SQLite.")
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT results_json FROM cache WHERE query_id = ?", (most_similar_id,))
                row = cursor.fetchone()
                return json.loads(row[0]) if row else None
        
        print("Cache MISS. No sufficiently similar query found.")
        return None

    def add_to_cache(self, query_text: str, results: list[str]):
        """
        Adds a new query and its results to the cache.

        Args:
            query_text (str): The user query.
            results (list[str]): The list of retrieved chunk IDs.
        """
        query_id = str(uuid.uuid4())
        query_embedding = self.embedding_model.encode([query_text]).tolist()[0]

        # 1. Add query embedding to ChromaDB
        self.chroma_collection.add(
            ids=[query_id],
            embeddings=[query_embedding]
        )

        # 2. Add results to SQLite
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO cache (query_id, results_json) VALUES (?, ?)", (query_id, json.dumps(results)))
            conn.commit()
        print(f"Added new entry to cache with ID: {query_id}")

if __name__ == '__main__':
    # --- Configuration for Standalone Test ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.join(BASE_DIR, "retrieval_cache_test")
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

    # Clean up previous test run
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    try:
        # 1. Initialize components
        print("--- Initializing test components ---")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        cache = RetrievalCache(cache_dir=CACHE_DIR, embedding_model=embedding_model, similarity_threshold = 0.67)

        # 2. First run (should be a cache miss)
        print("\n--- First Run: Testing Cache Miss ---")
        query1 = "What are the main ideas behind the Toyota Way?"
        cached_results1 = cache.check_cache(query1)
        if cached_results1 is None:
            # Simulate a retrieval process
            retrieved_data = ["chunk_1", "chunk_5", "chunk_12"]
            print(f"Simulating retrieval, got results: {retrieved_data}")
            cache.add_to_cache(query1, retrieved_data)
        
        # 3. Second run with identical query (should be a cache hit)
        print("\n--- Second Run: Testing Cache Hit (Identical Query) ---")
        cached_results2 = cache.check_cache(query1)
        if cached_results2:
            print(f"Successfully retrieved from cache: {cached_results2}")
            assert cached_results2 == retrieved_data

        # 4. Third run with semantically similar query (should be a cache hit)
        print("\n--- Third Run: Testing Cache Hit (Similar Query) ---")
        query3 = "Tell me about the principles of the Toyota Production System."
        cached_results3 = cache.check_cache(query3)
        if cached_results3:
            print(f"Successfully retrieved from cache: {cached_results3}")
            assert cached_results3 == retrieved_data

        # 5. Fourth run with a different query (should be a cache miss)
        print("\n--- Fourth Run: Testing Cache Miss (Different Query) ---")
        query4 = "What is the capital of Japan?"
        cached_results4 = cache.check_cache(query4)
        if cached_results4 is None:
            print("Correctly resulted in a cache miss as expected.")

    except Exception as e:
        print(f"An error occurred during the test: {e}")


