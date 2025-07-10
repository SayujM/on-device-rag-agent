import os
import json
import pickle
from collections import defaultdict

# Local module imports
from vector_db import get_chroma_client, COLLECTION_NAME
from embedding_model import load_embedding_model
from bm25_indexer import tokenize_text

class HybridRetriever:
    """
    A retriever that combines results from both dense (ChromaDB) and sparse (BM25)
    retrieval methods using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, db_dir: str, pdf_specific_dir: str, embedding_model_name: str):
        """
        Initializes the retriever by loading all necessary components.

        Args:
            db_dir (str): Path to the ChromaDB database directory.
            pdf_specific_dir (str): Path to the directory for the specific PDF,
                                  containing chunks and the BM25 index.
            embedding_model_name (str): The name of the sentence-transformer model.
        """
        print("Initializing Hybrid Retriever...")
        # 1. Load Embedding Model
        self.embedding_model = load_embedding_model(embedding_model_name)
        if not self.embedding_model:
            raise ValueError("Failed to load embedding model.")

        # 2. Initialize ChromaDB Client and Collection
        self.chroma_client = get_chroma_client(db_dir)
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"ChromaDB collection '{COLLECTION_NAME}' loaded.")

        # 3. Load BM25 Index
        bm25_index_path = os.path.join(pdf_specific_dir, "bm25_index.pkl")
        if not os.path.exists(bm25_index_path):
            raise FileNotFoundError(f"BM25 index not found at: {bm25_index_path}")
        with open(bm25_index_path, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25_index = bm25_data["index"]
            self.bm25_chunk_ids = bm25_data["chunk_ids"]
        print("BM25 index loaded.")

        # 4. Load all chunks content for result display
        chunks_json_path = os.path.join(pdf_specific_dir, "text_chunks.json")
        if not os.path.exists(chunks_json_path):
            raise FileNotFoundError(f"Chunks JSON file not found at: {chunks_json_path}")
        with open(chunks_json_path, "r", encoding="utf-8") as f:
            self.all_chunks = json.load(f)
            self.content_map = {chunk['metadata']['chunk_id']: chunk['content'] for chunk in self.all_chunks}
        print("All text chunks loaded into memory.")

    def dense_search(self, query_text: str, n_results: int = 10) -> list[str]:
        """
        Performs a dense vector search in ChromaDB.

        Returns:
            list[str]: A list of chunk IDs.
        """
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['metadatas']
        )
        return [meta['chunk_id'] for meta in results['metadatas'][0]]

    def sparse_search(self, query_text: str, n_results: int = 10) -> list[str]:
        """
        Performs a sparse keyword search using the BM25 index.

        Returns:
            list[str]: A list of chunk IDs.
        """
        tokenized_query = tokenize_text(query_text)
        # The get_top_n method expects the corpus (or in our case, the IDs mapping to it)
        return self.bm25_index.get_top_n(tokenized_query, self.bm25_chunk_ids, n=n_results)

    def retrieve(self, query_texts: list[str], n_dense: int = 10, n_sparse: int = 10, rrf_k: int = 60) -> list[tuple[str, float]]:
        """
        Performs hybrid search over a list of queries and fuses the results.

        Args:
            query_texts (list[str]): A list of queries (original + transformed).
            n_dense (int): Number of results to fetch per query for dense search.
            n_sparse (int): Number of results to fetch per query for sparse search.
            rrf_k (int): Constant used in the RRF calculation.

        Returns:
            list[tuple[str, float]]: A sorted list of (chunk_id, score) tuples.
        """
        all_dense_results = []
        all_sparse_results = []

        print(f"\nPerforming dense and sparse search for {len(query_texts)} queries...")
        for query in query_texts:
            all_dense_results.append(self.dense_search(query, n_results=n_dense))
            all_sparse_results.append(self.sparse_search(query, n_results=n_sparse))
        
        # --- Reciprocal Rank Fusion (RRF) --- 
        fused_scores = defaultdict(float)
        
        # Process dense results
        for result_list in all_dense_results:
            for i, doc_id in enumerate(result_list):
                fused_scores[doc_id] += 1 / (rrf_k + i + 1) # i is 0-indexed rank

        # Process sparse results
        for result_list in all_sparse_results:
            for i, doc_id in enumerate(result_list):
                fused_scores[doc_id] += 1 / (rrf_k + i + 1)

        # Sort documents by their fused scores in descending order
        reranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        print("Search complete. Results fused and re-ranked.")
        return reranked_results

if __name__ == "__main__":
    # --- Configuration for Standalone Test ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
    PDF_NAME = "The-toyota-way-second-edition-chapter_1"
    PDF_SPECIFIC_DIR = os.path.join(BASE_DIR, "pdf_files", "destination", PDF_NAME)
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    GGUF_MODEL_PATH = os.path.join(BASE_DIR, "gemma-3n-E4B-it-Q4_K_M.gguf")

    try:
        # 1. Get transformed queries (from the previous step)
        from query_transformer import QueryTransformer
        query_transformer = QueryTransformer(model_path=GGUF_MODEL_PATH)
        original_query = "What is the Toyota Way model?"
        transformed_queries = query_transformer.transform_query(original_query)
        all_queries = [original_query] + transformed_queries

        # 2. Initialize the retriever
        retriever = HybridRetriever(
            db_dir=CHROMA_DB_DIR,
            pdf_specific_dir=PDF_SPECIFIC_DIR,
            embedding_model_name=EMBEDDING_MODEL_NAME
        )

        # 3. Perform hybrid retrieval
        final_results = retriever.retrieve(all_queries)

        # 4. Print the top 5 results
        print("\n--- Top 5 Hybrid Search Results ---")
        for i, (chunk_id, score) in enumerate(final_results[:5]):
            print(f"Rank {i+1}: (Score: {score:.4f})")
            print(f"  Chunk ID: {chunk_id}")
            print(f"  Content: {retriever.content_map.get(chunk_id, 'Content not found...')[:300]}...")
            print()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("A required file was not found. Please ensure all previous steps have been run successfully.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
