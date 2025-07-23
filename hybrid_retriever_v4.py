import os
import json
import pickle
from collections import defaultdict
from typing import List, Dict

# Local module imports
from vector_db import get_chroma_client
from embedding_model import load_embedding_model
from text_utils import tokenize_text # Import from the new utility file

class HybridRetriever:
    """
    A retriever that combines results from both dense (ChromaDB) and sparse (BM25)
    retrieval methods using Reciprocal Rank Fusion (RRF).
    This version supports dynamic loading of PDF-specific data and per-PDF ChromaDB collections.
    """
    def __init__(self, db_dir: str, embedding_model_name: str):
        """
        Initializes the retriever.

        Args:
            db_dir (str): Path to the ChromaDB database directory.
            embedding_model_name (str): The name of the sentence-transformer model.
        """
        print("Initializing Hybrid Retriever (v4)...")
        self.db_dir = db_dir
        
        # 1. Load Embedding Model
        self.embedding_model = load_embedding_model(embedding_model_name)
        if not self.embedding_model:
            raise ValueError("Failed to load embedding model.")

        # 2. Initialize ChromaDB Client
        self.chroma_client = get_chroma_client(db_dir)
        print(f"ChromaDB client initialized at '{db_dir}'.")

        # Initialize active components to None/empty
        self.active_chroma_collection = None
        self.active_bm25_index = None
        self.active_bm25_chunk_ids = []
        self.active_all_chunks = []
        self.active_content_map = {}

    def load_pdf_for_retrieval(self, pdf_name_without_ext: str):
        """
        Loads and activates the retrieval components for a specific PDF.
        """
        print(f"Loading data for PDF: {pdf_name_without_ext} for retrieval...")
        # Import pdf_manager here to avoid circular dependency at top level
        import pdf_manager

        paths = pdf_manager.get_pdf_output_dirs(pdf_name_without_ext)

        # Set active ChromaDB collection
        self.active_chroma_collection = self.chroma_client.get_or_create_collection(name=pdf_name_without_ext)
        print(f"Active ChromaDB collection set to '{pdf_name_without_ext}'.")

        # Load BM25 index and chunks
        try:
            with open(paths["bm25_index_pkl_path"], "rb") as f:
                bm25_data = pickle.load(f)
                self.active_bm25_index = bm25_data["index"]
                self.active_bm25_chunk_ids = bm25_data["chunk_ids"]
            print("Active BM25 index loaded.")

            with open(paths["text_chunks_json_path"], "r", encoding="utf-8") as f:
                self.active_all_chunks = json.load(f)
                self.active_content_map = {chunk['metadata']['chunk_id']: chunk['content'] for chunk in self.active_all_chunks}
            print("Active text chunks loaded into memory.")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required files for PDF '{pdf_name_without_ext}' not found. Please ensure it's processed: {e}")
        except Exception as e:
            raise Exception(f"Error loading data for PDF '{pdf_name_without_ext}': {e}")

    def add_documents_to_chroma(self, new_chunks: List[dict], collection_name: str):
        """
        Adds a list of new document chunks to a specific ChromaDB collection.
        """
        if not new_chunks:
            print("No new chunks to add to ChromaDB.")
            return

        print(f"Adding {len(new_chunks)} documents to ChromaDB collection '{collection_name}'...")
        target_collection = self.chroma_client.get_or_create_collection(name=collection_name)

        ids = [chunk['metadata']['chunk_id'] for chunk in new_chunks]
        documents = [chunk['content'] for chunk in new_chunks]
        metadatas = [chunk['metadata'] for chunk in new_chunks]

        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()

        target_collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully added {len(new_chunks)} documents to collection '{collection_name}'.")

    def dense_search(self, query_text: str, n_results: int = 10) -> list[str]:
        """
        Performs a dense vector search in the active ChromaDB collection.

        Returns:
            list[str]: A list of chunk IDs.
        """
        if self.active_chroma_collection is None:
            raise RuntimeError("No active ChromaDB collection. Please load a PDF first.")

        query_embedding = self.embedding_model.encode([query_text]).tolist()
        results = self.active_chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['metadatas']
        )
        return [meta['chunk_id'] for meta in results['metadatas'][0]]

    def sparse_search(self, query_text: str, n_results: int = 10) -> list[str]:
        """
        Performs a sparse keyword search using the active BM25 index.

        Returns:
            list[str]: A list of chunk IDs.
        """
        if self.active_bm25_index is None:
            raise RuntimeError("No active BM25 index. Please load a PDF first.")

        tokenized_query = tokenize_text(query_text)
        return self.active_bm25_index.get_top_n(tokenized_query, self.active_bm25_chunk_ids, n=n_results)

    def retrieve(self, query_texts: list[str], n_dense: int = 10, n_sparse: int = 10, rrf_k: int = 60) -> list[tuple[str, float]]:
        """
        Performs hybrid search over a list of queries. It re-ranks results based on
        the consensus (occurrence count) between query variations, using a reciprocal
        rank score as a tie-breaker.

        Args:
            query_texts (list[str]): A list of queries (original + transformed).
            n_dense (int): Number of results to fetch per query for dense search.
            n_sparse (int): Number of results to fetch per query for sparse search.
            rrf_k (int): (Unused in this implementation) Constant for RRF calculation.

        Returns:
            list[tuple[str, tuple[int, float]]]: A sorted list of
            (chunk_id, (count, score)) tuples.
        """
        if self.active_chroma_collection is None or self.active_bm25_index is None:
            raise RuntimeError("Retrieval components not active. Please load a PDF first.")

        all_results = defaultdict(int)
        chunk_scores = defaultdict(float)

        print(f"\nPerforming dense and sparse search for {len(query_texts)} queries...")
        for query in query_texts:
            dense_results = self.dense_search(query, n_results=n_dense)
            sparse_results = self.sparse_search(query, n_results=n_sparse)

            # Combine and count occurrences
            for chunk_id in dense_results + sparse_results:
                all_results[chunk_id] += 1

            # Score based on rank (Reciprocal Rank)
            for i, chunk_id in enumerate(dense_results):
                chunk_scores[chunk_id] += 1 / (i + 1)
            for i, chunk_id in enumerate(sparse_results):
                chunk_scores[chunk_id] += 1 / (i + 1)

        # Combine scores and counts for final ranking
        combined_scores = {}
        for chunk_id, count in all_results.items():
            # Prioritize documents found by multiple queries, then by rank score
            combined_scores[chunk_id] = (count, chunk_scores[chunk_id])

        # Sort by count (desc), then by score (desc)
        reranked_results = sorted(combined_scores.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)

        print("Search complete. Results fused and re-ranked based on consensus.")
        return reranked_results

    def get_active_content_map(self) -> Dict[str, str]:
        """
        Returns the content map for the currently active PDF.
        """
        return self.active_content_map
