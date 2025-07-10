
import os
import json
import pickle
import re
from rank_bm25 import BM25Okapi

# A simple list of common English stop words for basic filtering.
# For a more robust solution in a production system, a library like NLTK would be preferable.
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot',
    'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
    'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor',
    'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
    'why', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
}

def tokenize_text(text: str) -> list[str]:
    """
    A more robust tokenizer that lowercases, removes punctuation, and removes stop words.
    This ensures that the search is case-insensitive and focuses on meaningful terms.

    Args:
        text (str): The input string to tokenize.

    Returns:
        list[str]: A list of processed tokens.
    """
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # 3. Split into words (tokenize)
    tokens = text.split()
    # 4. Remove stop words
    return [token for token in tokens if token not in STOP_WORDS]

def create_and_save_bm25_index(chunks_json_path: str, output_index_path: str):
    """
    Loads text chunks, creates a BM25 index using improved tokenization,
    and saves the index and chunk IDs to a pickle file.
    """
    print(f"Loading text chunks from: {chunks_json_path}")
    try:
        with open(chunks_json_path, "r", encoding="utf-8") as f:
            chunks_with_metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: Chunks JSON file not found at {chunks_json_path}")
        print("Please ensure text_chunker.py has been run first.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {chunks_json_path}: {e}")
        return None

    if not chunks_with_metadata:
        print("No chunks found in the file. Cannot create index.")
        return None

    corpus_contents = [chunk["content"] for chunk in chunks_with_metadata]
    chunk_ids = [chunk["metadata"]["chunk_id"] for chunk in chunks_with_metadata]

    # --- Improved Tokenization ---
    print(f"Tokenizing {len(corpus_contents)} documents with improved tokenizer...")
    tokenized_corpus = [tokenize_text(doc) for doc in corpus_contents]
    print("Tokenization complete.")
    # --- End Improvement ---

    # Create the BM25 index
    print("Creating BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 index created successfully.")

    bm25_data = {
        "index": bm25,
        "chunk_ids": chunk_ids
    }

    print(f"Saving BM25 index to: {output_index_path}")
    try:
        with open(output_index_path, "wb") as f_out:
            pickle.dump(bm25_data, f_out)
        print("BM25 index saved successfully.")
        return bm25_data
    except Exception as e:
        print(f"Error saving BM25 index to file: {e}")
        return None

def test_bm25_query(index_path: str, query: str, chunks_json_path: str):
    """
    Loads a saved BM25 index, performs a test query using improved tokenization,
    and prints the top results.
    """
    if not os.path.exists(index_path):
        print(f"Index file not found: {index_path}")
        return

    print(f"\n--- Testing Query on Saved Index ---")
    print(f"Loading index from: {index_path}")
    try:
        with open(index_path, "rb") as f_in:
            bm25_data = pickle.load(f_in)
        bm25_index = bm25_data["index"]
        chunk_ids = bm25_data["chunk_ids"]

        with open(chunks_json_path, "r", encoding="utf-8") as f:
            original_chunks = json.load(f)
            content_map = {chunk['metadata']['chunk_id']: chunk['content'] for chunk in original_chunks}

    except Exception as e:
        print(f"Error loading index or chunks for testing: {e}")
        return

    # --- Improved Tokenization for Query ---
    tokenized_query = tokenize_text(query)
    print(f"Processed query tokens: {tokenized_query}")
    # --- End Improvement ---

    top_docs = bm25_index.get_top_n(tokenized_query, chunk_ids, n=3)

    print(f"\nQuery: '{query}'")
    print("Top 3 Results:")
    for i, chunk_id in enumerate(top_docs):
        print(f"  Rank {i+1}:")
        print(f"    Chunk ID: {chunk_id}")
        print(f"    Content: {content_map.get(chunk_id, 'Content not found...')[:150]}...")
        print()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_name = "The-toyota-way-second-edition-chapter_1"
    pdf_specific_output_dir = os.path.join(base_dir, "pdf_files", "destination", pdf_name)
    
    chunks_json_file_path = os.path.join(pdf_specific_output_dir, "text_chunks.json")
    bm25_index_file_path = os.path.join(pdf_specific_output_dir, "bm25_index.pkl")

    create_and_save_bm25_index(chunks_json_file_path, bm25_index_file_path)

    test_bm25_query(
        index_path=bm25_index_file_path, 
        query="What is the Toyota Way?",
        chunks_json_path=chunks_json_file_path
    )

