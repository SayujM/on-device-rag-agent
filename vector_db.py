import chromadb
import os
import json

# --- Configuration ---
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "rag_chunks"

# --- ChromaDB Initialization ---
def get_chroma_client(db_dir: str):
    """
    Initializes and returns a persistent ChromaDB client.

    Args:
        db_dir (str): The directory where ChromaDB will store its data.

    Returns:
        chromadb.PersistentClient: An initialized ChromaDB client.
    """
    print(f"Initializing ChromaDB client at: {db_dir}")
    return chromadb.PersistentClient(path=db_dir)

# --- Data Loading and Storage ---
def populate_vector_db(client: chromadb.PersistentClient, embeddings_data_path: str, collection_name: str):
    """
    Loads embeddings and metadata from a JSON file and populates a ChromaDB collection.

    Args:
        client (chromadb.PersistentClient): The ChromaDB client instance.
        embeddings_data_path (str): Path to the JSON file containing embeddings and metadata.
        collection_name (str): The name of the ChromaDB collection to use.
    """
    print(f"Loading embeddings data from: {embeddings_data_path}")
    try:
        with open(embeddings_data_path, "r", encoding="utf-8") as f:
            embeddings_with_metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: Embeddings data file not found at {embeddings_data_path}")
        print("Please ensure embedding_model.py has been run.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {embeddings_data_path}: {e}")
        return

    if not embeddings_with_metadata:
        print("No embeddings found in the file. Skipping database population.")
        return

    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    metadatas = []
    documents = [] # This is the text content associated with the embedding

    for item in embeddings_with_metadata:
        ids.append(item["metadata"]["chunk_id"])
        embeddings.append(item["embedding"])
        metadatas.append(item["metadata"])
        # Assuming the original text content is available in metadata or can be derived
        # For now, we'll use a placeholder or assume it's not directly needed for add() if we have embeddings
        # However, for query results, having the original text is crucial.
        # Let's assume the original content is stored in the metadata for now, or we need to fetch it from text_chunks.json
        # For simplicity, let's add the content from the metadata if available, or a placeholder.
        # A better approach would be to load text_chunks.json here and map by chunk_id.
        # For now, let's assume the content is part of the metadata dict from text_chunker.py's output.
        # Re-reading text_chunks.json is more robust to get the actual content.
        # Let's modify this to read text_chunks.json to get the content.
        documents.append(item["metadata"].get("content", "")) # Placeholder if not directly in metadata

    # --- Refinement: Load original content from text_chunks.json for documents field ---
    # This is more robust than relying on content being in metadata from embedding_model.py
    original_chunks_path = os.path.join(os.path.dirname(embeddings_data_path), "text_chunks.json")
    original_chunks_map = {}
    try:
        with open(original_chunks_path, "r", encoding="utf-8") as f:
            all_original_chunks = json.load(f)
            for chunk in all_original_chunks:
                original_chunks_map[chunk["metadata"]["chunk_id"]] = chunk["content"]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load original chunk content from {original_chunks_path}: {e}. Documents field might be empty.")

    # Update documents list with actual content
    documents = [original_chunks_map.get(chunk_id, "") for chunk_id in ids]
    # --- End Refinement ---

    print(f"Adding {len(ids)} embeddings to collection '{collection_name}'...")
    collection = client.get_or_create_collection(name=collection_name)

    # ChromaDB add method requires lists of ids, embeddings, metadatas, and documents
    try:
        collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=ids
        )
        print(f"Successfully added {len(ids)} embeddings to ChromaDB collection '{collection_name}'.")
    except Exception as e:
        print(f"Error adding embeddings to ChromaDB: {e}")

# --- Query Function (for testing) ---
def query_vector_db(client: chromadb.PersistentClient, collection_name: str, query_text: str, embedding_model_instance, n_results: int = 5):
    """
    Performs a similarity search in the ChromaDB collection.

    Args:
        client (chromadb.PersistentClient): The ChromaDB client instance.
        collection_name (str): The name of the ChromaDB collection.
        query_text (str): The text query to search for.
        embedding_model_instance: The loaded SentenceTransformer model to embed the query.
        n_results (int): The number of top results to retrieve.

    Returns:
        dict: The query results from ChromaDB.
    """
    print(f"Querying collection '{collection_name}' with text: '{query_text}'")
    query_embedding = embedding_model_instance.encode([query_text]).tolist()[0]

    collection = client.get_collection(name=collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    return results

if __name__ == "__main__":
    # Example Usage:
    # 1. Get ChromaDB client
    chroma_client = get_chroma_client(CHROMA_DB_DIR)

    # 2. Define path to embeddings data (output from embedding_model.py)
    embeddings_data_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdf_files",
        "destination",
        "The-toyota-way-second-edition-chapter_1",
        "text_embeddings.json"
    )

    # 3. Populate the database
    populate_vector_db(chroma_client, embeddings_data_file_path, COLLECTION_NAME)

    # 4. Test query (requires loading the embedding model again for the query text)
    print("\n--- Testing Query Functionality ---")
    # Import and load the embedding model here for query embedding
    from embedding_model import load_embedding_model # Import the function
    query_embedding_model = load_embedding_model()

    if query_embedding_model:
        test_query = "What is the Toyota Way model?"
        results = query_vector_db(chroma_client, COLLECTION_NAME, test_query, query_embedding_model)

        print("\nQuery Results:")
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                print(f"Result {i+1}:")
                print(f"  Distance: {results["distances"][0][i]:.4f}")
                print(f"  Chunk ID: {results["metadatas"][0][i]["chunk_id"]}")
                print(f"  Source: {os.path.basename(results["metadatas"][0][i]["source_file"])}")
                print(f"  Page: {results["metadatas"][0][i]["page_number"]}")
                print(f"  Content: {results["documents"][0][i][:200]}...") # Print first 200 chars of content
                print("\n")
        else:
            print("No results found.")
    else:
        print("Could not load embedding model for querying.")