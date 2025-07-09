from sentence_transformers import SentenceTransformer
import os
import json # New import

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Embedding Model Loading ---
def load_embedding_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None

# --- Embedding Generation ---
def generate_embeddings(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """
    Generates dense vector embeddings for a list of text strings.

    Args:
        model (SentenceTransformer): The loaded SentenceTransformer model instance.
        texts (list[str]): A list of text strings for which to generate embeddings.

    Returns:
        list[list[float]]: A list of lists, where each inner list is the embedding vector for a text string.
    """
    if not texts:
        return []

    print(f"Generating embeddings for {len(texts)} texts...")
    try:
        # SentenceTransformer automatically handles batching for efficiency
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        print("Embeddings generated successfully!")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

if __name__ == "__main__":
    embedding_model = load_embedding_model()

    if embedding_model:
        # --- New: Read chunks from text_chunks.json ---
        chunks_json_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pdf_files",
            "destination",
            "The-toyota-way-second-edition-chapter_1",
            "text_chunks.json"
        )

        all_chunks_with_metadata = []
        chunk_contents_to_embed = []

        try:
            with open(chunks_json_file_path, "r", encoding="utf-8") as f:
                all_chunks_with_metadata = json.load(f)

            for chunk_data in all_chunks_with_metadata:
                chunk_contents_to_embed.append(chunk_data["content"])

        except FileNotFoundError:
            print(f"Error: Chunk JSON file not found at {chunks_json_file_path}")
            print("Please ensure text_chunker.py has been run for 'The-toyota-way-second-edition-chapter_1.pdf'.")
            exit()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {chunks_json_file_path}: {e}")
            exit()
        # --- End New ---

        embeddings_list = generate_embeddings(embedding_model, chunk_contents_to_embed)

        if embeddings_list:
            # Associate embeddings with their original metadata
            final_embeddings_with_metadata = []
            for i, embedding in enumerate(embeddings_list):
                chunk_metadata = all_chunks_with_metadata[i]["metadata"]
                final_embeddings_with_metadata.append({
                    "embedding": embedding,
                    "metadata": chunk_metadata
                })

            # --- New: Save final_embeddings_with_metadata to a JSON file ---
            # Determine the PDF-specific output directory (same as text_chunks.json)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            pdf_destination_base_dir = os.path.join(base_dir, "pdf_files", "destination")
            # Extract PDF name from the chunks_json_file_path
            pdf_output_subfolder_name = os.path.basename(os.path.dirname(chunks_json_file_path))
            pdf_specific_output_dir = os.path.join(pdf_destination_base_dir, pdf_output_subfolder_name)

            output_embeddings_file_path = os.path.join(pdf_specific_output_dir, "text_embeddings.json")

            try:
                with open(output_embeddings_file_path, "w", encoding="utf-8") as f_out:
                    json.dump(final_embeddings_with_metadata, f_out, indent=4)
                print(f"\nEmbeddings with metadata saved to: {output_embeddings_file_path}")
            except Exception as e:
                print(f"Error saving embeddings to file: {e}")
            # --- End New ---

            print(f"\nGenerated {len(final_embeddings_with_metadata)} embeddings.")
            print(f"Dimension of first embedding: {len(final_embeddings_with_metadata[0]["embedding"])}")
            
            # Print first 5 values of embeddings for a few chunks, including chunk_id
            num_samples = min(3, len(final_embeddings_with_metadata)) # Print up to 3 samples
            sample_indices = [0, len(final_embeddings_with_metadata) // 2, len(final_embeddings_with_metadata) - 1] # First, middle, last

            print("\n--- Sample Embeddings (with Metadata) ---")
            for i, idx in enumerate(sample_indices):
                if idx < len(final_embeddings_with_metadata):
                    sample_chunk = final_embeddings_with_metadata[idx]
                    print(f"Chunk ID: {sample_chunk["metadata"]["chunk_id"]}")
                    print(f"Embedding (first 5 values): {sample_chunk["embedding"][:5]}...")
                if i == num_samples - 1: # Break after last sample
                    break
        else:
            print("No embeddings generated.")
    else:
        print("Embedding model could not be loaded. Cannot generate embeddings.")