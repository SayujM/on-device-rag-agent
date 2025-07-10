

from llama_cpp import Llama
import os
import re

class QueryTransformer:
    """
    A class to handle the transformation of a user query into multiple,
    semantically different queries using a GGUF language model.
    """
    def __init__(self, model_path: str, n_gpu_layers: int = 0, n_ctx: int = 2048):
        """
        Initializes the QueryTransformer and loads the GGUF model.

        Args:
            model_path (str): The absolute path to the GGUF model file.
            n_gpu_layers (int): Number of layers to offload to GPU. 0 for CPU-only.
            n_ctx (int): The context window size for the model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        print(f"Loading GGUF model from: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print("GGUF model loaded successfully for query transformation!")

    def transform_query(self, user_query: str, num_variations: int = 3) -> list[str]:
        """
        Rephrases the user's query into multiple semantically different, affirmative
        forms suitable for retrieval.

        Args:
            user_query (str): The original query from the user.
            num_variations (int): The desired number of alternative queries.

        Returns:
            list[str]: A list containing the generated query variations.
                       Returns an empty list if generation fails.
        """
        # Using the official Gemma instruction-following format
        prompt = f"<start_of_turn>user\nGenerate {num_variations} diverse and semantically distinct search queries based on this question: '{user_query}' Each query should be on a new line, without numbers or bullets.<end_of_turn>\n<start_of_turn>model\n"

        print(f"\nTransforming query: '{user_query}'")
        try:
            response = self.llm(
                prompt=prompt,
                max_tokens=200,  # Allow ample space for the queries
                temperature=0.8, # Higher temperature for more creative/varied outputs
                stop=["<end_of_turn>"], # Stop token for Gemma
            )
            
            generated_text = response["choices"][0]["text"].strip()
            
            # --- Parsing the output ---
            # Split the text by newlines
            queries = generated_text.split('\n')
            # Clean up any leading/trailing whitespace or list-like markers
            cleaned_queries = [re.sub(r"^[\*\- \d\.]+", "", q).strip() for q in queries]
            # Filter out any empty strings that might result from the parsing
            final_queries = [q for q in cleaned_queries if q]
            
            print("Query transformation successful.")
            return final_queries

        except Exception as e:
            print(f"Error during query transformation: {e}")
            return []

if __name__ == "__main__":
    # --- Configuration for Standalone Test ---
    GGUF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "gemma-3n-E4B-it-Q4_K_M.gguf")

    try:
        # 1. Initialize the transformer
        query_transformer = QueryTransformer(model_path=GGUF_MODEL_PATH)

        # 2. Define a test query
        # test_query = "What are the core principles of the Toyota Production System?"
        test_query = "What is the Toyota Way?"

        # 3. Transform the query
        transformed_queries = query_transformer.transform_query(test_query, num_variations=4)

        # 4. Print the results
        if transformed_queries:
            print("\n--- Original Query ---")
            print(test_query)
            print("\n--- Transformed Queries ---")
            # Prepend the original query to the list for retrieval
            all_queries = [test_query] + transformed_queries
            for i, query in enumerate(all_queries):
                print(f"{i+1}. {query}")
        else:
            print("\nNo transformed queries were generated.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the GGUF model file 'gemma-3n-E4B-it-Q4_K_M.gguf' is in the project root directory.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
