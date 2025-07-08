from llama_cpp import Llama
import os

# --- Configuration ---
# Path to the downloaded GGUF model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "gemma-3n-E4B-it-Q4_K_M.gguf")

# --- Model Loading ---
print(f"Loading GGUF model from: {MODEL_PATH}")

# Initialize the Llama model
# n_gpu_layers: Set to a positive value to offload some layers to the GPU if available.
#               -1 will offload all layers. 0 will use only CPU.
#               Adjust based on your GPU VRAM (e.g., 20-30 for a modest integrated GPU).
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,  # Context window size
    n_gpu_layers=0, # Set to 0 for CPU-only, or >0 for GPU offloading
    verbose=False, # Set to True for more detailed loading logs
)
print("GGUF model loaded successfully!")

# --- Text Generation ---
# Define the chat messages in the format expected by Llama.cpp
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

print("Generating response...")

# Create a chat completion
response = llm.create_chat_completion(
    messages=messages,
    max_tokens=50,  # Maximum number of tokens to generate
    temperature=0.7, # Controls randomness (0.0-1.0)
)

# Extract and print the generated content
generated_text = response["choices"][0]["message"]["content"]
print(f"\n--- Generated Text ---\n{generated_text}")
