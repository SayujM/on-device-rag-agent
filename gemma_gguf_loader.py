from llama_cpp import Llama
import os
from config import MODEL_PATH

# --- Configuration ---
# Path to the downloaded GGUF model file is imported from config.py

# --- Model Loading ---
print(f"Loading GGUF model from: {MODEL_PATH}")

# Initialize the Llama model
# n_gpu_layers: -1 offloads all layers to GPU if CUDA is available, or falls back to CPU.
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,  # Context window size
    n_gpu_layers=-1, # -1 offloads to GPU when available
    verbose=False, # Set to True for more detailed loading logs
)
print("GGUF model loaded successfully!")

# --- Text Generation Test 1: Chat Completion ---
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

print("Generating response via chat completion...")
response = llm.create_chat_completion(
    messages=messages,
    max_tokens=50,
    temperature=0.7,
)
generated_text = response["choices"][0]["message"]["content"]
print(f"--- Chat Completion Output ---\n{generated_text}\n")

# --- Text Generation Test 2: Raw Prompt Invocation (used across RAG agent nodes) ---
prompt = "<start_of_turn>user\nWhat is the capital of Japan? Reply in one sentence.<end_of_turn>\n<start_of_turn>model\n"
print("Generating response via raw prompt invocation...")
raw_response = llm(prompt=prompt, max_tokens=50, stop=["<end_of_turn>"])
raw_text = raw_response["choices"][0]["text"].strip()
print(f"--- Raw Prompt Output ---\n{raw_text}\n")
print("All model tests passed successfully!")
