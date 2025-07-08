import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
import streamlit as st
import os


# Load environment variables for HF_TOKEN
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

@st.cache_resource
def load_gemma_model(model_id: str):
    """The 4-bit quantization failed - shifting to 8-bit"""
    # Define quantization configuration for 4-bit
    # This is the most aggressive and memory-efficient option
    # nf4_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4", # NormalFloat 4-bit
    #     bnb_4bit_use_double_quant=True, # Double quantization for extra memory saving
    #     bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for computation if supported by CPU, otherwise torch.float16 or torch.float32
    # )

    # Define quantization configuration for 8-bit
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True, # Use 8-bit quantization
        # No bnb_4bit_quant_type or bnb_4bit_use_double_quant here, as they are for 4-bit
        # No bnb_4bit_compute_dtype needed for 8-bit, it uses float32 for compute by default.
    )

    print(f"Loading Gemma model: {model_id} with 4-bit quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    # For CPU-only, ensure torch.device('cpu') is used.
    # `accelerate` will handle device placement.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cpu", # Explicitly set to CPU for resource-constrained laptops without strong GPUs
        token=HF_TOKEN,
    )
    print("Gemma model loaded successfully!")
    return tokenizer, model


if __name__ == "__main__":
    # Test model loading
    """gemma-3n-E4B-it couldn't be loaded on my WSL"""
    # MODEL_ID = "google/gemma-3n-E4B-it" # Or "google/gemma-3n-E2B-it"
    MODEL_ID = "google/gemma-2b-it" 
    tokenizer, model = load_gemma_model(MODEL_ID)
    # --- Basic Text Generation Test ---
    # We must use the tokenizer's chat template to format the prompt correctly.
    chat = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    print(f"Test prompt: {prompt}")

    # Tokenize the input text and move it to the same device as the model (CPU).
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate a response from the model, explicitly providing the end-of-sequence token ID.
    outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1, eos_token_id=model.config.eos_token_id)

    # Decode the generated token IDs back into a string and print the raw output.
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Raw generated output:\n---\n{decoded_output}\n---")