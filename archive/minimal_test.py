import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

# --- Minimal Setup ---
try:
    HF_TOKEN = os.environ["HF_TOKEN"]
except KeyError:
    print("Error: HF_TOKEN environment variable not set.")
    exit(1)

MODEL_ID = "google/gemma-2b-it"

# --- Model Loading ---
print(f"Loading model: {MODEL_ID}")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cpu",
    token=HF_TOKEN,
)
print("Model loaded successfully.")

# --- Text Generation ---
chat = [
    {"role": "user", "content": "What is the capital of France?"},
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

print(f"Using prompt:\n---\n{prompt}\n---")

inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

print("Generating text...")
outputs = model.generate(**inputs, max_new_tokens=50, num_beams=1, do_sample=False)

# --- Output ---
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\n--- Generated Output ---\n{decoded_output}")

# Clean parsing of the final answer
try:
    final_answer = decoded_output.split("model\n")[1].strip()
except IndexError:
    final_answer = "Could not parse the answer from the model's output."

print(f"\n--- Final Answer ---\n{final_answer}")
