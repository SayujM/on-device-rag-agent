from llama_cpp import Llama
import os
import base64
from llama_cpp.llama_chat_format import Llava15ChatHandler # New import

# --- Configuration ---
VLM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "llava-v1.5-7b-Q4_K_M.gguf")
CLIP_MODEL_PATH = os.path.join(os.path.dirname(__file__), "llava-v1.5-7b-mmproj-model-f16.gguf") # New path for CLIP model

# --- VLM Loading ---
def load_vlm_model():
    print(f"Loading VLM model from: {VLM_MODEL_PATH}")
    try:
        # Instantiate the chat handler for LLaVA
        chat_handler = Llava15ChatHandler(clip_model_path=CLIP_MODEL_PATH)

        llm = Llama(
            model_path=VLM_MODEL_PATH,
            n_ctx=4096,  # Increased context window size for image embedding
            n_gpu_layers=0, # Set to 0 for CPU-only, or >0 for GPU offloading
            verbose=False, # Set to True for more detailed loading logs
            chat_handler=chat_handler, # Pass the chat handler
            logits_all=True, # Needed for LLaVA to work correctly
        )
        print("VLM model loaded successfully!")
        return llm
    except Exception as e:
        print(f"Error loading VLM model: {e}")
        return None

# --- Image Description Generation ---
def generate_image_description(llm: Llama, image_path: str, prompt: str = "Analyze this image. Describe all visual elements, including any graphs, diagrams, or text. If there is text, please describe its content or purpose. If there is a graph, describe its title, axes, and the data presented.") -> str:
    """
    Generates a textual description of an image using the loaded LLaVA VLM.

    Args:
        llm (Llama): The loaded LLaVA model instance.
        image_path (str): The path to the image file.
        prompt (str): The prompt to give to the VLM for image description.

    Returns:
        str: The generated textual description of the image.
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"

    # LLaVA expects image in base64 format within the messages
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt}
            ],
        }
    ]

    print(f"Generating description for {image_path}...")
    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=200,  # Max tokens for the description
            temperature=0.2, # Controls randomness - lowered for more deterministic output
        )
        generated_text = response["choices"][0]["message"]["content"]
        return generated_text
    except Exception as e:
        return f"Error generating description: {e}"

if __name__ == "__main__":
    # Example Usage:
    vlm_model = load_vlm_model()

    if vlm_model:
        # Use a sample image from pdf_processor.py output
        # Make sure to run pdf_processor.py first to generate these images
        sample_image_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pdf_files",
            "destination",
            "The-toyota-way-second-edition-chapter_1", # Subfolder for the PDF
            "extracted_images",
            "page1.png" # Example: first page rendered image
        )

        if os.path.exists(sample_image_path):
            description = generate_image_description(vlm_model, sample_image_path)
            print(f"\n--- Generated Description for {os.path.basename(sample_image_path)} ---")
            print(description)
        else:
            print(f"Sample image not found: {sample_image_path}")
            print("Please ensure you have run pdf_processor.py to generate images.")
    else:
        print("VLM model could not be loaded. Cannot generate description.")