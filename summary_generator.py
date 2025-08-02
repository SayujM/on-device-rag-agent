
import os
import json
from typing import List, Dict

# Local module imports
from pdf_processor import process_pdf
from text_chunker import chunk_text
from query_transformer import QueryTransformer

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_SOURCE_DIR = os.path.join(BASE_DIR, "pdf_files", "source")
PDF_SUMMARIES_PATH = os.path.join(BASE_DIR, "pdf_summaries.json")
GGUF_MODEL_PATH = os.path.join(BASE_DIR, "gemma-3n-E4B-it-Q4_K_M.gguf")

# --- Prompts ---
CHUNK_SUMMARIZATION_PROMPT = """<start_of_turn>user
[YOUR TASK]
Analyze the text chunk below. Your goal is to write a concise, well-written summary of its content in a single paragraph.

[RULES]
- The summary MUST be a single, short paragraph.
- The summary MUST accurately reflect the main points of the text.
- **You MUST ONLY use information from the provided [TEXT CHUNK]. Do NOT use any prior knowledge or information from outside the text.**
- Do NOT use markdown, bullet points, or any other special formatting.
- Do NOT add any conversational text, introduction, or explanation. Your response should be the summary itself.

[TEXT CHUNK]
---
{chunk_text}
---

[CONCISE SUMMARY PARAGRAPH]
<end_of_turn>
<start_of_turn>model
"""

FINAL_COMBINATION_PROMPT = """<start_of_turn>user
[YOUR TASK]
You are an expert editor. You will be given a collection of summary paragraphs from different parts of a single document. Your single goal is to synthesize these into one final, cohesive, and well-written summary.

[RULES]
- The final summary MUST be a single, comprehensive paragraph.
- It MUST accurately integrate the key information from the provided paragraphs.
- **You MUST ONLY use information from the provided [PROVIDED SUMMARY PARAGRAPHS]. Do NOT use any prior knowledge or information from outside the text.**
- Do NOT use markdown, bullet points, or any other special formatting.
- Do NOT add any conversational text, introduction, or explanation. Your response should be the final summary itself.

[PROVIDED SUMMARY PARAGRAPHS]
---
{combined_summaries}
---

[FINAL COHESIVE SUMMARY]
<end_of_turn>
<start_of_turn>model
"""

def load_llm():
    """Loads the GGUF model using the QueryTransformer class."""
    print("Loading the LLM...")
    try:
        # We only need the llm instance from the transformer
        query_transformer = QueryTransformer(model_path=GGUF_MODEL_PATH)
        print("LLM loaded successfully.")
        return query_transformer.llm
    except Exception as e:
        print(f"Fatal Error: Could not load the LLM. {e}")
        raise

def load_existing_summaries() -> Dict[str, str]:
    """Loads existing summaries from the JSON file if it exists."""
    if os.path.exists(PDF_SUMMARIES_PATH):
        try:
            with open(PDF_SUMMARIES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse existing summaries file. Starting fresh. Error: {e}")
            return {}
    return {}

def save_summaries(summaries: Dict[str, str]):
    """Saves the summaries dictionary to the JSON file."""
    try:
        with open(PDF_SUMMARIES_PATH, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=4)
    except IOError as e:
        print(f"Error: Could not save summaries to {PDF_SUMMARIES_PATH}. Error: {e}")

def generate_summary_for_pdf(pdf_path: str, llm) -> str:
    """
    Generates a summary for a single PDF using a hierarchical Map-Reduce strategy
    with token-aware dynamic batching to prevent context window overflows.
    """
    pdf_filename = os.path.basename(pdf_path)
    pdf_name_without_ext = os.path.splitext(pdf_filename)[0]
    print(f"--- Starting summary generation for: {pdf_filename} ---")

    # 1. Extract and Chunk Text
    print("Step 1: Extracting and chunking text...")
    try:
        # Create a unique subdirectory for this PDF's temporary images to prevent conflicts
        unique_dummy_dir = os.path.join(BASE_DIR, "temp_images", pdf_name_without_ext)
        os.makedirs(unique_dummy_dir, exist_ok=True)

        extracted_texts, _ = process_pdf(pdf_path, unique_dummy_dir)
        if not extracted_texts:
            return "Error: No text could be extracted from the PDF."
        chunks_with_metadata = chunk_text(extracted_texts, pdf_path)
        text_chunks = [chunk['content'] for chunk in chunks_with_metadata]
        print(f"Extracted {len(text_chunks)} chunks.")
    except Exception as e:
        return f"Error during text extraction/chunking: {e}"

    # 2. Map Phase: Summarize each chunk into a dense topic list
    print(f"Step 2: Summarizing {len(text_chunks)} chunks into topic lists (Map Phase)...")
    chunk_topics = []
    for i, chunk in enumerate(text_chunks):
        print(f"  - Processing chunk {i+1}/{len(text_chunks)}")
        try:
            prompt = CHUNK_SUMMARIZATION_PROMPT.format(chunk_text=chunk)
            response = llm(prompt=prompt, max_tokens=256, stop=["<end_of_turn>"])
            topics = response["choices"][0]["text"].strip()
            if topics: chunk_topics.append(topics)
        except Exception as e:
            print(f"    - Warning: Could not process chunk {i+1}. Error: {e}")

    # 3. Hierarchical Reduce with Token-Aware Dynamic Batching
    print(f"Step 3: Combining {len(chunk_topics)} topic lists (Hierarchical Reduce)...")
    summaries_to_reduce = chunk_topics
    level = 1
    CONTEXT_WINDOW_LIMIT = 2048
    PROMPT_OVERHEAD = 250  # Estimated tokens for the prompt template
    MAX_TOKENS_FOR_RESPONSE = 512

    while len(summaries_to_reduce) > 1:
        print(f"  - Reduce Level {level}: Combining {len(summaries_to_reduce)} summaries...")
        next_level_summaries = []
        current_batch = []
        current_batch_tokens = 0

        for summary in summaries_to_reduce:
            # Simple token estimation (words * 1.5)
            summary_tokens = len(summary.split()) * 1.5 

            if current_batch_tokens + summary_tokens + PROMPT_OVERHEAD > CONTEXT_WINDOW_LIMIT - MAX_TOKENS_FOR_RESPONSE:
                # Process the current batch if it's not empty
                if current_batch:
                    print(f"    - Processing batch of {len(current_batch)} summaries ({int(current_batch_tokens)} tokens)... ")
                    try:
                        combined_text = "\n".join(current_batch)
                        prompt = FINAL_COMBINATION_PROMPT.format(combined_summaries=combined_text)
                        response = llm(prompt=prompt, max_tokens=MAX_TOKENS_FOR_RESPONSE, stop=["<end_of_turn>"])
                        new_summary = response["choices"][0]["text"].strip()
                        if new_summary: next_level_summaries.append(new_summary)
                    except Exception as e:
                        print(f"      - Warning: Could not combine a batch. Error: {e}")
                # Reset for the next batch
                current_batch = [summary]
                current_batch_tokens = summary_tokens
            else:
                current_batch.append(summary)
                current_batch_tokens += summary_tokens
        
        # Process the last remaining batch
        if current_batch:
            print(f"    - Processing final batch of {len(current_batch)} summaries ({int(current_batch_tokens)} tokens)... ")
            try:
                combined_text = "\n".join(current_batch)
                prompt = FINAL_COMBINATION_PROMPT.format(combined_summaries=combined_text)
                response = llm(prompt=prompt, max_tokens=MAX_TOKENS_FOR_RESPONSE, stop=["<end_of_turn>"])
                new_summary = response["choices"][0]["text"].strip()
                if new_summary: next_level_summaries.append(new_summary)
            except Exception as e:
                print(f"      - Warning: Could not combine final batch. Error: {e}")

        summaries_to_reduce = next_level_summaries
        level += 1

    if len(summaries_to_reduce) == 1:
        final_summary = summaries_to_reduce[0]
        # Final formatting is handled here in the code, not by the LLM
        final_summary = "Summary of the PDF selected: " + final_summary
        print("--- Summary generation complete. ---")
        return final_summary
    else:
        return "Error: Hierarchical reduction failed to produce a single summary."



def main():
    """
    Main function to run the summary generation process for all new PDFs.
    """
    print("--- Starting PDF Summary Generator ---")
    llm = load_llm()
    
    summaries = load_existing_summaries()
    print(f"Loaded {len(summaries)} existing summaries.")
    
    if not os.path.exists(PDF_SOURCE_DIR):
        print(f"Error: PDF source directory not found at {PDF_SOURCE_DIR}")
        return

    all_pdfs = [f for f in os.listdir(PDF_SOURCE_DIR) if f.lower().endswith(".pdf")]
    new_pdfs = [f for f in all_pdfs if f not in summaries]

    if not new_pdfs:
        print("All PDFs are already summarized. No new files to process.")
        return

    print(f"Found {len(new_pdfs)} new PDF(s) to process: {', '.join(new_pdfs)}")

    for pdf_filename in new_pdfs:
        pdf_path = os.path.join(PDF_SOURCE_DIR, pdf_filename)
        summary = generate_summary_for_pdf(pdf_path, llm)
        
        if "Error:" not in summary:
            print(f"Successfully generated summary for {pdf_filename}.")
            summaries[pdf_filename] = summary
            # Save after each successful generation to prevent data loss
            save_summaries(summaries)
            print(f"Saved updated summaries to {PDF_SUMMARIES_PATH}")
        else:
            print(f"Failed to generate summary for {pdf_filename}. Reason: {summary}")

    print("--- PDF Summary Generation Process Finished ---")

if __name__ == "__main__":
    main()
