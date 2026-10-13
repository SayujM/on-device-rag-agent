import os
import sys
import json
import shutil
import argparse
from typing import List, Dict

# Local module imports
from pdf_processor import process_pdf
from query_transformer import QueryTransformer
from config import BASE_DIR, PDF_SOURCE_DIR, PDF_SUMMARIES_PATH, MODEL_PATH

# --- High-Context Chapter Summarization Prompt ---
CHAPTER_SUMMARIZATION_PROMPT = """<start_of_turn>user
[YOUR TASK]
You are an expert technical editor analyzing a chapter from the book "The Toyota Way".
Read the complete OCR-extracted text of the chapter below.

Synthesize a comprehensive topic summary for semantic retrieval routing.
Your summary MUST accurately capture:
1. The exact Chapter Title and Core Principle(s) discussed.
2. Key historical background, operational principles (e.g. TPS, Jidoka, One-piece flow, Heijunka, Genchi Genbutsu), metrics, and case studies mentioned in this specific text.

[RULES]
- Start directly with: "Topics included: "
- Write a single, dense, cohesive paragraph (between 120 and 220 words).
- Ground your summary STRICTLY in the provided text. Do NOT add external assumptions.
- Do NOT use bullet points, bold formatting, markdown headers, or conversational introductions.

[CHAPTER TEXT]
{chapter_text}

[SUMMARY]
<end_of_turn>
<start_of_turn>model
"""

COMBINATION_PROMPT = """<start_of_turn>user
[YOUR TASK]
Synthesize the following section summaries of a single chapter into one unified, cohesive paragraph summary.
Start directly with: "Topics included: "
Do NOT use bullet points or conversational text.

[SECTION SUMMARIES]
{section_summaries}

[FINAL UNIFIED SUMMARY]
<end_of_turn>
<start_of_turn>model
"""


def load_llm():
    """Loads the GGUF model with high context for chapter-level analysis."""
    print("Loading LLM (Gemma 3 4B) with 8192 context window...")
    try:
        query_transformer = QueryTransformer(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=8192
        )
        print("LLM loaded successfully for summarization.")
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
            print(f"Warning: Could not read existing summaries file. Starting fresh. Error: {e}")
            return {}
    return {}


def save_summaries(summaries: Dict[str, str]):
    """Saves the summaries dictionary to the JSON file."""
    try:
        with open(PDF_SUMMARIES_PATH, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=4)
        print(f"Saved updated summaries to {PDF_SUMMARIES_PATH}")
    except IOError as e:
        print(f"Error: Could not save summaries to {PDF_SUMMARIES_PATH}. Error: {e}")


def generate_summary_for_pdf(pdf_path: str, llm) -> str:
    """
    Extracts text page-by-page via OCR and generates a chapter-level topic summary.
    """
    pdf_filename = os.path.basename(pdf_path)
    pdf_name_without_ext = os.path.splitext(pdf_filename)[0]
    print(f"\n" + "=" * 60)
    print(f"Processing: {pdf_filename}")
    print("=" * 60)

    # 1. OCR Text Extraction
    temp_dir = os.path.join(BASE_DIR, "temp_images", pdf_name_without_ext)
    os.makedirs(temp_dir, exist_ok=True)
    try:
        print("Step 1: Extracting text page-by-page (with OCR fallback)...")
        extracted_texts, _ = process_pdf(pdf_path, temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    non_empty_pages = [
        f"--- Page {i+1} ---\n{text.strip()}"
        for i, text in enumerate(extracted_texts)
        if text.strip()
    ]

    if not non_empty_pages:
        return "Error: No text could be extracted from the PDF."

    total_extracted_chars = sum(len(p) for p in non_empty_pages)
    print(f"Extracted {len(non_empty_pages)} non-empty pages ({total_extracted_chars} characters).")

    # 2. Context Assessment & Summarization
    # ~4 characters per token estimate. 8192 tokens ~ 30,000 characters headroom.
    MAX_DIRECT_CHARS = 24000

    if total_extracted_chars <= MAX_DIRECT_CHARS:
        full_text = "\n\n".join(non_empty_pages)
        print("Step 2: Generating full chapter summary in a single context-aware pass...")
        prompt = CHAPTER_SUMMARIZATION_PROMPT.format(chapter_text=full_text)
        try:
            response = llm(prompt=prompt, max_tokens=400, temperature=0.3, stop=["<end_of_turn>"])
            summary = response["choices"][0]["text"].strip()
        except Exception as e:
            return f"Error during generation: {e}"
    else:
        # Segment large chapters into 2 halves
        print("Step 2: Chapter is large; summarizing in two balanced sections...")
        midpoint = len(non_empty_pages) // 2
        part1 = "\n\n".join(non_empty_pages[:midpoint])
        part2 = "\n\n".join(non_empty_pages[midpoint:])

        prompt1 = CHAPTER_SUMMARIZATION_PROMPT.format(chapter_text=part1)
        resp1 = llm(prompt=prompt1, max_tokens=300, temperature=0.3, stop=["<end_of_turn>"])
        summary1 = resp1["choices"][0]["text"].strip()

        prompt2 = CHAPTER_SUMMARIZATION_PROMPT.format(chapter_text=part2)
        resp2 = llm(prompt=prompt2, max_tokens=300, temperature=0.3, stop=["<end_of_turn>"])
        summary2 = resp2["choices"][0]["text"].strip()

        print("Step 3: Combining section summaries into final chapter summary...")
        combine_prompt = COMBINATION_PROMPT.format(section_summaries=f"Part 1:\n{summary1}\n\nPart 2:\n{summary2}")
        final_resp = llm(prompt=combine_prompt, max_tokens=400, temperature=0.3, stop=["<end_of_turn>"])
        summary = final_resp["choices"][0]["text"].strip()

    if not summary.startswith("Topics included:"):
        summary = "Topics included: " + summary

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate accurate PDF topic summaries using Gemma 3 4B.")
    parser.add_argument("--file", type=str, default=None, help="Specific PDF filename to process (e.g. The-toyota-way-second-edition-01_chapter_1.pdf)")
    parser.add_argument("--force", action="store_true", help="Force re-generation of summaries for all PDFs in source directory.")
    args = parser.parse_args()

    print("=" * 60)
    print("PDF SUMMARY GENERATOR (High-Context Engine)")
    print("=" * 60)

    llm = load_llm()
    summaries = load_existing_summaries()

    if not os.path.exists(PDF_SOURCE_DIR):
        print(f"Error: Source directory not found: {PDF_SOURCE_DIR}")
        return

    if args.file:
        target_pdfs = [args.file]
    elif args.force:
        target_pdfs = sorted([f for f in os.listdir(PDF_SOURCE_DIR) if f.lower().endswith(".pdf")])
    else:
        all_pdfs = sorted([f for f in os.listdir(PDF_SOURCE_DIR) if f.lower().endswith(".pdf")])
        target_pdfs = [f for f in all_pdfs if f not in summaries]

    if not target_pdfs:
        print("No target PDFs to process. Use --force to re-generate all, or --file <name> for a specific file.")
        return

    print(f"Target PDFs ({len(target_pdfs)}): {', '.join(target_pdfs)}")

    for pdf_filename in target_pdfs:
        pdf_path = os.path.join(PDF_SOURCE_DIR, pdf_filename)
        if not os.path.exists(pdf_path):
            print(f"Error: File not found: {pdf_path}")
            continue

        summary = generate_summary_for_pdf(pdf_path, llm)
        if not summary.startswith("Error"):
            print(f"\n[Generated Summary for {pdf_filename}]:\n{summary}\n")
            summaries[pdf_filename] = summary
            save_summaries(summaries)
        else:
            print(f"[FAILED] {pdf_filename}: {summary}")

    print("\nSummary generation finished.")


if __name__ == "__main__":
    main()
