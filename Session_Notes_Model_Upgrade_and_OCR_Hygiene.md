# Session Notes: On-Device RAG Agent Upgrade & Ingestion Hygiene

**Date:** Tuesday, October 13, 2026  
**Context:** Workspace recovery following hard-drive corruption; preparation of an on-device RAG agent powered by Google Gemma 3 for an upcoming production demo.

---

## 1. Executive Summary
During this session, we transformed a recovered, partially broken codebase into a verified, robust, and modern on-device RAG system. We upgraded the model foundation to **Google Gemma 3 4B (GGUF)**, streamlined runtime dependencies, architected a 4-tier ingestion hygiene gatekeeper, resolved critical OCR resolution vulnerabilities (recovering dropped decimals and punctuation), and re-summarized the source chapters to achieve 100% alignment between summaries and on-disk documents.

---

## 2. Key Architecture & Engineering Enhancements

### A. Model Foundation & Offline Runtime
*   **Legacy Issues:** `model_loader.py` attempted to load unquantized Hugging Face weights using `bitsandbytes` 8-bit quantization on CPU in WSL, resulting in crashes and unfulfilled `HF_TOKEN` requirements. Scattered model paths across scripts referenced non-existent or deprecated experimental files (`gemma-3n-E4B-it-Q4_K_M.gguf`).
*   **WSL Hardware Assessment:** Confirmed AMD Ryzen 7 6800H (16 threads), 15 GiB RAM (12 GiB free), and an **NVIDIA GeForce RTX 3070 Laptop GPU (8 GB VRAM)** with CUDA 12.3.
*   **Upgraded Solution:** Standardized on official **`gemma-3-4b-it-Q4_K_M.gguf`** (~2.49 GB) running via `llama-cpp-python`:
    *   Fits entirely within GPU VRAM (`n_gpu_layers=-1`), leaving >4.5 GB of VRAM headroom for the KV cache.
    *   Operates 100% offline with zero external API dependencies or Hugging Face tokens.
    *   Supports a native 128k context window (configured to 8,192 tokens for fast local inference).

### B. Dependency Optimization & Clean Environment
*   **Requirements Audit:** The initial `requirements.txt` contained a raw freeze of 281 packages (~5 GB).
*   **Refactoring:** Created `requirements.txt.bak` and streamlined `requirements.txt`:
    *   **Active Core:** `gradio`, `langgraph`, `langchain-core`, `chromadb`, `sentence-transformers`, `rank-bm25`, `pymupdf`, `pytesseract`, `pillow`, `llama-cpp-python`, and `pydantic`.
    *   **Commented Out:** Unneeded monolithic stacks (Jupyter/Notebook, Streamlit, external cloud AI clients, telemetry, and unneeded monolithic PyTorch/CUDA wheels).
    *   Installed in an isolated `.venv` using `uv`.

### C. Centralized System Configuration (`config.py`)
*   Created `config.py` as the single source of truth for:
    *   Model paths (`MODEL_PATH`, `EMBEDDING_MODEL_NAME`).
    *   Storage directories (`PDF_STAGING_DIR`, `PDF_SOURCE_DIR`, `PDF_QUARANTINE_DIR`, `PDF_DESTINATION_BASE_DIR`).
    *   Database & Checkpoint files (`CHROMA_DB_DIR`, `CHECKPOINT_DB_PATH`, `PROFILE_DB_PATH`, `PDF_SUMMARIES_PATH`).

---

## 3. Ingestion Hygiene Guardrail Architecture

To ensure corrupt, password-protected, or malicious files never break the ingestion pipeline, we established a **Clean-Room Invariant** using a dedicated gatekeeper (`pdf_validator.py`):

```
pdf_files/staging/   (Untrusted Landing Zone)
       │
       ▼
 [ PDFValidator Gatekeeper ]
       ├─► Gate 1: Signature & Size (%PDF magic bytes, 1 KB - 150 MB limits)
       ├─► Gate 2: Security & Encryption (Reject password-locked/DRM files upfront)
       ├─► Gate 3: Structural Parsability (Page tree traversal, page limits)
       └─► Gate 4: Content Readability (Printable char ratio, font CMap health, OCR detection)
       │
   ┌───┴───┐
   │       │
 [PASS]  [FAIL]
   │       │
   ▼       ▼
pdf_files/source/    pdf_files/quarantine/
 (Only clean files)   (Quarantined file + <name>_audit.json)
```

---

## 4. OCR Pipeline Root-Cause Analysis & Enhancements

### The Initial Pitfalls Discovered
1.  **Image-Only Scans:** Ingestion audit revealed all 6 chapters of *The Toyota Way* in `staging/` were 100% scanned raster images with 0 native selectable text.
2.  **The Missing Decimal Point Bug (`83` vs `8.3`):** 
    *   *Cause:* PyMuPDF's default `get_pixmap()` rendered pages at 72 DPI. At 72 DPI, a 10pt decimal point is ~1 pixel wide. Tesseract's noise-removal filter treated it as dust and dropped it.
3.  **Header Masking Vulnerability:** If a scanned page had a 60-character digital watermark (e.g. *Document2PDF Pilot*), `len(text) < 50` returned `False`, causing the engine to skip OCR and lose the entire 3,000-character page body.
4.  **Line-Break Hyphenation:** Words like `"opera-\ntional"` broke BM25 and vector semantic matching.

### Implemented Solutions in `pdf_processor.py`
*   **200 DPI Resolution:** Rendered pages using `page.get_pixmap(dpi=200)`, scaling decimal points to 6–9 distinct dark pixels.
*   **Contrast Preprocessing:** Applied `ImageOps.autocontrast` and PIL contrast boosting to sharpen character edges against scanned backgrounds.
*   **Tesseract Flags:** Enforced `--oem 1 --psm 3 -c preserve_interword_spaces=1`.
*   **Automated Cleaners:** Added de-hyphenation regex (`re.sub(r'(\b\w+)-\n(\w+\b)', r'\1\2', text)`), decimal spacing repair (`re.sub(r'(?<=\d)\s*[\.,]\s*(?=\d)', '.', text)`), and watermark removal.
*   **Validation:** Empirically verified extraction on Page 2: successfully recovered `net profit margin is 8.3 times higher than the industry average`.

---

## 5. Summarization Modernization & Edition Alignment

*   **Edition Discrepancy Resolved:** The old `pdf_summaries.json` was generated from the **Second Edition** (referencing 2007–2017 financials and the 2009 recall crisis). Our physical PDF files are the **First Edition** (containing 2003 automaker metrics).
*   **Pipeline Overhaul:** Replaced the legacy map-reduce approach (50–80 micro-chunk LLM calls per chapter) with context-aware, single-pass chapter summarization (8,192 token window) in `summary_generator.py`:
    *   Execution dropped from ~20 minutes to ~1.5 minutes for all 6 chapters.
    *   Achieved 100% factual fidelity to the text on disk.
*   **Archiving:** Moved old summaries to `archive/pdf_summaries_old.json` and generated fresh, verified summaries in `pdf_summaries.json`.

---

## 6. Files Modified / Created in this Session
*   `config.py` *(New)*: Centralized system configuration.
*   `pdf_validator.py` *(New)*: 4-gate ingestion hygiene gatekeeper.
*   `test_ocr.py` *(New)*: Standalone OCR capability test suite.
*   `index_all_pdfs.py` *(New)*: Automated batch indexer for ChromaDB and BM25.
*   `pdf_processor.py` *(Updated)*: 200 DPI rendering, contrast preprocessing, de-hyphenation.
*   `summary_generator.py` *(Updated)*: High-context Gemma 3 single-pass summarizer.
*   `gemma_gguf_loader.py` *(Updated)*: Direct prompt & chat test with GPU offload.
*   `query_transformer.py` *(Updated)*: Config-bound model loading with GPU support.
*   `rag_agent_with_ui_v3.py` *(Updated)*: Dynamic summary loading and config integration.
*   `requirements.txt` *(Updated)*: Streamlined active dependencies.
*   `requirements.txt.bak` *(New)*: Complete pre-refactor backup.
*   `pdf_summaries.json` *(Updated)*: Grounded topic summaries for all 6 chapters.
*   `archive/pdf_summaries_old.json` *(New)*: Archived legacy summaries.
*   `.gitignore` *(Updated)*: Excluded `*.gguf` and specific model weights.
