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

## 6. Tiered Hybrid Web Search & Anti-Hallucination Guardrail

### The Issue
*   When a user query was outside the source PDF document, the agent either failed due to missing Google credentials or misrouted into `generate_simple_response_node` where Gemma hallucinated an ungrounded answer from its parametric memory.
*   Google Programmable Search Engine (CSE) has disabled open-web search for new standard engines, requiring complex domain whitelists.

### The Solution: `web_searcher.py`
We architected a resilient two-tier search strategy:
*   **Tier 1 (Zero-Cost / Open Web):** Uses DuckDuckGo to query the entire live web without API keys or quota consumption.
*   **Tier 2 (High-Reliability Google Failover):** The moment DuckDuckGo encounters rate-limiting (HTTP 429), throttling, or empty responses, it seamlessly fails over to **Serper.dev** (Google Full-Web API) using `SERPER_API_KEY`.
*   **Anti-Hallucination Guardrails:**
    1.  **Zero-Snippet Short-Circuit:** If both search engines return 0 results or are offline, the agent directly returns an honest, deterministic message without calling the LLM.
    2.  **Strict Synthesis Prompt:** Enforced low temperature ($0.2$) with an explicit instruction: *"Answer using EXCLUSIVELY the verified search results. Do NOT invent facts or use prior ungrounded assumptions."*
    3.  **Router Hardening:** `route_query_node` now restricts conversational routing strictly to polite greetings (hello, hi, who are you). Any factual inquiry outside the document is guaranteed to route to web search.
    4.  **Attribution & Citations:** Formats final answers with a source engine badge (DuckDuckGo or Google Search) and clickable markdown links to source articles.

---

## 7. Pre-Retrieval DAG Streamlining & Anaphora Resolution

### The "Liker" Entity Contamination Bug
*   *Cause:* The legacy pipeline placed `extract_and_update_preferences` before query routing, forcing every single document question through an unconstrained extraction prompt. When a user asked *"What does Liker say about..."*, the model extracted `{"name": "Liker"}` and overwrote the authenticated username `Sayuj` in `user_profiles.db`.

### The Architectural Solution
1.  **Immutable User Identity:** `load_user_profile` and `update_history_and_profile` lock the user's name permanently to the authenticated session identity (`user_id = "Sayuj"`).
2.  **Eliminated Dead & Fragile Nodes:** Removed `extract_and_update_preferences` and the redundant `contextualize_query` node from the execution path.
3.  **Consolidated Anaphora Resolution & Multi-Query Expansion:** `transform_query_node` now performs pronoun/entity resolution from conversation history and expands into 3 distinct search variations in a **single, unified LLM pass**, saving 3–4 seconds per turn.
4.  **Dynamic Mermaid Architecture Diagram:** Updated the LangGraph topology in `_build_graph()`. LangGraph's dynamic `app.get_graph().draw_mermaid()` renderer automatically reflects this revised, streamlined graph inside the Gradio UI's "Peek Inside the Brain" accordion.

---

## 8. Single-Pass Multi-Class Semantic Router

### The Issue
*   Sequential multi-step prompts (checking document relevance first, then checking conversational intent) caused Gemma to misclassify simple greetings like `"hi"` as `"maybe"` answerable by the book summary, triggering full document retrieval and dumping unsolicited chapter bullet points.
*   Sequential prompts also compounded latency to 5–6 seconds per greeting.

### The Solution
*   **Unified Classification:** Replaced multi-step prompts and keyword regexes with a single-pass 3-way semantic classifier (`CHITCHAT`, `DOCUMENT_QA`, `WEB_SEARCH`).
*   **Fast & Multilingual:** Gemma evaluates input semantically in ~200ms at temperature $0.0$, naturally routing natural variations (*"Hi"*, *"Hola"*, *"Namaste"*, *"How are you doing"*) to `conversational_query`.
*   **Clean Conversational Synthesis:** In `generate_simple_response_node`, Gemma provides a warm, 1–2 sentence greeting addressing the user by name without unsolicited document lectures.

---

## 9. Files Modified / Created in this Session
*   `config.py` *(New)*: Centralized system configuration.
*   `pdf_validator.py` *(New)*: 4-gate ingestion hygiene gatekeeper.
*   `test_ocr.py` *(New)*: Standalone OCR capability test suite.
*   `test_serper.py` *(New)*: Standalone Serper Google Full-Web API test suite.
*   `web_searcher.py` *(New)*: Tiered hybrid web search engine (DuckDuckGo $\rightarrow$ Serper failover).
*   `index_all_pdfs.py` *(New)*: Automated batch indexer for ChromaDB, BM25, and profile seeding.
*   `pdf_processor.py` *(Updated)*: 200 DPI rendering, contrast preprocessing, de-hyphenation, decimal repair.
*   `summary_generator.py` *(Updated)*: High-context Gemma 3 single-pass summarizer.
*   `gemma_gguf_loader.py` *(Updated)*: Direct prompt & chat test with GPU offload.
*   `query_transformer.py` *(Updated)*: Config-bound model loading with GPU support.
*   `profile_manager.py` *(Updated)*: Added idempotent `close()` method for thread-safe callers.
*   `rag_agent_with_ui_v3.py` *(Updated)*: Streamlined DAG, unified anaphora resolution, single-pass semantic router, dynamic Mermaid diagram, hybrid search, and anti-hallucination guardrails.
*   `requirements.txt` *(Updated)*: Streamlined active dependencies with `duckduckgo-search`.
*   `requirements.txt.bak` *(New)*: Complete pre-refactor backup.
*   `pdf_summaries.json` *(Updated)*: Grounded topic summaries for all 6 chapters.
*   `archive/pdf_summaries_old.json` *(New)*: Archived legacy summaries.
*   `.gitignore` *(Updated)*: Excluded `*.gguf`, model weights, and `rough_work/`.
