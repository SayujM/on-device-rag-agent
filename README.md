# On-Device Agentic RAG System

> **High-Performance, Fully Grounded On-Device Document Intelligence**  
> Powered by **Google Gemma 3 4B (GGUF)**, **LangGraph**, and **Gradio** with GPU acceleration, 4-tier ingestion hygiene, 200 DPI OCR, consensus hybrid retrieval, and tiered web search.

---

## 1. System Overview

This project implements an enterprise-grade, stateful **On-Device Retrieval-Augmented Generation (RAG) Agent**. Designed to run completely offline on local consumer hardware (WSL/Linux with NVIDIA RTX GPU or multi-threaded CPU), the agent processes scanned technical books and documents, verifies extraction quality, and conducts grounded multi-turn conversations through a clean **Gradio Web Interface**.

### Key Architectural Pillars
*   **Local LLM Inference:** Powered by **Google Gemma 3 4B Instruction-Tuned (`gemma-3-4b-it-Q4_K_M.gguf`)** via `llama-cpp-python` with CUDA layer offloading, providing rapid sub-second generation and a native 128k context window.
*   **Ingestion Hygiene Gatekeeper (`pdf_validator.py`):** Enforces a clean-room invariant across a 4-tier storage lifecycle (`staging/` $\rightarrow$ `source/` $\rightarrow$ `destination/` & `quarantine/`), verifying magic bytes, size limits, password encryption, and font CMap health before indexing.
*   **High-Resolution OCR Extraction (`pdf_processor.py`):** 200 DPI rendering with grayscale autocontrast preprocessing, Tesseract neural LSTM flags, automatic de-hyphenation, decimal punctuation repair (`8.3`), and watermark removal.
*   **Single-Pass 3-Way Semantic Router:** Classifies queries in ~200ms into `CHITCHAT`, `DOCUMENT_QA`, or `WEB_SEARCH`, eliminating brittle regexes and false-positive document lookups.
*   **Consolidated Anaphora Resolution & Multi-Query Expansion:** Resolves pronouns from conversation history and expands the search query into 3 distinct variations in a single, high-context LLM pass.
*   **Consensus RRF Hybrid Retrieval (`hybrid_retriever_v4.py`):** Combines dense vector search (`all-MiniLM-L6-v2` in ChromaDB) and sparse keyword search (`rank-bm25`) using consensus Reciprocal Rank Fusion.
*   **Tiered Hybrid Web Search (`web_searcher.py`):** Queries **DuckDuckGo** (zero-quota free tier) and automatically fails over to **Serper.dev** (Google Full-Web API) if rate-limited, backed by strict anti-hallucination guardrails and source citations.
*   **Interactive Gradio Dashboard:** Features dynamic topic summary banners, a live **Mermaid architecture diagram** ("Peek Inside the Brain"), turn-by-turn telemetry, and citation badges.

---

## 2. Repository Structure

```
on-device-rag-agent/
├── README.md                         # Main project documentation
├── requirements.txt                  # Active minimal dependencies
├── config.py                         # Centralized system paths, model names & database configs
├── run_app.py                        # Clean application entrypoint launcher
├── .env.example                      # Environment variable configuration template
├── .gitignore                        # Excludes weights, checkpoints, rough_work
│
├── src/                              # Core Application Source Code
│   ├── ingestion/                    # PDF parsing, OCR, and validation
│   │   ├── pdf_validator.py          # 4-stage ingestion hygiene gatekeeper
│   │   ├── pdf_processor.py          # 200 DPI OCR extraction & contrast preprocessing
│   │   ├── pdf_manager.py            # Processing coordinator & artifact manager
│   │   ├── text_chunker.py           # Recursive sentence-aware character text splitter
│   │   └── text_utils.py             # Case-folding tokenizer & stop-word cleaner
│   ├── retrieval/                    # Search, vector DB, and caching
│   │   ├── hybrid_retriever_v4.py    # Consensus RRF hybrid search (Dense + Sparse)
│   │   ├── bm25_indexer_v4.py        # BM25 sparse indexer & serializer
│   │   ├── vector_db.py              # ChromaDB client & collection manager
│   │   ├── retrieval_cache_v4.py     # Semantic vector cache for low-latency repeat queries
│   │   └── web_searcher.py           # Tiered hybrid search (DuckDuckGo -> Serper Google)
│   ├── models/                       # Model wrappers and user state
│   │   ├── embedding_model.py        # SentenceTransformer (all-MiniLM-L6-v2) loader
│   │   ├── query_transformer.py      # Paraphrasing & query expansion module
│   │   └── profile_manager.py        # Thread-safe SQLite user memory manager
│   └── agent/                        # LangGraph workflow & Gradio UI
│       └── rag_agent_with_ui_v3.py   # Stateful RAG engine & Gradio dashboard
│
├── scripts/                          # Pipeline & Maintenance Scripts
│   ├── index_all_pdfs.py             # Batch pre-indexer for ChromaDB & BM25
│   └── summary_generator.py          # High-context Gemma 3 single-pass chapter summarizer
│
├── tests/                            # Verification & Diagnostic Test Suites
│   ├── test_ocr.py                   # Standalone Tesseract & PIL OCR verification test
│   ├── test_serper.py                # Standalone Serper Google Full-Web API test
│   └── gemma_gguf_loader.py          # Standalone GGUF model load test
│
├── data/                             # Metadata & Seed Knowledge Bases
│   ├── pdf_summaries.json            # Grounded topic summaries for source chapters
│   ├── user_memory.json              # Seed personalization profile for user Sayuj
│   └── user_profiles.db              # SQLite user profiles
│
├── docs/                             # Dedicated documentation & architectural notes
│   ├── Session_Notes_Model_Upgrade_and_OCR_Hygiene.md
│   ├── Phase-1_Project-Key-takeaways.md
│   └── personalization_context_enhancement.md
│
├── pdf_files/                        # Document lifecycle storage
│   ├── staging/                      # Untrusted landing zone for incoming PDFs
│   ├── source/                       # Verified clean-room PDFs ready for indexing
│   ├── quarantine/                   # Rejected PDFs & audit JSON reports
│   └── destination/                  # Chunks, BM25 pickles & metadata per document
│
├── chroma_db/                        # Persistent ChromaDB vector database
├── langgraph_checkpoints/            # Multi-turn SQLite conversation checkpoints
├── rough_work/                       # Scratchpad & experimental ADK files (Git ignored)
└── archive/                          # Legacy, deprecated, and superseded scripts
```

---

## 3. Quickstart & Installation

### Prerequisites
*   **Operating System:** Linux or Windows Subsystem for Linux (WSL2 / Ubuntu 22.04+).
*   **Hardware:** 8+ GB RAM (16 GB recommended). NVIDIA GPU with 6+ GB VRAM recommended for GPU acceleration.
*   **System Binaries:**
    ```bash
    sudo apt update && sudo apt install -y build-essential tesseract-ocr tesseract-ocr-eng
    ```

### Step 1: Set Up Python Environment
We recommend using [`uv`](https://github.com/astral-sh/uv) for fast package resolution:
```bash
# Create virtual environment with Python 3.12
uv venv .venv --python 3.12

# Activate environment
source .venv/bin/activate

# Install curated dependencies
uv pip install -r requirements.txt
```

### Step 2: Download Model Weights
Download the official quantized **Gemma 3 4B GGUF** model (~2.49 GB) into the workspace root:
```bash
curl -L -C - -o gemma-3-4b-it-Q4_K_M.gguf https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf
```

### Step 3: Configure Environment Variables (Optional for Web Search)
Copy `.env.example` to `.env` and add your **Serper API Key** for Google Full-Web search failover:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
SERPER_API_KEY="your_serper_api_key_here"
```
*(Get a free key with 2,500 queries at [serper.dev](https://serper.dev)).*

---

## 4. Ingestion & Indexing Workflow

### 1. Stage and Validate Documents
Drop any PDF into `pdf_files/staging/` and run the hygiene audit:
```bash
python3 src/ingestion/pdf_validator.py
```
*   Healthy files are automatically promoted to `pdf_files/source/`.
*   Corrupt, encrypted, or invalid files are isolated into `pdf_files/quarantine/` with an audit JSON.

### 2. Generate Grounded Chapter Summaries
To generate high-context chapter summaries using Gemma 3 4B:
```bash
# Summarize newly added source PDFs
python3 scripts/summary_generator.py

# Or force re-generation for all source files
python3 scripts/summary_generator.py --force
```

### 3. Pre-Index Documents & Seed Profile
Pre-build ChromaDB vector collections, BM25 indices, and seed user memory:
```bash
python3 scripts/index_all_pdfs.py
```

---

## 5. Running the Application

Launch the interactive Gradio RAG Agent:
```bash
python3 run_app.py
```

1.  Open your browser at **`http://127.0.0.1:7860`** (or use the generated public live link).
2.  **Select a PDF** from the dropdown (e.g. `The-toyota-way-second-edition-01_chapter_1.pdf`).
3.  **Enter User ID:** `Sayuj`.
4.  Click **"Start Chat"** (loads instantly with pre-computed indices).
5.  **Interactive Features:**
    *   **Architecture Visualization:** Expand the *"Peek Inside the Brain"* accordion to view the live Mermaid workflow diagram.
    *   **Document Q&A:** Ask chapter-specific questions to see consensus RRF retrieval, LLM retrieval grading, and grounded answers with citations.
    *   **Conversational Chitchat:** Say *"Hi"*, *"Hola"*, or *"How are you?"* for instant sub-second conversational replies.
    *   **Web Fallback:** Ask out-of-document factual questions to trigger the tiered DuckDuckGo / Serper Google search failover.

---

## 6. Detailed Architectural Documentation

For full technical specifications, empirical failure reproductions, and architectural notes, see:
*   **[Session Notes: Model Upgrade, Ingestion Hygiene & OCR](docs/Session_Notes_Model_Upgrade_and_OCR_Hygiene.md)**
*   **[Phase 1 Architectural Key Takeaways](docs/Phase-1_Project-Key-takeaways.md)**
*   **[Personalization Context Enhancement](docs/personalization_context_enhancement.md)**
