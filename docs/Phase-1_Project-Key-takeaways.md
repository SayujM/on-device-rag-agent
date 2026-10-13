# Phase 1: Project Key Takeaways - On-Device RAG Agent

This document summarizes the key architectural patterns, modules, and takeaways from the **On-Device RAG Agent** project (analyzed on Monday, June 29, 2026), specifically focusing on `archive/rag_agent_v4.py` and its supporting codebase. This serves as a primary reference for building a new, state-of-the-art on-device RAG system.

---

## 1. Architectural Deep-Dive: `archive/rag_agent_v4.py`

At its core, `rag_agent_v4.py` is a **stateful, routing-based RAG Agent** built using **LangGraph**. It models a user's conversational turn as a directed graph where each processing step is an isolated, testable "node." State is preserved across multiple turns using an SQLite database checkpoint saver (`SqliteSaver`), enabling contextual, long-running multi-turn conversations.

### A. Core State Architecture (`AgentState`)
Instead of passing variables manually between functions, LangGraph passes a single unified `AgentState` object. Key fields tracked in this state include:
*   `initial_query` & `original_query`: Track both the raw user input and transformed queries (acting as fallback/retries).
*   `query_route`: Set by the routing node to dynamically path the execution tree (e.g., `rag_query`, `conversational_query`, `web_search_query`, or `ingest_pdf`).
*   `chat_history` & `summarized_history`: Stores message objects and pre-summarizes older messages to prevent the LLM context window from overflowing.
*   `current_turn_messages` & `current_conversation_summary`: Captures step-by-step telemetry of the *current* turn (e.g., "Transformed query to X", "Retrieved Y docs"), enabling the agent to know *why* a retrieval succeeded or failed.
*   `personalization_context`: Holds the loaded JSON user profile to inject user-specific tones (e.g., formal/informal) or context into the LLM prompt.
*   `retrieved_chunks` & `transformed_queries`: Real-time data storage during search execution.

---

### B. The LangGraph Node Workflow (Processing Steps)

The execution of a single turn follows this logical pipeline:

1.  **`load_user_profile`**: Fetches the user's past interaction history and preferences from an SQLite database.
2.  **`add_initial_query_to_history` & `summarize_history`**: Maintains context efficiently. If the history grows beyond 6 messages, it uses the local GGUF model to summarize older exchanges, maintaining a condensed summary along with the last 4 messages.
3.  **`route_query`**: Acts as a semantic router. It checks for commands like `/ingest <path>`, assesses document relevance (based on pre-computed document summaries), and routes the execution path.
4.  **`transform_query`**: Uses the local GGUF model (Gemma) to rewrite the query. It contextualizes the query with the chat history and user profile to produce a standalone query optimized for search, then generates several query variations.
5.  **`retrieve_documents`**: Queries the retrieval pipeline (using cached hits where possible).
6.  **`grade_retrieval`**: Employs the LLM to verify whether the retrieved context can actually answer the user's question. This prevents "hallucinatory grounding."
7.  **`summarize_current_turn`**: Synthesizes the results of the retrieval steps into a short telemetry summary.
8.  **`rewrite_query` (Conditional Loop)**: If the grade is `no`, the query is rewritten from a *fundamentally different approach* using the previous failure reason. It retries the search loop once before routing to a graceful fallback.
9.  **`generate_response` / `generate_simple_response` / `generate_failure_response`**: Synthesizes the final output grounded strictly in the retrieved context (or falls back to conversational memory or web search if configured).
10. **`update_history_and_profile` & `cleanup_state`**: Updates chat histories, summarizes preferences/topics to save back to the user profile, resets runtime lists (to save memory), and saves state checkpoints to SQLite.

---

## 2. Module Analysis: Roles & Responsibilities

The codebase separates concerns across specialized local modules:

| Module File | Responsibility / Role in System | Key Algorithms / Tech Used |
| :--- | :--- | :--- |
| **`query_transformer.py`** | Paraphrases a single user question into multiple semantic variations to maximize retrieval recall. | `llama-cpp-python` (with local GGUF Gemma model running on CPU/GPU), official Gemma instruction format parser. |
| **`hybrid_retriever_v4.py`**| Merges vector search and lexical search results, executing searches across all query variations. | **Consensus RRF (Reciprocal Rank Fusion)**: Ranks chunks primarily by their occurrence frequency across different paraphrased queries, and uses standard RRF scores (`1 / (rank + 1)`) as a secondary tie-breaker. |
| **`retrieval_cache_v4.py`** | An isolated semantic cache layer per-PDF to avoid redundant indexing/search/grading for similar queries. | `ChromaDB` (persistent client) for cosine similarity checks of query embeddings, mapped to cached IDs in a local `SQLite` database. |
| **`embedding_model.py`** | Standardized helper to load models and generate vector embeddings. | `SentenceTransformer` (using `all-MiniLM-L6-v2` locally). |
| **`profile_manager.py`** | Handles persistent user storage. Refactored to be thread-safe (essential for asynchronous web UI contexts). | `sqlite3` using single-query, short-lived transactional connections. |
| **`pdf_manager.py`** | High-level indexer orchestrator. Dictates standardized path directories for text chunks, images, and index files. | Custom directory structure: saves artifacts inside `pdf_files/destination/<pdf_name_without_ext>/`. |
| **`pdf_processor.py`** | Core parser that reads raw PDFs. Handles scanned documents dynamically. | `PyMuPDF` (`fitz`) for fast text extraction. Automatically falls back to **Tesseract OCR** (`pytesseract`) if page text is minimal (<50 characters). Includes standard deviation image filtering. |
| **`text_chunker.py`** | Standardizes splitting document streams into searchable fragments. | LangChain's `RecursiveCharacterTextSplitter` with sentence-aware regular expressions: `["\n\n", "\n", r"(?<=\. )", " ", ""]`. |
| **`text_utils.py`** | Low-level token cleaning utility. | Custom case-folding tokenizer, stripping regex punctuation and filtering words against a lightweight custom English `STOP_WORDS` set. |
| **`vector_db.py`** | Set-up module for ChromaDB instances. | Instantiates `chromadb.PersistentClient` configuring local persistence and setting distance metric space to `cosine`. |

---

## 3. Checklist of Reference Files for Your New Project

To copy these learnings and build a fresh, state-of-the-art on-device RAG system, you need to migrate the **current active modules** (which have been updated with bug fixes and refinements, represented by the `_v4` versions and newer files in the root folder, rather than the deprecated, unrefined scripts in the `/archive` subfolder).

Here is the exact set of files you should copy over:

### A. Core Orchestration & UI (Choose one or both)
*   `rag_agent_with_ui_v3.py` — **Highly Recommended**: This contains the latest complete production version of the LangGraph engine connected to a beautiful web UI via **Gradio**. It supports interactive PDF uploads, user selection dropdowns, dynamic user-profile toggling, and visual chunk citations.
*   `archive/rag_agent_v4.py` — Useful strictly as a CLI-only terminal implementation of the same LangGraph workflow.

### B. Retrieval & Search Engine
*   `hybrid_retriever_v4.py` — The core hybrid search engine with multi-query consensus ranking.
*   `bm25_indexer_v4.py` — Code to build and pickle BM25 search indexes.
*   `retrieval_cache_v4.py` — The ChromaDB + SQLite semantic caching mechanism.
*   `vector_db.py` — General ChromaDB initialization and embedding population scripts.

### C. Content Processing Pipeline
*   `pdf_manager.py` — High-level path configuration and index orchestrator.
*   `pdf_processor.py` — PyMuPDF parsing with conditional Tesseract OCR and embedded image filters.
*   `text_chunker.py` — Recursive text splitter preserving page numbers and unique IDs.

### D. Models & Personalization
*   `query_transformer.py` — Paraphraser/rewriter using the GGUF Gemma model locally.
*   `embedding_model.py` — Dense vectors model wrapper.
*   `profile_manager.py` — Thread-safe user preference / memory manager.

### E. Utilities & Dependencies
*   `text_utils.py` — Case-folding and stop-word filtering tokenizer.
*   `requirements.txt` — Project dependencies (PyMuPDF, rank-bm25, ChromaDB, langchain, langgraph, sentence-transformers, llama-cpp-python, gradio, pytesseract, etc.).
*   `pdf_summaries.json` — Pre-computed summaries of the standard source PDFs to seed query routing.

---

## 4. Key Learnings to Preserve in Your New Project

1.  **Multi-Query Consensus Fusion**: Instead of searching with a single noisy query, rewrite variations of the query, run hybrid search on *all of them*, and prioritize chunks based on **occurrence frequency** across searches. It acts as an ensemble vote.
2.  **Thread-Safe SQLite Connections**: Do not open a single SQLite connection inside a class's `__init__` and leave it open if your UI uses threads (like Gradio/Streamlit). Initialize and close connections per transaction inside your manager classes (just like `profile_manager.py` does).
3.  **Local Semantic Caching**: By verifying queries against a fast local semantic index before running heavy retrieval or calling the LLM, you can drastically reduce search latency and token consumption for repeated queries.
4.  **OCR Fallback**: Standard PDF extractors fail on scanned images. Check the character count of extracted pages; if it's below a threshold (like 50 chars), trigger local Tesseract OCR on the page's rendered image buffer to pull the text seamlessly.
