### Version 2: Dynamic Document Handling, Intelligent Routing, and Web Search Integration

**Focus:** Enhancing the RAG agent's ability to manage multiple document sources, intelligently route user queries, and leverage external web search for comprehensive information retrieval.

**Scope:** Implement dynamic PDF ingestion and selection, refine query routing logic, and integrate web search capabilities.

---

### Relevant Details for Version 2:

#### Overall Approach & Key Considerations

*   **Modularity:** Continue breaking down the agent into distinct, testable components.
*   **Efficiency:** Prioritize efficient processing and retrieval, especially with dynamic document loading.
*   **LangGraph Orchestration:** Utilize LangGraph for managing complex flows, including new routing paths and conditional logic.
*   **Hybrid Search:** Maintain the combination of dense (semantic) and sparse (keyword) retrieval.
*   **Persistence:** Ensure all processed data (ChromaDB, BM25, retrieval cache) is persistently stored and isolated per document.
*   **LLM-driven Logic:** Leverage the `gemma-3n` model for intelligent routing decisions and effective summarization of web search results.
*   **User Experience:** Provide clear feedback to the user regarding document ingestion status and search strategies.

---

#### 1. Dynamic PDF Ingestion and Selection

**Objective:** Allow users to select and load different PDF documents dynamically, ensuring efficient processing and isolated retrieval for each document. The system must guarantee that a PDF is processed only once and that retrieval is strictly limited to the currently selected document.

**Key Considerations:**

*   **User Selection:** The agent will prompt the user to select a PDF from a designated source directory (`~/repos/multi-modal_RAG/pdf_files/source`).
*   **"Process Once" Guarantee:** A fixed naming convention and artifact location will be used to determine if a PDF has already been processed.
*   **Isolated Storage:** Each PDF's processed data (ChromaDB embeddings, BM25 index, retrieval cache) will be stored in a dedicated, isolated manner.
*   **Dynamic Loading:** The `HybridRetriever` and `RetrievalCache` will be dynamically configured to use the data corresponding to the currently selected PDF.

**Detailed Plan:**

**A. New Utility File: `pdf_manager.py`**
    *   **`get_pdf_output_dirs(pdf_name_without_ext: str) -> dict`:**
        *   **Purpose:** To provide standardized absolute paths for all processed artifacts related to a given PDF.
        *   **Input:** `pdf_name_without_ext` (e.g., "The-toyota-way-second-edition-chapter_1").
        *   **Output:** A dictionary containing absolute paths:
            *   `pdf_specific_output_dir`: `BASE_DIR/pdf_files/destination/<pdf_name_without_ext>`
            *   `extracted_images_dir`: `.../extracted_images`
            *   `text_chunks_json_path`: `.../text_chunks.json`
            *   `bm25_index_pkl_path`: `.../bm25_index.pkl`
            *   `retrieval_cache_db_path`: `.../retrieval_cache.db`
    *   **`check_pdf_processed(pdf_name_without_ext: str) -> bool`:**
        *   **Purpose:** To efficiently determine if a PDF has already been processed and its artifacts saved locally.
        *   **Input:** `pdf_name_without_ext`.
        *   **Implementation:** Uses `get_pdf_output_dirs()` and checks for the existence of `text_chunks_json_path` and `bm25_index_pkl_path`.
        *   **Output:** `True` if processed, `False` otherwise.
    *   **`process_and_index_pdf(pdf_path: str) -> tuple[bool, str]`:**
        *   **Purpose:** Orchestrates the end-to-end processing and indexing of a PDF.
        *   **Input:** Absolute path to the source PDF (e.g., from `pdf_files/source`).
        *   **Implementation:**
            *   Derives `pdf_name_without_ext` (which will also serve as the ChromaDB `collection_name`).
            *   Calls `check_pdf_processed()`. If `True`, returns `(True, "Already processed.")`.
            *   If `False`:
                *   Gets output directories using `get_pdf_output_dirs()`.
                *   Ensures all necessary output directories exist (`os.makedirs`).
                *   Calls `pdf_processor.process_pdf(pdf_path, extracted_images_dir)` to extract `extracted_texts` and `saved_image_paths`. (The existing logic in `pdf_processor.py` for text extraction, OCR, and image saving is fully maintained).
                *   **Chunking:** Performs chunking of `extracted_texts` into `chunks_with_metadata`. Each chunk will include metadata such as `source_file` (full PDF path), `page_number`, `chunk_id`, `chunk_type`.
                *   Saves `chunks_with_metadata` to `text_chunks_json_path`.
                *   Calls `bm25_indexer.create_and_save_bm25_index(chunks_with_metadata, bm25_index_pkl_path)`.
                *   **Calls `hybrid_retriever.add_documents_to_chroma(chunks_with_metadata, collection_name=pdf_name_without_ext)`**.
            *   **Output:** `(True, "Successfully processed and indexed.")` or `(False, "Error message.")`.

**B. Modifications to `hybrid_retriever.py`**
    *   **`__init__(self, db_dir: str, embedding_model_name: str)`:**
        *   **Purpose:** Initialize the retriever without tying it to a specific PDF's data.
        *   **Implementation:**
            *   Initializes `self.chroma_client` (using `get_chroma_client(db_dir)`).
            *   Initializes `self.embedding_model`.
            *   Sets all active data components to `None` or empty: `self.active_chroma_collection = None`, `self.active_bm25_index = None`, `self.active_bm25_chunk_ids = []`, `self.active_all_chunks = []`, `self.active_content_map = {}`.
    *   **`load_pdf_for_retrieval(self, pdf_name_without_ext: str)`:**
        *   **Purpose:** To load and activate the retrieval components for a specific PDF.
        *   **Input:** `pdf_name_without_ext`.
        *   **Implementation:**
            *   Uses `pdf_manager.get_pdf_output_dirs()` to locate the `text_chunks.json` and `bm25_index.pkl` for the specified PDF.
            *   Sets `self.active_chroma_collection = self.chroma_client.get_or_create_collection(name=pdf_name_without_ext)`.
            *   Loads the chunks from `text_chunks.json` into `self.active_all_chunks` and `self.active_content_map`.
            *   Loads the BM25 index from `bm25_index_pkl_path` into `self.active_bm25_index` and `self.active_bm25_chunk_ids`.
    *   **`add_documents_to_chroma(self, chunks: List[dict], collection_name: str)`:**
        *   **Purpose:** To add document chunks to a *specific* ChromaDB collection.
        *   **Input:** `chunks` (list of dicts), `collection_name`.
        *   **Implementation:**
            *   Gets/creates the target ChromaDB collection using `self.chroma_client.get_or_create_collection(name=collection_name)`.
            *   Generates embeddings for the `chunks`.
            *   Adds the documents to *that specific collection*.
    *   **Modification to `dense_search`:**
        *   **Purpose:** Ensure dense search queries the currently active ChromaDB collection.
        *   **Implementation:** Use `self.active_chroma_collection.query()`.
    *   **Modification to `sparse_search` and `retrieve`:**
        *   **Purpose:** Ensure sparse search and hybrid fusion use the currently active BM25 index and chunk data.
        *   **Implementation:** Use `self.active_bm25_index`, `self.active_bm25_chunk_ids`, `self.active_all_chunks`, and `self.active_content_map`.

**C. Modifications to `bm25_indexer.py`**
    *   **`create_and_save_bm25_index(chunks_with_metadata: List[dict], output_index_path: str)`:**
        *   **Purpose:** To create and persist a BM25 index from a list of chunks.
        *   **Input:** `chunks_with_metadata` (list of dicts), `output_index_path`.
        *   **Implementation:** Accepts `chunks_with_metadata` directly (instead of a file path) to make it a reusable function.

**D. Modifications to `retrieval_cache.py`**
    *   **`__init__(self, cache_db_path: str, embedding_model, similarity_threshold: float = 0.7)`:**
        *   **Purpose:** To initialize the cache with a specific SQLite database file.
        *   **Implementation:** Accepts `cache_db_path` as a required argument and initializes its SQLite connection using this path.
    *   **`check_cache` and `add_to_cache`:**
        *   **Purpose:** Ensure cache operations are performed on the correct, isolated database.
        *   **Implementation:** Operate on the SQLite connection established in `__init__`.

**E. Modifications to `rag_agent_v4.py` (Main Loop and `ingest_pdf_node`):**
    *   **Initial Setup in `run_agent()`:**
        *   **PDF Selection:** List available PDFs in `pdf_files/source` and prompt the user to select one.
        *   **Processing & Loading:**
            *   Call `pdf_manager.process_and_index_pdf()` for the selected PDF.
            *   Call `hybrid_retriever.load_pdf_for_retrieval()` to set the active PDF for retrieval.
            *   **Re-initialize `retrieval_cache`:** Create a new `RetrievalCache` instance with the `retrieval_cache_db_path` obtained from `pdf_manager.get_pdf_output_dirs()` for the active PDF. This instance will be passed to the nodes that use it.
    *   **`ingest_pdf_node` (repurposed):**
        *   **Purpose:** Handles the `/ingest <pdf_path>` command to process and switch to a new PDF.
        *   **Implementation:**
            *   Extracts `pdf_path` from `state.original_query`.
            *   Calls `pdf_manager.process_and_index_pdf()` for the new PDF.
            *   Calls `hybrid_retriever.load_pdf_for_retrieval()` to switch the active PDF.
            *   **Re-initialize `retrieval_cache`:** Create a new `RetrievalCache` instance with the correct `retrieval_cache_db_path` for the newly loaded PDF.

---

#### 2. Intelligent Query Routing

**Objective:** Improve the `route_query_node` to intelligently decide whether a user query requires RAG (from indexed documents), external web search, or a simple conversational response. This aims to prevent unnecessary RAG attempts for irrelevant queries and improve overall agent efficiency.

**Key Considerations:**

*   **LLM-based Relevance Judgment:** Use the LLM to assess if a query is likely answerable by the currently indexed documents.
*   **Document Summary:** Provide the LLM with a high-level summary of the indexed documents to aid in relevance judgment.
*   **Clear Routing Paths:** Define distinct paths for RAG, web search, and conversational responses.

**Detailed Plan:**

**A. `AgentState` Update:**
    *   Add `indexed_document_summary: str = ""` to store a high-level description of the currently loaded documents. This will be initialized with a default value (e.g., "No documents loaded.") and updated upon PDF ingestion.

**B. `route_query_node` Enhancements:**
    *   **Command Handling:** Prioritize special commands like `/ingest <pdf_path>`. If the command is recognized, route to `ingest_pdf`. If the PDF path is invalid, route to `conversational_query` with an error message.
    *   **Document Relevance Check (LLM Call):**
        *   **Prompt:** "Given the user's question and the summary of the documents this agent has access to, is the question likely to be answerable by these documents? Respond with 'yes' if it's highly relevant, 'no' if it's not relevant, or 'maybe' if it's partially relevant or requires external search."
        *   **Context:** Provide `state.indexed_document_summary` and `state.initial_query`.
        *   **Output:** `relevance_grade` (`yes`/`no`/`maybe`).
    *   **Conversational/Web Search Check (LLM Call - if not document-relevant):**
        *   **Condition:** If `relevance_grade` is `no`.
        *   **Prompt:** "Given the user's question, is it a simple greeting, a question about the AI itself, or a general conversational query that does NOT require specific information retrieval from documents or the web? Respond with 'conversational' or 'needs_web_search'."
        *   **Context:** `state.initial_query`.
        *   **Output:** `conversational_type` (`conversational`/`needs_web_search`).
    *   **Routing Logic:**
        *   If `relevance_grade` is `yes` or `maybe` -> `rag_query`
        *   If `relevance_grade` is `no` AND `conversational_type` is `conversational` -> `conversational_query`
        *   If `relevance_grade` is `no` AND `conversational_type` is `needs_web_search` -> `web_search_query`

**C. `ingest_pdf_node` Update:**
    *   After a successful PDF ingestion and loading, ensure it updates `state.indexed_document_summary` to reflect the newly loaded document's content. This summary can initially be a simple string (e.g., "Documents now include content from [PDF Name].") and can be enhanced with an LLM-generated summary of the PDF's content in a future version.

---

#### 3. Web Search Integration

**Objective:** Provide external search capabilities using Tavily (or Google Web Search) for queries not covered by indexed documents, and as a fallback mechanism for RAG failures.

**Key Considerations:**

*   **Tool Usage:** Utilize the `google_web_search` tool (as Tavily API key is not provided yet).
*   **Result Summarization:** Summarize web search results effectively using an LLM to provide concise answers.
*   **Integration Points:** Integrate web search into the initial routing and as a fallback after RAG failures.

**Detailed Plan:**

**A. New Node: `web_search_node(state: AgentState) -> dict`:**
    *   **Purpose:** To perform a web search and summarize the results.
    *   **Input:** `state.initial_query` (or a rephrased query if a dedicated rephrasing node for web search is added later).
    *   **Implementation:**
        *   Calls `default_api.google_web_search(query=search_query)`.
        *   Extracts relevant snippets (e.g., top 3-5).
        *   Uses an LLM call to summarize the snippets to answer `state.initial_query`.
        *   Handles cases where no relevant search results are found.
        *   **Output:** `final_answer` containing the summarized web search result or a "not found" message.

**B. Routing from `route_query_node`:**
    *   Add `web_search_query` as a possible `query_route` from `route_query_node`.

**C. Fallback from RAG Failure:**
    *   **New Conditional Edge:** Introduce a conditional edge from `generate_failure_response_node` to `web_search_node`.
    *   **`should_route_from_failure(state: AgentState) -> str`:** A new conditional function that determines if, after a RAG failure, a web search should be attempted. Initially, this can always return `"web_search_query"`. In future versions, this could be made more intelligent (e.g., based on the `current_conversation_summary` of the RAG failure).

**D. `update_history_and_profile_node`:**
    *   Ensure that the `final_answer` generated by the `web_search_node` is also captured and appended to the `chat_history`.

---
