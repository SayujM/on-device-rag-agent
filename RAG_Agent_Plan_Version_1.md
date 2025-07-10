### Version 1: RAG Agent with Chat Ability in English

**Focus:** Core RAG functionality for English text.

**Scope:** Implement Phase 1 (English content), Phase 2, Phase 3 (English responses), and Phase 4 (English context). Phase 0 (voice/multilingual) is deferred.

---

### Relevant Details for Version 1:

#### Overall Approach & Key Considerations (for Version 1)

*   **Modularity:** Breaking down the agent into distinct, testable components.
*   **Efficiency:** Prioritizing libraries and techniques known for CPU performance and memory efficiency.
*   **LangGraph Orchestration:** Using LangGraph to manage the complex flow of the agent, including iterative retrieval and conditional logic.
*   **Hybrid Search:** Combining the strengths of dense (semantic) and sparse (keyword) retrieval.
*   **Multi-modal Handling:** Separately processing text and images. VLM integration for generating image descriptions has been identified as requiring significant further research and development due to current model limitations (e.g., hallucinations), thus it is deferred to a later version.
*   **Persistence:** Implementing caching for retrieved results to minimize redundant work.
*   **LLM-driven Query Transformation:** Leveraging the `gemma-3n` model for dynamic and semantically varied query generation.
*   **Semantic Caching:** Implementing a cache that understands query similarity, not just exact matches.
*   **Personalization Hooks:** Laying the groundwork for user-specific context integration.

#### Phase 1: Data Ingestion & Core Indexing (for Version 1)

*   **Step 1.1: Document Loading & Preprocessing**
    *   **Objective:** Extract raw text and images from PDF files.
    *   **Implementation:** Use a library like `PyMuPDF` (fitz) for efficient PDF parsing. It can extract both text and images. For scanned documents, OCR (e.g., Tesseract) will be used to extract text from rendered page images if `PyMuPDF`'s text extraction is insufficient.
    *   **Output:** Raw text content per page, and image files (e.g., PNG/JPEG) saved to a designated local directory for visual inspection.
    *   **Robustness:** Implement error handling for corrupted PDFs or missing files.

*   **Step 1.2: Visual Language Model (VLM) Integration (Deferred for Version 1)**
    *   **Objective:** Generate textual descriptions for extracted images.
    *   **Challenge Faced:** Initial attempts with LLaVA-v1.5-7B (Q4_K_M GGUF) demonstrated significant limitations, including severe hallucinations and inability to accurately interpret document-specific visual content (e.g., misinterpreting text-only pages as book covers, or misrepresenting graph details).
    *   **Decision:** Due to these limitations and the need for substantial further research, model selection, and fine-tuning to achieve reliable performance for document image understanding, VLM integration is deferred to a later version. For Version 1, the RAG agent will primarily rely on text-based content.

*   **Step 1.3: Chunking Strategy**
    *   **Objective:** Divide processed text and image descriptions into meaningful, retrievable chunks with rich metadata.
    *   **Text Chunking:** Implement a strategy (e.g., fixed-size with overlap, or more advanced semantic chunking if feasible) for text content.
    *   **Image Chunking:** (Deferred for Version 1) Treat VLM-generated image descriptions as text and chunk them similarly, ensuring metadata links back to the original image and its source. This step is dependent on successful VLM integration.
    *   **Metadata:** Each chunk (text or image description) must include metadata such as: `source_file`, `page_number`, `chunk_id`, `chunk_type` (text/image), and for images, `image_id`.
    *   **Robustness:** Experiment with chunk sizes and overlap to optimize retrieval performance and minimize noise.

*   **Step 1.4: Embedding Model Selection & Setup**
    *   **Objective:** Generate dense vector embeddings for all text chunks. (Image descriptions are deferred for Version 1).
    *   **Selection:** Choose a compact, performant `sentence-transformers` model (e.g., `all-MiniLM-L6-v2` or similar) that runs efficiently on CPU. (Note: For Version 1, this would be an English-focused model).
    *   **Implementation:** Load the embedding model and create embeddings for each chunk.
    *   **Robustness:** Consider batching embedding generation for efficiency.

*   **Step 1.5: Local Vector Database (ChromaDB)**
    *   **Objective:** Store dense embeddings and associated metadata for efficient semantic search.
    *   **Implementation:** Set up ChromaDB (an open-source, in-process vector database) to store the embeddings and all chunk metadata.
    *   **Robustness:** Ensure persistence of the ChromaDB collection on disk.

*   **Step 1.6: BM25 Indexing**
    *   **Objective:** Create a sparse index for keyword-based search.
    *   **Implementation:** Use the `rank_bm25` library to build a BM25 index over the text content of all chunks.
    *   **Tokenization Strategy (Version 1):** For Version 1, tokenization is optimized for **English** using a simple, custom tokenizer (lowercase, punctuation removal, basic stop word filtering). This approach is chosen for its simplicity, transparency, and minimal dependencies. For future versions, especially when adding support for **Indian Languages**, upgrading to a more robust library like **NLTK** for linguistically-aware tokenization and lemmatization will be critical to improve search accuracy.
    *   **Storage:** Persist the BM25 index (e.g., using Python's `pickle` module) to disk for quick loading.
    *   **Robustness:** Handle large indices efficiently.

#### Phase 2: Agentic Retrieval & Hybrid Search (for Version 1)

*   **Step 2.1: Query Transformation Module (LLM-based)**
    *   **Objective:** Rephrase the user's initial query into multiple semantically different, affirmative forms suitable for retrieval.
    *   **Implementation:** Utilize the `gemma-3n` GGUF model (the one we just got working) to generate 3-5 diverse rephrased queries from the original user query. This will involve crafting a specific prompt for the `gemma-3n` model to guide this rephrasing. (Note: For Version 1, this would be for English queries).
    *   **Agentic RAG:** This module will be a key part of the agent's iterative process, generating multiple queries for parallel or sequential retrieval attempts.

*   **Step 2.2: Dense Retriever**
    *   **Objective:** Perform semantic search using the user's (transformed) query against the ChromaDB vector store.
    *   **Implementation:** Query ChromaDB with the embedding of each transformed query to retrieve top-k similar chunks.

*   **Step 2.3: Sparse Retriever**
    *   **Objective:** Perform keyword search using the user's (transformed) query against the BM25 index.
    *   **Implementation:** Query the BM25 index with each transformed query to retrieve top-k relevant chunks based on keyword overlap.

*   **Step 2.4: Hybrid Search Fusion**
    *   **Objective:** Combine and re-rank results from both dense and sparse retrievers.
    *   **Implementation:** Use Reciprocal Rank Fusion (RRF) to merge the ranked lists from the dense and sparse retrievers into a single, highly relevant list of chunks.

*   **Step 2.5: On-Disk Semantic Cache for Retrieved Results**
    *   **Objective:** Store query-result pairs to minimize redundant retrieval for semantically similar queries.
    *   **Implementation:**
        *   **Cache Query Embeddings:** Create a separate, small ChromaDB collection to store embeddings of past user queries (or their transformed versions) as cache keys.
        *   **Associated Results Storage:** Store the actual retrieved results (e.g., a list of chunk IDs and their content/metadata) in a persistent key-value store (e.g., `sqlite3` or `diskcache`) where the key is linked to the query embedding in ChromaDB.
        *   **Similarity Search for Cache Hits:** When a new query comes in, embed it and perform a similarity search against the cache query embeddings. If a sufficiently similar query is found (above a defined threshold), retrieve the cached results from the key-value store.
    *   **Robustness:** Define a similarity threshold for cache hits. Implement cache invalidation strategies if the underlying document set changes.

#### Phase 3: LangGraph Orchestration & Response Generation (for Version 1)

*   **Step 3.1: LangGraph Agent Setup**
    *   **Objective:** Define the state and flow of the RAG agent using LangGraph.
    *   **State Definition:** Define the agent's state (e.g., `user_query`, `transformed_queries`, `retrieved_chunks`, `final_answer`, `personalization_context`).
    *   **Nodes:** Define the following nodes:
        *   `personalization_node`: Reads user profile/memory and injects context into the state.
        *   `query_transform_node`: Calls Step 2.1, potentially using `personalization_context`.
        *   `hybrid_retrieval_node`: Calls Steps 2.2, 2.3, 2.4 (and incorporates Step 2.5 for caching).
        *   `vlm_processing_node`: (Deferred for Version 1)
        *   `iterative_query_node`: Analyzes retrieved chunks; if unsatisfactory, generates new transformed queries (via `query_transform_node`) and routes back to `hybrid_retrieval_node`. This node will use the `gemma-3n` model to assess retrieval quality and decide on re-querying.
        *   `response_generation_node`: Calls Step 3.2.
    *   **Edges & Conditional Routing:** Define the flow between nodes, including conditional edges for iterative retrieval based on the quality of initial results.

*   **Step 3.2: Response Generation with Gemma GGUF**
    *   **Objective:** Generate a coherent and accurate answer using the Gemma GGUF model based on the user query and retrieved context.
    *   **Implementation:** Integrate the Gemma GGUF model (from our `gemma_gguf_loader.py`) into the `response_generation_node`.
    *   **Prompt Engineering:** Craft a sophisticated prompt that instructs Gemma to synthesize information from the retrieved chunks to answer the user's original query, potentially incorporating `personalization_context`. The prompt should emphasize grounding the answer in the provided context. (Note: For Version 1, this would be for English responses).

*   **Step 3.3: Initial Agent Testing**
    *   **Objective:** Verify the end-to-end functionality of the RAG agent with a single query.
    *   **Implementation:** Run the LangGraph agent with a test query and observe its behavior, retrieved results, and final answer.

#### Phase 4: Personalization & User Memory (for Version 1)

*   **Step 4.1: User Profile/Memory Management**
    *   **Objective:** Store and retrieve user-specific information.
    *   **Implementation:** Start with a simple, persistent JSON file or a small `sqlite3` database to store user preferences, past interactions, or explicit memory points.
    *   **Integration:** The `personalization_node` in LangGraph will read from this store.

*   **Step 4.2: Personalization in Query Transformation/Response Generation**
    *   **Objective:** Use the retrieved user context to influence the agent's behavior.
    *   **Implementation:** Pass the `personalization_context` to the `query_transform_node` (to bias query rephrasing) and the `response_generation_node` (to tailor the final answer's tone or content). This will primarily be done via prompt engineering.

#### Robustness & Resource Considerations (for Version 1)

*   **LLM Memory Management:** When using `llama-cpp-python` in standalone scripts, a benign `TypeError` may be observed on exit. This is due to a race condition in the library's garbage collection and does not affect the script's output. This will be resolved in the final application (e.g., the Streamlit app) by implementing explicit model lifecycle management to ensure the model is loaded once and shut down cleanly.
*   **Error Handling & Logging:** Implement comprehensive `try-except` blocks and detailed logging at each stage to facilitate debugging and monitor agent behavior.
*   **Configuration Management:** Use a `config.yaml` or similar file to manage all configurable parameters (model paths, chunk sizes, database paths, VLM parameters, etc.).
*   **Resource Monitoring:** Advise on using system monitoring tools (e.g., `htop`, `nvidia-smi` if GPU is used for VLM) to track CPU, RAM, and VRAM usage during development and testing.
*   **Quantization Levels:** For GGUF models (Gemma and VLM), we can experiment with different quantization levels (e.g., Q4_K_M, Q5_K_M, Q8_0) to find the optimal balance between performance and quality for your specific desktop.
*   **Batch Processing:** Explore opportunities to batch inputs for embedding generation and VLM inference to improve throughput.
*   **Streamlit/CLI Interface:** Once the core agent is functional, we can discuss adding a simple Streamlit UI or a robust CLI interface for interaction.

---

**Explicitly Deferred for Version 1:**

*   **Phase 0: User Interface & Interaction (Voice & Multilingual)** (Entire phase)
*   **Voice Interaction** (from Overall Approach)
*   **Multilingual Support** (from Overall Approach)
*   All "Update" notes related to multilingual capabilities within Phase 1, 2, and 3.
*   **VLM Integration:** Due to significant challenges with current model performance (e.g., hallucinations) for document image understanding, VLM integration for generating image descriptions is deferred to a later version.
*   **Multilingual Model Evaluation** (from Robustness & Resource Considerations)
*   **Tokenization for Indian Languages** (from Robustness & Resource Considerations)
