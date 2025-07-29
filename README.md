# on-device-rag-agent

Agentic RAG using Gemma model & LangGraph Framework

This project implements a sophisticated, on-device Retrieval-Augmented Generation (RAG) agent that leverages the power of a local Gemma model, the flexibility of the LangGraph framework, and a Gradio-based user interface for easy interaction. The agent can process and chat with PDF documents, utilizing a hybrid retrieval system that combines dense and sparse search methods for optimal results.

## Core Modules

The project is structured into several key Python modules, each responsible for a specific part of the RAG pipeline:

-   **`rag_agent_with_ui_v3.py`**: The main entry point of the application. It orchestrates the entire RAG process, from user input to final response generation, and integrates with a Gradio UI for a seamless user experience.

-   **`query_transformer.py`**: Takes a user's query and transforms it into multiple, semantically similar queries. This query expansion technique helps to improve the recall of the retrieval system.

-   **`hybrid_retriever_v4.py`**: The core of the retrieval system. It combines the results of both dense (vector-based) and sparse (keyword-based) search to retrieve the most relevant document chunks for a given query.

-   **`retrieval_cache_v4.py`**: Implements a semantic cache for retrieval results. This helps to speed up the retrieval process for repeated or similar queries.

-   **`embedding_model.py`**: Responsible for loading the sentence-transformer model and generating embeddings for text chunks.

-   **`profile_manager.py`**: Manages user profiles, allowing for a personalized user experience.

-   **`pdf_manager.py`**: Handles the processing and indexing of PDF documents. It extracts text and images, chunks the text, and creates the necessary indexes for the retrieval system.

-   **`pdf_processor.py`**: A utility module for extracting text and images from PDF files.

-   **`text_chunker.py`**: Chunks the extracted text into smaller, more manageable pieces.

-   **`text_utils.py`**: Provides utility functions for text processing, such as tokenization and stop-word removal.

-   **`vector_db.py`**: Manages the vector database (ChromaDB) for storing and retrieving text embeddings.

-   **`bm25_indexer_v4.py`**: Creates and manages the BM25 index for sparse retrieval.

## How to Run

1.  **Install dependencies:**

    ```bash
    uv pip install -r requirements.txt
    ```

2.  **Set up Google API Keys (Optional, for Web Search):**
    If you wish to enable web search functionality, you need to provide your Google Custom Search API Key and Search Engine ID.
    Create a `.env` file in the project root (or copy `.env.example` and rename it to `.env`) and populate it with your keys:

    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    SEARCH_ENGINE_ID="your_search_engine_id_here"
    ```
    You can obtain these keys by following the instructions [here](https://developers.google.com/custom-search/v1/overview).

3.  **Place your PDF files** in the `pdf_files/source` directory.

4.  **Run the Gradio UI:**

    ```bash
    python rag_agent_with_ui_v1.py
    ```

5.  **Open your web browser** and navigate to the URL provided by Gradio (usually `http://127.0.0.1:7860`).

6.  **Select a PDF**, enter a user ID, and start chatting with your document!