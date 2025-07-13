import os
import json
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph

# Local module imports
from query_transformer import QueryTransformer
from hybrid_retriever import HybridRetriever
from retrieval_cache import RetrievalCache
from embedding_model import load_embedding_model
from text_utils import tokenize_text

# --- Agent State Definition ---
class AgentState(BaseModel):
    """
    Represents the state of our RAG agent, validated by Pydantic.
    """
    initial_query: str
    original_query: str
    chat_history: List[BaseMessage] = Field(default_factory=list)
    personalization_context: dict = Field(default_factory=dict)
    transformed_queries: List[str] = Field(default_factory=list)
    retrieved_chunks: List[dict] = Field(default_factory=list)
    is_relevant: str = ""
    final_answer: str = ""
    iteration_count: int = 0

# --- Initialize Components ---
print("--- Initializing RAG Agent Components ---")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GGUF_MODEL_PATH = os.path.join(BASE_DIR, "gemma-3n-E4B-it-Q4_K_M.gguf")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
PDF_NAME = "The-toyota-way-second-edition-chapter_1"
PDF_SPECIFIC_DIR = os.path.join(BASE_DIR, "pdf_files", "destination", PDF_NAME)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(BASE_DIR, "retrieval_cache")

query_transformer = QueryTransformer(model_path=GGUF_MODEL_PATH)
llm = query_transformer.llm # Reuse the loaded LLM for all nodes
embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
hybrid_retriever = HybridRetriever(
    db_dir=CHROMA_DB_DIR,
    pdf_specific_dir=PDF_SPECIFIC_DIR,
    embedding_model_name=EMBEDDING_MODEL_NAME
)
retrieval_cache = RetrievalCache(cache_dir=CACHE_DIR, embedding_model=embedding_model, similarity_threshold=0.67)
print("--- All Components Initialized ---")

def personalization_node(state: AgentState) -> dict:
    """
    Node that loads user-specific context from a file.
    """
    print("--- Node: Loading Personalization Context ---")
    try:
        with open("user_memory.json", "r") as f:
            personalization_context = json.load(f)
        print("Loaded personalization context.")
        return {"personalization_context": personalization_context}
    except FileNotFoundError:
        print("No personalization file found. Skipping.")
        return {"personalization_context": {}}

# --- Node Definitions ---
def transform_query_node(state: AgentState) -> dict:
    """
    Node that transforms the user's original query into multiple variations,
    including a personalized one if context is available.
    """
    print("--- Node: Transforming Query ---")
    original_query = state.original_query
    chat_history = state.chat_history
    personalization_context = state.personalization_context

    # Always generate base transformed queries for robustness
    transformed_queries = query_transformer.transform_query(original_query)

    # If there's context, generate an additional, personalized query
    if chat_history or personalization_context:
        # Limit chat history to the last 4 messages (2 turns) to manage context size
        last_messages = chat_history[-4:]
        formatted_history = "\n".join([f"{msg.type}: {msg.content}" for msg in last_messages])

        prompt = f"""<start_of_turn>user
        Based on the following chat history, user profile, and the user's latest question, generate a new, standalone search query.

        Chat History:
        {formatted_history}

        User Profile:
        {json.dumps(personalization_context, indent=2)}

        User Question: {original_query}

        New Search Query:<end_of_turn>
        <start_of_turn>model
        """
        response = llm(prompt=prompt, max_tokens=100, stop=["<end_of_turn>"])
        personalized_query = response["choices"][0]["text"].strip()
        transformed_queries.append(personalized_query)

    all_queries = [original_query] + transformed_queries
    print(f"Generated {len(all_queries)} queries.")
    return {"transformed_queries": all_queries}

def retrieve_documents_node(state: AgentState) -> dict:
    """
    Node that performs hybrid retrieval and uses the cache.
    """
    print("--- Node: Retrieving Documents ---")
    queries = state.transformed_queries
    
    # Use the primary query for this retrieval attempt for cache lookup and storage
    current_retrieval_query = queries[0] 
    
    # Force a cache miss if it's a retry (iteration_count > 0)
    # This ensures that rewritten queries always trigger a fresh retrieval
    if state.iteration_count > 0:
        print(f"Forcing cache MISS for retry (iteration {state.iteration_count}). Performing full hybrid retrieval.")
        cached_results = None # Explicitly set to None to force cache miss
    else:
        cached_results = retrieval_cache.check_cache(current_retrieval_query)
    
    if cached_results:
        print("Cache HIT. Using cached retrieval results.")
        retrieved_ids = cached_results
    else:
        print("Cache MISS. Performing full hybrid retrieval.")
        retrieved_results_with_scores = hybrid_retriever.retrieve(queries)
        retrieved_ids = [result[0] for result in retrieved_results_with_scores]
        
        # Add the results to the cache using the primary query for this attempt
        retrieval_cache.add_to_cache(current_retrieval_query, retrieved_ids)
        
    retrieved_docs = [chunk for chunk in hybrid_retriever.all_chunks if chunk["metadata"]["chunk_id"] in retrieved_ids]
    print(f"Retrieved {len(retrieved_docs)} documents.")
    return {"retrieved_chunks": retrieved_docs}

def grade_retrieval_node(state: AgentState) -> dict:
    """
    Node that uses the LLM to grade the relevance of the retrieved documents.
    """
    print("--- Node: Grading Retrieval Relevance ---")
    # Limit context to top 3 chunks for grading to avoid exceeding context window
    context = "\n\n".join([chunk['content'] for chunk in state.retrieved_chunks[:3]])
    prompt = f"""<start_of_turn>user
    Given the user's question:
    '{state.initial_query}'

    And the following retrieved context:
    '{context}'

    Is the context relevant enough to answer the question? Answer with a simple 'yes' or 'no'.<end_of_turn>
    <start_of_turn>model
    """
    response = llm(prompt=prompt, max_tokens=10, stop=["<end_of_turn>"])
    grade = response["choices"][0]["text"].strip().lower()
    print(f"Relevance Grade: '{grade}'")
    return {"is_relevant": grade}

def rewrite_query_node(state: AgentState) -> dict:
    """
    Node that re-writes the query if the initial retrieval was not relevant.
    The rewritten query is then sent back to the transform_query_node.
    """
    print("--- Node: Rewriting Query ---")
    failed_queries = "\n- ".join(state.transformed_queries)
    prompt = f"""<start_of_turn>user
    You are a search expert. A user asked: '{state.initial_query}'.
    Our attempt to search with these related queries failed:
    - {failed_queries}
    Generate a single, new search query that takes a different approach to finding the answer. Do not repeat previous queries.<end_of_turn>
    <start_of_turn>model
    """
    response = llm(prompt=prompt, max_tokens=100, stop=["<end_of_turn>"])
    new_query = response["choices"][0]["text"].strip()
    print(f"Generated new query: '{new_query}'")
    return {"original_query": new_query, "iteration_count": state.iteration_count + 1}

def generate_response_node(state: AgentState) -> dict:
    """
    Node that generates the final answer using the LLM.
    """
    print("--- Node: Generating Final Answer ---")
    # Limit context to top 5 chunks and last 4 messages to manage context size
    context = "\n\n".join([chunk['content'] for chunk in state.retrieved_chunks[:5]])
    last_messages = state.chat_history[-4:]
    formatted_history = "\n".join([f"{msg.type}: {msg.content}" for msg in last_messages])
    personalization_context = state.personalization_context

    prompt = f"""<start_of_turn>user
    You are a helpful research assistant. Use the following context, chat history, and user profile to answer the user's question.

    Chat History:
    {formatted_history}

    User Profile:
    {json.dumps(personalization_context, indent=2)}

    Context:
    {context}

    Question: {state.original_query}
    Answer:<end_of_turn>
    <start_of_turn>model
    """
    response = llm(prompt=prompt, max_tokens=500, stop=["<end_of_turn>"])
    answer = response["choices"][0]["text"].strip()
    return {"final_answer": answer}

def generate_failure_response_node(state: AgentState) -> dict:
    """
    Node that generates a graceful failure message.
    """
    print("--- Node: Generating Graceful Failure Response ---")
    context = "\n\n".join([chunk['content'] for chunk in state.retrieved_chunks[:3]]) # Show top 3 chunks
    failure_message = (
        f"Unfortunately, I couldn't find a direct answer to your question: '{state.initial_query}'. "
        f"However, here is some potentially related information I found:\n\n- {context}"
    )
    return {"final_answer": failure_message}

# --- Conditional Edge Logic ---
def should_continue(state: AgentState) -> str:
    """
    Determines the next step based on the retrieval grade.
    """
    if state.is_relevant == "yes":
        return "generate_response"
    else:
        if state.iteration_count >= 1: # Allow one re-try
            return "handle_failure"
        else:
            return "rewrite_query"

# --- Main Agent Execution ---
def run_agent():
    # --- Define the Graph ---
    workflow = StateGraph(AgentState)

    workflow.add_node("personalization", personalization_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("grade_retrieval", grade_retrieval_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("generate_failure_response", generate_failure_response_node)

    workflow.set_entry_point("personalization")
    workflow.add_edge("personalization", "transform_query")
    workflow.add_edge("transform_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "grade_retrieval")
    workflow.add_conditional_edges(
        "grade_retrieval",
        should_continue,
        {
            "generate_response": "generate_response",
            "rewrite_query": "rewrite_query",
            "handle_failure": "generate_failure_response"
        }
    )
    workflow.add_edge("rewrite_query", "transform_query")
    workflow.add_edge("generate_response", END)
    workflow.add_edge("generate_failure_response", END)

    app = workflow.compile()

    # --- Interactive Chat Loop ---
    print("\n--- RAG Agent Ready ---")
    chat_history = []
    while True:
        query = input("Your query ( respond with exit or quit to quit): ")
        if query.lower() in ["exit", "quit"]:
            break
        
        inputs = {"initial_query": query, "original_query": query, "chat_history": chat_history}
        final_state = app.invoke(inputs)
        
        print(f"Agent: {final_state['final_answer']}")
        
        # Update chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=final_state['final_answer']))

if __name__ == "__main__":
    run_agent()