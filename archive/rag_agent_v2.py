import os
import json
import sqlite3
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
import uuid

# Local module imports
from query_transformer import QueryTransformer
from hybrid_retriever import HybridRetriever
from retrieval_cache import RetrievalCache
from embedding_model import load_embedding_model
from text_utils import tokenize_text
from profile_manager import ProfileManager

# --- Agent State Definition ---
class AgentState(BaseModel):
    """
    Represents the core state of the RAG agent during a single run.
    """
    initial_query: str
    original_query: str
    query_route: str = ""
    chat_history: List[BaseMessage] = Field(default_factory=list) # Re-added chat_history
    transformed_queries: List[str] = Field(default_factory=list)
    retrieved_chunks: List[dict] = Field(default_factory=list)
    is_relevant: str = ""
    final_answer: str = ""
    iteration_count: int = 0
    personalization_context: dict = Field(default_factory=dict)

def prepare_contextual_history(history: List[BaseMessage]) -> str:
    """
    Prepares a condensed, context-rich string from the chat history,
    managing the context window by summarizing older messages.
    """
    if not history:
        return ""

    # If history is short, just format the last few messages
    if len(history) <= 6:
        return "\n".join([f"{msg.type}: {msg.content}" for msg in history])

    # If history is long, summarize older parts
    recent_messages = history[-4:]
    older_messages = history[:-4]

    summary_prompt = f"""<start_of_turn>user
    Briefly summarize the following conversation between a user and an AI assistant.

    CONVERSATION:
    { "\n".join([f'{msg.type}: {msg.content}' for msg in older_messages]) }

    SUMMARY:<end_of_turn>
    <start_of_turn>model
    """
    summary_response = llm(prompt=summary_prompt, max_tokens=200, stop=["<end_of_turn>"])
    summary = summary_response["choices"][0]["text"].strip()

    formatted_recent = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_messages])

    return f"[Summary of earlier conversation: {summary}]\n\n[Recent messages:]\n{formatted_recent}"

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
# --- Initialize Profile Manager and Checkpointer ---
PROFILE_DB_PATH = os.path.join(BASE_DIR, "user_profiles.db")
profile_manager = ProfileManager(db_path=PROFILE_DB_PATH)


def load_user_profile_node(state: AgentState, config: dict) -> dict:
    """
    Loads the user's profile from the ProfileManager using the user_id from the config.
    If the user is new, creates a default profile in the state.
    """
    print("--- Node: Loading User Profile ---")
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        # This is a fallback, in a real app we'd enforce user_id presence.
        print("Warning: No user_id found in config. Proceeding with a default profile.")
        return {"personalization_context": {
            "name": "Default User",
            "preferences": {},
            "past_interactions_summary": ""
        }}

    profile = profile_manager.get_profile(user_id)
    if profile:
        print(f"Loaded profile for user: {user_id}")
        return {"personalization_context": profile}
    else:
        print(f"No profile found for new user: {user_id}. Creating default.")
        default_profile = {
            "name": user_id,
            "preferences": {"tone": "formal"},
            "past_interactions_summary": ""
        }
        return {"personalization_context": default_profile}

def route_query_node(state: AgentState) -> dict:
    """
    Routes the user's query to either the RAG pipeline or a simple conversational response.
    """
    print("--- Node: Routing Query ---")
    prompt = f"""<start_of_turn>user
    Given the user's query, classify it as either 'rag_query' or 'conversational_query'.
    - A 'rag_query' seeks specific information that must be looked up in documents.
    - A 'conversational_query' is a simple greeting, a question about the AI itself, or a follow-up that does not require new information retrieval.
    Respond with only the classification.

    User Query: {state.initial_query}<end_of_turn>
    <start_of_turn>model
    """
    response = llm(prompt=prompt, max_tokens=15, stop=["<end_of_turn>"])
    route = response["choices"][0]["text"].strip().lower()
    print(f"Route determined: '{route}'")
    if "rag_query" in route:
        return {"query_route": "rag_query"}
    else:
        return {"query_route": "conversational_query"}

def generate_simple_response_node(state: AgentState) -> dict:
    """
    Generates a simple conversational response for non-RAG queries.
    """
    print("--- Node: Generating Simple Conversational Response ---")
    # The checkpointer loads the history into the state. We access it here.
    chat_history = state.chat_history
    formatted_history = prepare_contextual_history(chat_history)

    prompt = f"""<start_of_turn>user
    You are a helpful AI assistant. Use the chat history to provide a friendly, conversational response to the user's message.

    Chat History:
    {formatted_history}

    User Message: {state.initial_query}
    
    Your Response:<end_of_turn>
    <start_of_turn>model
    """
    response = llm(prompt=prompt, max_tokens=200, stop=["<end_of_turn>"])
    answer = response["choices"][0]["text"].strip()
    return {"final_answer": answer}

# --- Node Definitions ---
def transform_query_node(state: AgentState) -> dict:
    """
    Node that transforms the user's original query into multiple variations,
    first contextualizing it based on chat history and personalization.
    """
    print("--- Node: Transforming Query ---")
    original_query = state.original_query
    chat_history = state.chat_history
    personalization_context = state.personalization_context

    formatted_history = prepare_contextual_history(chat_history)

    # Step 1: Contextualize the original query based on history and personalization
    contextualize_prompt = f"""<start_of_turn>user
    Given the following chat history, user profile, and the user's latest question, rewrite the latest question into a standalone, context-aware search query.
    The rewritten query should be understandable without needing to refer back to the chat history.

    Chat History:
    {formatted_history}

    User Profile:
    {json.dumps(personalization_context, indent=2)}

    User's Latest Question: {original_query}

    Standalone Search Query:<end_of_turn>
    <start_of_turn>model
    """
    contextualized_query_response = llm(prompt=contextualize_prompt, max_tokens=100, stop=["<end_of_turn>"])
    contextualized_query = contextualized_query_response["choices"][0]["text"].strip()
    print(f"Contextualized Query: {contextualized_query}")

    # Step 2: Generate base transformed queries from the contextualized query
    transformed_queries = query_transformer.transform_query(contextualized_query)

    # Step 3: Generate an additional, personalized query from the contextualized query
    # This step is now always performed if there's any context (history or personalization)
    if chat_history or personalization_context:
        personalized_query_prompt = f"""<start_of_turn>user
        Given the following chat history, user profile, and the contextualized search query, generate an additional, semantically different search query that might yield more relevant results, considering the user's preferences.

        Chat History:
        {formatted_history}

        User Profile:
        {json.dumps(personalization_context, indent=2)}

        Contextualized Search Query: {contextualized_query}

        Additional Search Query:<end_of_turn>
        <start_of_turn>model
        """
        personalized_query_response = llm(prompt=personalized_query_prompt, max_tokens=100, stop=["<end_of_turn>"])
        personalized_query = personalized_query_response["choices"][0]["text"].strip()
        transformed_queries.append(personalized_query)

    # The list of all queries to be used for retrieval
    all_queries = [contextualized_query] + transformed_queries
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
    chat_history = state.chat_history
    personalization_context = state.personalization_context

    formatted_history = prepare_contextual_history(chat_history)

    prompt = f"""<start_of_turn>user
    You are a search expert. The user's original question was: '{state.initial_query}'.
    The current conversation history is:
    {formatted_history}

    User Profile:
    {json.dumps(personalization_context, indent=2)}

    Our previous attempts to search with these related queries failed:
    - {failed_queries}

    Considering the entire conversation and the user's profile, generate a single, new search query that takes a different approach to finding the answer.
    Crucially, ensure the new query maintains the original conversational context and does not introduce unrelated topics.
    Do not repeat previous queries.<end_of_turn>
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
    chat_history = state.chat_history # Get chat history from state
    formatted_history = prepare_contextual_history(chat_history)
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

def update_history_and_profile_node(state: AgentState, config: dict) -> dict:
    """
    Updates the chat history and summarizes the interaction to update the user's profile.
    Saves the updated profile back to the database.
    """
    print("--- Node: Updating History and Profile ---")
    user_id = config.get("configurable", {}).get("user_id")
    
    # 1. Update chat history in the state
    # The checkpointer handles loading the history, we just append the latest turn
    state.chat_history.append(HumanMessage(content=state.initial_query))
    state.chat_history.append(AIMessage(content=state.final_answer))

    if not user_id:
        print("Warning: No user_id found. Cannot save profile.")
        return {"chat_history": state.chat_history}

    # 2. Update personalization context with LLM summarization
    personalization_context = state.personalization_context
    
    prompt = f"""<start_of_turn>user
    You are a profile manager. Given the user's existing profile, their latest question, and the answer they received, update the profile's 'past_interactions_summary'.
    Condense the new information into the existing summary. If no new, meaningful preference or topic is revealed, you can make minimal or no changes.
    Keep the summary concise.

    Existing Profile:
    {json.dumps(personalization_context, indent=2)}

    User's Latest Question:
    {state.initial_query}

    Agent's Answer:
    {state.final_answer}

    Respond with ONLY the updated JSON for the user's profile. Do not add any other text.<end_of_turn>
    <start_of_turn>model
    """
    
    try:
        response = llm(prompt=prompt, max_tokens=500, stop=["<end_of_turn>"])
        updated_profile_json = response["choices"][0]["text"].strip()
        updated_profile = json.loads(updated_profile_json)
        print(f"Updated profile summary for user: {user_id}")
        personalization_context = updated_profile
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Warning: Could not parse LLM response for profile update. Using old profile. Error: {e}")
        personalization_context = state.personalization_context

    # 3. Save the updated profile to the database
    profile_manager.save_profile(user_id, personalization_context)

    # 4. Return the updated history and context for the checkpointer
    return {
        "chat_history": state.chat_history,
        "personalization_context": personalization_context
    }

def cleanup_state_node(state: AgentState) -> dict:
    """
    Cleans up the state by removing transient data before final checkpointing.
    """
    print("--- Node: Cleaning Up State ---")
    return {
        "retrieved_chunks": [],
        "transformed_queries": [],
        "is_relevant": "",
        "iteration_count": 0
    }

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

    CHECKPOINT_DIR = os.path.join(BASE_DIR, "langgraph_checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    CHECKPOINT_DB_PATH = os.path.join(CHECKPOINT_DIR, "checkpoints.sqlite")
    # memory = SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH)
    # sqlite_connection = sqlite3.connect(CHECKPOINT_DB_PATH)
    # memory = SqliteSaver(conn=sqlite_connection)

    # --- Define the Graph ---
    with SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as memory:
        workflow = StateGraph(AgentState)

        workflow.add_node("load_user_profile", load_user_profile_node)
        workflow.add_node("route_query", route_query_node)
        workflow.add_node("transform_query", transform_query_node)
        workflow.add_node("retrieve_documents", retrieve_documents_node)
        workflow.add_node("grade_retrieval", grade_retrieval_node)
        workflow.add_node("rewrite_query", rewrite_query_node)
        workflow.add_node("generate_response", generate_response_node)
        workflow.add_node("generate_simple_response", generate_simple_response_node)
        workflow.add_node("generate_failure_response", generate_failure_response_node)
        workflow.add_node("update_history_and_profile", update_history_and_profile_node)
        workflow.add_node("cleanup_state", cleanup_state_node)

        workflow.set_entry_point("load_user_profile")
        workflow.add_edge("load_user_profile", "route_query")

        workflow.add_conditional_edges(
            "route_query",
            lambda state: state.query_route,
            {
                "rag_query": "transform_query",
                "conversational_query": "generate_simple_response",
            },
        )

        workflow.add_edge("transform_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "grade_retrieval")
        workflow.add_conditional_edges(
            "grade_retrieval",
            should_continue,
            {
                "generate_response": "generate_response",
                "rewrite_query": "rewrite_query",
                "handle_failure": "generate_failure_response",
            },
        )
        workflow.add_edge("rewrite_query", "transform_query")

        # All paths lead to updating history and profile, then cleanup, then END
        workflow.add_edge("generate_response", "update_history_and_profile")
        workflow.add_edge("generate_simple_response", "update_history_and_profile")
        workflow.add_edge("generate_failure_response", "update_history_and_profile")
        workflow.add_edge("update_history_and_profile", "cleanup_state")
        workflow.add_edge("cleanup_state", END)

        # with SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as memory:
        #     app = workflow.compile(checkpointer=memory)
        app = workflow.compile(checkpointer=memory)
        # --- Interactive Chat Loop ---
        print("\n--- RAG Agent Ready ---")
        user_id = input("Enter your User ID (e.g., 'user123'): ")
        thread_id = input("Enter a Conversation ID (e.g., 'chat_session_1' or leave blank for new): ")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            print(f"Generated new conversation ID: {thread_id}")

        while True:
            query = input("Your query (respond with exit or quit to quit): ")
            if query.lower() in ["exit", "quit"]:
                break
            
            # The initial state for the graph run
            inputs = {"initial_query": query, "original_query": query}
            
            # Pass user_id and thread_id via configurable for checkpointer and profile loading
            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

            # Invoke the graph. LangGraph will load the state for this thread_id automatically.
            final_state = app.invoke(inputs, config=config)
            
            print(f"Agent: {final_state['final_answer']}")

    # Close the profile manager connection when done
    profile_manager.close()
    # sqlite_connection.close()

if __name__ == "__main__":
    run_agent()
