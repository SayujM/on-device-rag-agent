import os
import json
import sqlite3
from typing import List
from dotenv import load_dotenv
from googleapiclient.discovery import build

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
import uuid

# Local module imports
from query_transformer import QueryTransformer
from hybrid_retriever_v4 import HybridRetriever # Updated import
from retrieval_cache_v4 import RetrievalCache # Updated import
from embedding_model import load_embedding_model
from text_utils import tokenize_text
from profile_manager import ProfileManager
from pdf_processor import process_pdf
import pdf_manager # New import

load_dotenv()


# --- Agent State Definition ---
class AgentState(BaseModel):
    """
    Represents the core state of the RAG agent during a single run.
    """
    initial_query: str
    original_query: str
    query_route: str = ""
    chat_history: List[BaseMessage] = Field(default_factory=list)
    summarized_history: str = Field(default_factory=str) # New field for pre-computed summary
    current_turn_messages: List[BaseMessage] = Field(default_factory=list) # New field for current turn's messages
    current_conversation_summary: str = Field(default_factory=str) # New field for LLM-generated summary of current turn
    transformed_queries: List[str] = Field(default_factory=list)
    retrieved_chunks: List[dict] = Field(default_factory=list)
    is_relevant: str = ""
    final_answer: str = ""
    iteration_count: int = 0
    personalization_context: dict = Field(default_factory=dict)
    indexed_document_summary: str = "" # New field to store summary of indexed docs

# --- Initialize Components ---
print("--- Initializing RAG Agent Components ---")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GGUF_MODEL_PATH = os.path.join(BASE_DIR, "gemma-3n-E4B-it-Q4_K_M.gguf")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
PDF_NAME = "The-toyota-way-second-edition-chapter_1" # Default PDF
PDF_SPECIFIC_DIR = os.path.join(BASE_DIR, "pdf_files", "destination", PDF_NAME)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(BASE_DIR, "retrieval_cache")

query_transformer = QueryTransformer(model_path=GGUF_MODEL_PATH)
llm = query_transformer.llm # Reuse the loaded LLM for all nodes
embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
hybrid_retriever = HybridRetriever(
    db_dir=CHROMA_DB_DIR,
    # pdf_specific_dir=PDF_SPECIFIC_DIR,
    embedding_model_name=EMBEDDING_MODEL_NAME
)
retrieval_cache = RetrievalCache(cache_dir=CACHE_DIR, embedding_model=embedding_model, similarity_threshold=0.67)
print("-- All Components Initialized ---")
# --- Initialize Profile Manager and Checkpointer ---
PROFILE_DB_PATH = os.path.join(BASE_DIR, "user_profiles.db")
profile_manager = ProfileManager(db_path=PROFILE_DB_PATH)

def google_web_search(query: str, api_key: str, cse_id: str, **kwargs) -> dict:
    """Performs a web search using the Google Custom Search API."""
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()
    return res

# --- implement functions for the Agent ---
def load_user_profile_node(state: AgentState, config: dict) -> dict:
    """
    Loads the user's profile from the ProfileManager using the user_id from the config.
    If the user is new, creates a default profile in the state.
    """
    print("--- Node: Loading User Profile ---")
    user_id = config.get("configurable", {}).get("user_id", None)
    if not user_id:
        # This is a fallback, in a real app we'd enforce user_id presence.
        print("Warning: No user_id found in config. Proceeding with a default profile.")
        return {"personalization_context": {
            "name": "Default User",
            "preferences": {"tone": "formal"},
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

def add_initial_query_to_history_node(state: AgentState) -> dict:
    """
    Adds the initial user query to the chat history at the beginning of the turn.
    """
    print("--- Node: Adding Initial Query to History ---")
    updated_chat_history = state.chat_history + [HumanMessage(content=state.initial_query)]
    return {"chat_history": updated_chat_history}

def summarize_history_node(state: AgentState) -> dict:
    """
    Prepares a condensed, context-rich string from the chat history once per run,
    managing the context window by summarizing older messages.
    """
    print("--- Node: Summarizing History ---")
    chat_history = state.chat_history
    if not chat_history:
        return {"summarized_history": ""}

    if len(chat_history) <= 6:
        summary = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
        return {"summarized_history": summary}

    recent_messages = chat_history[-4:]
    older_messages = chat_history[:-4]

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
    full_summary = f"[Summary of earlier conversation: {summary}]\n\n[Recent messages:]\n{formatted_recent}"
    return {"summarized_history": full_summary}

def route_query_node(state: AgentState) -> dict:
    """
    Routes the user's query to either the RAG pipeline, a simple conversational response, or web search.
    """
    print("--- Node: Routing Query ---")
    # First, check for special commands like /ingest
    if state.initial_query.lower().startswith("/ingest "):
        pdf_path = state.initial_query[len("/ingest "):].strip()
        if os.path.exists(pdf_path):
            return {"query_route": "ingest_pdf", "original_query": pdf_path} # Pass path as original_query for ingest node
        else:
            return {"query_route": "conversational_query", "final_answer": f"Error: PDF file not found at {pdf_path}"} # Handle file not found

    # Determine if the query is relevant to the indexed documents
    relevance_prompt = f"""<start_of_turn>user
    Given the user's question and the summary of the documents this agent has access to, is the question likely to be answerable by these documents?
    Respond with 'yes' if it's highly relevant, 'no' if it's not relevant, or 'maybe' if it's partially relevant or requires external search.

    Document Summary: {state.indexed_document_summary}

    User Question: {state.initial_query}

    Answer (yes/no/maybe):<end_of_turn>
    <start_of_turn>model
    """
    relevance_response = llm(prompt=relevance_prompt, max_tokens=10, stop=["<end_of_turn>"])
    relevance_grade = relevance_response["choices"][0]["text"].strip().lower()
    print(f"Document Relevance Grade: '{relevance_grade}'")

    if "yes" in relevance_grade:
        return {"query_route": "rag_query"}
    elif "maybe" in relevance_grade:
        # For 'maybe', we can try RAG first, then fallback to web search if RAG fails
        return {"query_route": "rag_query"}
    else:
        # If not relevant to docs, check if it's a simple conversation or needs web search
        conversational_check_prompt = f"""<start_of_turn>user
        Given the user's question, is it a simple greeting, a question about the AI itself, or a general conversational query that does NOT require specific information retrieval from documents or the web?
        Respond with 'conversational' or 'needs_web_search'.

        User Question: {state.initial_query}

        Answer (conversational/needs_web_search):<end_of_turn>
        <start_of_turn>model
        """
        conversational_check_response = llm(prompt=conversational_check_prompt, max_tokens=20, stop=["<end_of_turn>"])
        conversational_type = conversational_check_response["choices"][0]["text"].strip().lower()
        print(f"Conversational Type: '{conversational_type}'")

        if "conversational" in conversational_type:
            return {"query_route": "conversational_query"}
        else:
            return {"query_route": "web_search_query"}

def generate_simple_response_node(state: AgentState) -> dict:
    """
    Generates a simple conversational response for non-RAG queries.
    """
    print("--- Node: Generating Simple Conversational Response ---")
    formatted_history = state.summarized_history

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

def transform_query_node(state: AgentState) -> dict:
    """
    Node that transforms the user's original query into multiple variations,
    first contextualizing it based on chat history and personalization.
    """
    print("--- Node: Transforming Query ---")
    original_query = state.original_query
    personalization_context = state.personalization_context
    formatted_history = state.summarized_history

    # Step 1: Contextualize the original query based on history and personalization
    contextualize_prompt = f"""<start_of_turn>user
    Given the following chat history, user profile, and the user's latest question, generate a standalone, context-aware search query.
    The rewritten query should be optimized for document retrieval and understandable without needing to refer back to the chat history.
    It should rephrase the original question for effective search, without attempting to answer it or include specific details that may not be present in the documents.

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
    print(f"Contextualized Query (from LLM): {contextualized_query}") # Added print statement

    # Step 2: Generate base transformed queries from the contextualized query
    transformed_queries = query_transformer.transform_query(contextualized_query)

    # The list of all queries to be used for retrieval
    all_queries = [contextualized_query] + transformed_queries
    print(f"Generated {len(all_queries)} queries.")
    
    # Add to current_turn_messages
    updated_current_turn_messages = state.current_turn_messages + [AIMessage(content="User query transformed into multiple search queries.")]

    return {"transformed_queries": all_queries, "current_turn_messages": updated_current_turn_messages}

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
        
    retrieved_docs = [chunk for chunk in hybrid_retriever.active_all_chunks if chunk["metadata"]["chunk_id"] in retrieved_ids]
    print(f"Retrieved {len(retrieved_docs)} documents.")

    # Add to current_turn_messages
    updated_current_turn_messages = state.current_turn_messages + [AIMessage(content=f"Retrieved {len(retrieved_docs)} documents.")]

    return {"retrieved_chunks": retrieved_docs, "current_turn_messages": updated_current_turn_messages}

def grade_retrieval_node(state: AgentState) -> dict:
    """
    Node that uses the LLM to grade the relevance of the retrieved documents.
    """
    print("--- Node: Grading Retrieval Relevance ---")
    # Limit context to top 3 chunks for grading to avoid exceeding LLM context window
    context = "\n\n".join([chunk['content'] for chunk in state.retrieved_chunks[:3]])
    print(f"Retrieved chunks for grading: {context}") # Added for debugging
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

    # Generate reason for the grade
    reason_prompt = f"""<start_of_turn>user
    Given the user's question: '{state.initial_query}'
    And the retrieved context: '{context}'
    And the grade: '{grade}'

    Briefly explain the reason for this grade (e.g., why the context is relevant or not relevant to the question). Focus on the content of the context.
    Reason:<end_of_turn>
    <start_of_turn>model
    """
    reason_response = llm(prompt=reason_prompt, max_tokens=100, stop=["<end_of_turn>"])
    reason = reason_response["choices"][0]["text"].strip()
    print(f"Grade Reason: '{reason}'")

    # Add to current_turn_messages
    updated_current_turn_messages = state.current_turn_messages + [AIMessage(content=f"Retrieval graded as: {grade}. Reason: {reason}")]

    return {"is_relevant": grade, "current_turn_messages": updated_current_turn_messages}

def summarize_current_turn_node(state: AgentState) -> dict:
    """
    Summarizes the messages collected during the current turn using an LLM.
    """
    print("--- Node: Summarizing Current Turn ---")
    if not state.current_turn_messages:
        return {"current_conversation_summary": ""}

    messages_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state.current_turn_messages])

    summary_prompt = f"""<start_of_turn>user
    Summarize the following sequence of events and observations from the current interaction turn. Focus on the key actions taken and their outcomes, especially regarding query transformation, document retrieval, and relevance grading.

    CURRENT TURN EVENTS:
    {messages_str}

    SUMMARY:<end_of_turn>
    <start_of_turn>model
    """
    summary_response = llm(prompt=summary_prompt, max_tokens=200, stop=["<end_of_turn>"])
    summary = summary_response["choices"][0]["text"].strip()

    return {"current_conversation_summary": summary}

def rewrite_query_node(state: AgentState) -> dict:
    """
    Node that re-writes the query if the initial retrieval was not relevant.
    The rewritten query is then sent back to the transform_query_node.
    """
    print("--- Node: Rewriting Query ---")
    failed_queries = "\n- ".join(state.transformed_queries)
    personalization_context = state.personalization_context
    formatted_history = state.summarized_history
    current_turn_summary = state.current_conversation_summary

    prompt = f"""<start_of_turn>user
    You are a highly skilled search strategist. Your task is to generate a single, new search query that takes a *fundamentally different approach* to finding information, given that previous search attempts for the user's question were unsuccessful.

    Here's the context:

    - Original User Question:
    {state.initial_query}

    - Previous Conversation History (for long-term context):
    {formatted_history}

    - User Profile (for personalization and preferences):
    {json.dumps(personalization_context, indent=2)}

    - Summary of Current Search Attempt (CRITICAL: This explains *why* the previous search was unsuccessful. Use this to inform your new search strategy.):
    {current_turn_summary}

    - Queries that have already been tried and failed to yield relevant results:
    {failed_queries}

    Based on all the information, especially the reason for the previous search failure, generate a new, semantically distinct search query. This query must:
    1. Be a single, standalone search query.
    2. Maintain the core intent of the original question.
    3. Avoid any queries listed in the "failed queries" section.
    4. Focus on exploring alternative keywords, synonyms, or rephrasing the question to uncover new relevant documents.
    5. The query should be designed purely for *information discovery*, not to include any part of a potential answer or assumed facts.

    New Search Query:<end_of_turn>
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
    formatted_history = state.summarized_history
    personalization_context = state.personalization_context
    current_turn_summary = state.current_conversation_summary

    prompt = f"""<start_of_turn>user
    You are an expert research assistant, tasked with providing accurate and comprehensive answers based on provided information. Your primary objective is to synthesize the given context to directly answer the user's question.

    Here is the information available to you:

    - User's Original Question:
    {state.original_query}

    - Relevant Context (CRITICAL: Base your answer primarily on this information):
    {context}

    - Summary of Current Search Attempt (Provides insight into how the context was obtained):
    {current_turn_summary}

    - Previous Conversation History (for conversational flow and continuity):
    {formatted_history}

    - User Profile (for tailoring the tone or style of the response, if applicable):
    {json.dumps(personalization_context, indent=2)}

    Carefully analyze the "Relevant Context" to formulate your answer. Use the "Summary of Current Search Attempt" to understand the retrieval process. Integrate insights from the "Previous Conversation History" and "User Profile" to ensure a natural and personalized response.

    Your answer should be:
    1. Directly responsive to the "User's Original Question."
    2. Grounded strictly in the "Relevant Context." Do not introduce outside information.
    3. Clear, concise, and well-structured.

    Answer:<end_of_turn>
    <start_of_turn>model
    """
    response = llm(prompt=prompt, max_tokens=500, stop=["<end_of_turn>"])
    answer = response["choices"][0]["text"].strip()
    return {"final_answer": answer}

def generate_failure_response_node(state: AgentState) -> dict:
    """
    Node that generates a graceful failure message using the LLM.
    """
    print("--- Node: Generating Graceful Failure Response ---")
    context = "\n\n".join([chunk['content'] for chunk in state.retrieved_chunks[:3]]) # Show top 3 chunks
    formatted_history = state.summarized_history
    personalization_context = state.personalization_context
    current_turn_summary = state.current_conversation_summary

    prompt = f"""<start_of_turn>user
    You are a helpful and transparent AI assistant. Your task is to generate a graceful and informative response when you cannot directly answer the user's question based on the available information.

    Here is the context for generating this failure message:

    - User's Original Question:
    {state.initial_query}

    - Previous Conversation History:
    {formatted_history}

    - User Profile:
    {json.dumps(personalization_context, indent=2)}

    - Summary of Current Search Attempt (Explains why a direct answer could not be found):
    {current_turn_summary}

    - Potentially Related Information Found (If any, provide as context for the user):
    {context}

    Based on the above, craft a response that:
    1. Clearly states that a direct answer could not be found.
    2. Briefly explains the difficulty or the outcome of the search attempt, referencing the "Summary of Current Search Attempt."
    3. If "Potentially Related Information Found" is available, offer it as additional context.
    4. Maintains a polite, helpful, and apologetic tone.

    Failure Response:<end_of_turn>
    <start_of_turn>model
    """
    response = llm(prompt=prompt, max_tokens=500, stop=["<end_of_turn>"])
    failure_message = response["choices"][0]["text"].strip()
    return {"final_answer": failure_message}

def web_search_node(state: AgentState) -> dict:
    """
    Performs a web search using the initial query and summarizes the results.
    """
    print("--- Node: Performing Web Search ---")
    search_query = state.original_query
    print(f"Searching the web for: {search_query}")
    
    # Get API key and CSE ID from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("SEARCH_ENGINE_ID")

    if not api_key or not cse_id:
        print("Error: GOOGLE_API_KEY and SEARCH_ENGINE_ID must be set in the environment.")
        return {"final_answer": "Web search is not configured. Missing API key or search engine ID."}

    try:
        search_results = google_web_search(search_query, api_key, cse_id, num=3)
        
        if "items" in search_results:
            snippets = [f"Title: {item.get('title')}\nLink: {item.get('link')}\nSnippet: {item.get('snippet')}" for item in search_results["items"]]
            combined_snippets = "\n\n".join(snippets)
            
            if not combined_snippets:
                return {"final_answer": f"I couldn't find relevant information for '{search_query}' on the web."}

            summary_prompt = f"""<start_of_turn>user
            Summarize the following web search results to answer the user's question: '{state.initial_query}'.
            Focus on providing a concise answer based on the snippets. If the snippets do not directly answer the question, state that.

            Web Search Results:
            {combined_snippets}

            Summary:<end_of_turn>
            <start_of_turn>model
            """
            summary_response = llm(prompt=summary_prompt, max_tokens=500, stop=["<end_of_turn>"])
            answer = summary_response["choices"][0]["text"].strip()
            return {"final_answer": f"Here's what I found on the web regarding '{state.initial_query}':\n\n{answer}"}
        else:
            return {"final_answer": f"I couldn't find relevant information for '{search_query}' on the web."}
    except Exception as e:
        print(f"Error during web search: {e}")
        return {"final_answer": f"An error occurred while trying to search the web for '{search_query}'."}


def ingest_pdf_node(state: AgentState) -> dict:
    """
    Processes a new PDF, extracts text, chunks, creates embeddings, and updates indices.
    """
    print(f"--- Node: Ingesting PDF: {state.original_query} ---")
    pdf_path = state.original_query # The PDF path is passed via original_query for this special route

    pdf_name_without_ext = os.path.splitext(os.path.basename(pdf_path))[0]

    # Process and index the selected PDF
    success, message = pdf_manager.process_and_index_pdf(pdf_path)
    print(message)
    if not success:
        return {"final_answer": message}

    # Load the processed PDF data into the retriever
    hybrid_retriever.load_pdf_for_retrieval(pdf_name_without_ext)

    # Re-initialize RetrievalCache (v4) for the newly loaded PDF
    paths = pdf_manager.get_pdf_output_dirs(pdf_name_without_ext)
    # We need to pass the embedding_model to the RetrievalCache constructor
    # This requires making embedding_model accessible in the node, perhaps via a global or by passing it in config
    # For now, assuming embedding_model is globally accessible (as it is in run_agent scope)
    global embedding_model # Access the global embedding_model
    global retrieval_cache # Access the global retrieval_cache instance to update it
    retrieval_cache = RetrievalCache(cache_dir=paths["retrieval_cache_dir"], embedding_model=embedding_model)

    # Update indexed_document_summary
    # updated_indexed_document_summary = f"Documents now include content from {os.path.basename(pdf_path)}. Topics include [add general topics here]."
    # updated_indexed_document_summary = (
    # f"This agent currently has access to documents from '{os.path.basename(pdf_path)}'. "
    # "Topics included: The Toyota Way philosophy, operational excellence, "
    # "annual net income/loss of major automakers (Toyota, Ford, Volkswagen, General Motors, Honda) from 2004-2018, "
    # "vehicle quality and long-term reliability ratings (J.D. Power, Dashboard Light) for various car brands "
    # "(Toyota, Lexus, Scion, Porsche, Mercedes-Benz, Infiniti, Honda, Saab, Chevrolet, GMC, Jeep, Pontiac, Buick, "
    # "Mitsubishi, Cadillac, Acura, Mercury, Hyundai, Lincoln, BMW, Dodge, Audi, Land Rover, Subaru, Volvo, Saturn, "
    # "Jaguar, Nissan, Mazda, Ford, Kia, Chrysler, Volkswagen), "
    # "the Toyota Production System (TPS), and scientific thinking as applied to problem-solving and continuous "
    # "improvement within an organization."
    # )
    updated_indexed_document_summary = (
    f"This agent currently has access to documents from '{os.path.basename(pdf_path)}'. "
    "Topics included: Electric charges and their properties, conductors and insulators, "
    "charging by induction, Coulomb's Law, the principle of superposition for forces between multiple charges, "
    "electric field, electric field lines, electric dipole, electric field due to a point charge, "
    "electric field due to an electric dipole (on axial and equatorial lines), "
    "torque on an electric dipole in a uniform electric field, "
    "continuous charge distribution (linear, surface, and volume charge density), "
    "Gauss's Law and its applications (electric field due to an infinitely long straight uniformly charged wire, "
    "uniformly charged infinite plane sheet, and uniformly charged thin spherical shell)."
    )

    return {"final_answer": f"Successfully ingested {os.path.basename(pdf_path)} and updated indices.",
            "indexed_document_summary": updated_indexed_document_summary}

def update_history_and_profile_node(state: AgentState, config: dict) -> dict:
    """
    Updates the chat history and summarizes the interaction to update the user's profile.
    Saves the updated profile back to the database.
    """
    print("--- Node: Updating History and Profile ---")
    user_id = config.get("configurable", {}).get("user_id")
    
    # 1. Update chat history in the state
    # The checkpointer handles loading the history, we just append the latest turn
    # The initial query is now added by add_initial_query_to_history_node
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
        "iteration_count": 0,
        "current_turn_messages": [], # Reset current turn messages
        "current_conversation_summary": "" # Reset current conversation summary
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

def should_route_from_failure(state: AgentState) -> str:
    """
    Determines if, after a RAG failure, we should try web search or give up.
    """
    # For now, always try web search after RAG failure
    return "web_search_query"

# --- Main Agent Execution ---
def run_agent():
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "langgraph_checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    CHECKPOINT_DB_PATH = os.path.join(CHECKPOINT_DIR, "checkpoints.sqlite")

    # Pass initialized retriever and embedding model (from global scope) to pdf_manager
    pdf_manager.set_global_retriever_and_embedding_model(hybrid_retriever, embedding_model)

    # --- PDF Selection and Processing ---
    available_pdfs = [f for f in os.listdir(pdf_manager.PDF_SOURCE_DIR) if f.lower().endswith(".pdf")]
    if not available_pdfs:
        print(f"No PDF files found in {pdf_manager.PDF_SOURCE_DIR}. Please add some PDFs to proceed.")
        return

    print("Available PDFs for ingestion:")
    for i, pdf_name in enumerate(available_pdfs):
        print(f"{i+1}. {pdf_name}")

    selected_pdf_name = None
    while selected_pdf_name is None:
        try:
            choice = input("Enter the number of the PDF to use (or 'exit' to quit): ")
            if choice.lower() == 'exit' or choice.lower() == 'quit':
                return
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_pdfs):
                selected_pdf_name = available_pdfs[choice_idx]
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    selected_pdf_path = os.path.join(pdf_manager.PDF_SOURCE_DIR, selected_pdf_name)
    pdf_name_without_ext = os.path.splitext(selected_pdf_name)[0]

    # Process and index the selected PDF
    success, message = pdf_manager.process_and_index_pdf(selected_pdf_path)
    print(message)
    if not success:
        print("Exiting due to PDF processing failure.")
        return

    # Load the processed PDF data into the retriever
    hybrid_retriever.load_pdf_for_retrieval(pdf_name_without_ext)

    # Initialize RetrievalCache (v4) for the selected PDF
    paths = pdf_manager.get_pdf_output_dirs(pdf_name_without_ext)
    retrieval_cache = RetrievalCache(cache_dir=paths["retrieval_cache_dir"], embedding_model=embedding_model)

    print("--- All Components Initialized and PDF Loaded ---")

    # --- Initialize Profile Manager and Checkpointer ---
    PROFILE_DB_PATH = os.path.join(BASE_DIR, "user_profiles.db")
    profile_manager = ProfileManager(db_path=PROFILE_DB_PATH)

    # Initial indexed document summary (can be updated by ingest_pdf_node)
    # initial_indexed_doc_summary = f"This agent currently has access to documents from '{selected_pdf_name}'. Topics include [add general topics here]." # Placeholder
    # initial_indexed_doc_summary = (
    # f"This agent currently has access to documents from '{selected_pdf_name}'. "
    # "Topics included: The Toyota Way philosophy, operational excellence, "
    # "the Toyota Production System (TPS), and scientific thinking as applied to problem-solving and continuous "
    # "improvement within an organization."
    # "annual net income/loss of major automakers (Toyota, Ford, Volkswagen, General Motors, Honda) from 2004-2018, "
    # "vehicle quality and long-term reliability ratings (J.D. Power, Dashboard Light) for various car brands "
    # "(Toyota, Lexus, Scion, Porsche, Mercedes-Benz, Infiniti, Honda, Saab, Chevrolet, GMC, Jeep, Pontiac, Buick, "
    # "Mitsubishi, Cadillac, Acura, Mercury, Hyundai, Lincoln, BMW, Dodge, Audi, Land Rover, Subaru, Volvo, Saturn, "
    # "Jaguar, Nissan, Mazda, Ford, Kia, Chrysler, Volkswagen), "
    # )
    initial_indexed_doc_summary = (
    f"This agent currently has access to documents from '{selected_pdf_name}'. "
    "Topics included: Electric charges and their properties, conductors and insulators, "
    "charging by induction, Coulomb's Law, the principle of superposition for forces between multiple charges, "
    "electric field, electric field lines, electric dipole, electric field due to a point charge, "
    "electric field due to an electric dipole (on axial and equatorial lines), "
    "torque on an electric dipole in a uniform electric field, "
    "continuous charge distribution (linear, surface, and volume charge density), "
    "Gauss's Law and its applications (electric field due to an infinitely long straight uniformly charged wire, "
    "uniformly charged infinite plane sheet, and uniformly charged thin spherical shell)."
    )
    # --- Define the Graph ---
    with SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as memory:
        workflow = StateGraph(AgentState)

        workflow.add_node("load_user_profile", load_user_profile_node)
        workflow.add_node("add_initial_query_to_history", add_initial_query_to_history_node)
        workflow.add_node("summarize_history", summarize_history_node)
        workflow.add_node("route_query", route_query_node)
        workflow.add_node("transform_query", transform_query_node)
        workflow.add_node("retrieve_documents", retrieve_documents_node)
        workflow.add_node("grade_retrieval", grade_retrieval_node)
        workflow.add_node("summarize_current_turn", summarize_current_turn_node)
        workflow.add_node("rewrite_query", rewrite_query_node)
        workflow.add_node("generate_response", generate_response_node)
        workflow.add_node("generate_simple_response", generate_simple_response_node)
        workflow.add_node("generate_failure_response", generate_failure_response_node)
        workflow.add_node("web_search", web_search_node) # New node
        workflow.add_node("ingest_pdf", ingest_pdf_node) # New node
        workflow.add_node("update_history_and_profile", update_history_and_profile_node)
        workflow.add_node("cleanup_state", cleanup_state_node)

        workflow.set_entry_point("load_user_profile")
        workflow.add_edge("load_user_profile", "add_initial_query_to_history")
        workflow.add_edge("add_initial_query_to_history", "summarize_history")
        workflow.add_edge("summarize_history", "route_query")

        workflow.add_conditional_edges(
            "route_query",
            lambda state: state.query_route,
            {
                "rag_query": "transform_query",
                "conversational_query": "generate_simple_response",
                "web_search_query": "web_search", # New route
                "ingest_pdf": "ingest_pdf", # New route
            },
        )

        workflow.add_edge("transform_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "grade_retrieval")
        workflow.add_edge("grade_retrieval", "summarize_current_turn")

        workflow.add_conditional_edges(
            "summarize_current_turn",
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
        workflow.add_edge("web_search", "update_history_and_profile") # New edge
        workflow.add_edge("ingest_pdf", "update_history_and_profile") # New edge

        # After RAG failure, route to web search
        workflow.add_conditional_edges(
            "generate_failure_response",
            should_route_from_failure,
            {
                "web_search_query": "web_search",
                # Add other failure handling routes here if needed
            }
        )

        workflow.add_edge("update_history_and_profile", "cleanup_state")
        workflow.add_edge("cleanup_state", END)

        app = workflow.compile(checkpointer=memory)
        # --- Interactive Chat Loop ---
        print("\n--- RAG Agent Ready ---")
        user_id = input("Enter your User ID (e.g., 'user123'): ")
        thread_id = input("Enter a Conversation ID (e.g., 'chat_session_1' or leave blank for new): ")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            print(f"Generated new conversation ID: {thread_id}")

        # Initialize indexed_document_summary for the first run
        # In a real app, this would be loaded from persistent storage
        # initial_indexed_doc_summary = "This agent currently has access to documents related to 'The Toyota Way', lean manufacturing, and operational excellence. These documents primarily cover the principles, history, and application of the Toyota Production System." # Hardcoded for now

        while True:
            query = input("Your query (respond with exit or quit to quit): ")
            if query.lower() in ["exit", "quit"]:
                break
            
            # The initial state for the graph run
            inputs = {"initial_query": query, "original_query": query, "indexed_document_summary": initial_indexed_doc_summary}
            
            # Pass user_id and thread_id via configurable for checkpointer and profile loading
            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

            # Invoke the graph. LangGraph will load the state for this thread_id automatically.
            final_state = app.invoke(inputs, config=config)
            
            print(f"Agent: {final_state['final_answer']}")

            # Update initial_indexed_doc_summary if it was changed by ingestion
            if final_state.get("indexed_document_summary"):
                initial_indexed_doc_summary = final_state["indexed_document_summary"]

    # Close the profile manager connection when done
    profile_manager.close()

if __name__ == "__main__":
    run_agent()
