import os
import sys
import json
import uuid
from typing import List, Tuple
import re
import html
import base64
import gradio as gr
from dotenv import load_dotenv
from googleapiclient.discovery import build
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

# Local module imports
from query_transformer import QueryTransformer
from hybrid_retriever_v4 import HybridRetriever
from retrieval_cache_v4 import RetrievalCache
from embedding_model import load_embedding_model
from profile_manager import ProfileManager
from web_searcher import hybrid_web_search
import pdf_manager
import config

# Load environment variables from .env file
load_dotenv()

# --- Dynamic Configuration & Summaries ---
BASE_DIR = config.BASE_DIR
CHECKPOINT_DIR = config.CHECKPOINT_DIR
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_DB_PATH = config.CHECKPOINT_DB_PATH

def get_pdf_summaries() -> dict:
    """Dynamically loads PDF summaries from the JSON storage."""
    if os.path.exists(config.PDF_SUMMARIES_PATH):
        try:
            with open(config.PDF_SUMMARIES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read {config.PDF_SUMMARIES_PATH}: {e}")
    return {}

# --- Agent State Definition ---
class AgentState(BaseModel):
    initial_query: str
    original_query: str
    query_route: str = ""
    chat_history: List[BaseMessage] = Field(default_factory=list)
    summarized_history: str = Field(default_factory=str)
    current_turn_messages: List[BaseMessage] = Field(default_factory=list)
    current_conversation_summary: str = Field(default_factory=str)
    transformed_queries: List[str] = Field(default_factory=list)
    retrieved_chunks: List[dict] = Field(default_factory=list)
    is_relevant: str = ""
    final_answer: str = ""
    iteration_count: int = 0
    personalization_context: dict = Field(default_factory=dict)
    indexed_document_summary: str = ""


# --- RAG Agent Class ---
class RAGAgent:
    def __init__(self, selected_pdf_path: str, memory: SqliteSaver):
        print("--- Initializing RAG Agent ---")
        self.base_dir = config.BASE_DIR
        gguf_model_path = config.MODEL_PATH
        chroma_db_dir = config.CHROMA_DB_DIR
        embedding_model_name = config.EMBEDDING_MODEL_NAME

        print("Loading Query Transformer and Embedding Model...")
        self.query_transformer = QueryTransformer(model_path=gguf_model_path, n_gpu_layers=-1, n_ctx=4096)
        self.llm = self.query_transformer.llm
        self.embedding_model = load_embedding_model(embedding_model_name)

        print("Initializing Hybrid Retriever...")
        self.hybrid_retriever = HybridRetriever(
            db_dir=chroma_db_dir,
            embedding_model_name=embedding_model_name
        )

        print(f"Processing selected PDF: {os.path.basename(selected_pdf_path)}")
        pdf_manager.set_global_retriever_and_embedding_model(self.hybrid_retriever, self.embedding_model)
        success, message = pdf_manager.process_and_index_pdf(selected_pdf_path)
        if not success:
            raise RuntimeError(f"Failed to process PDF: {message}")
        print(message)

        print("Initializing Retrieval Cache...")
        pdf_name_without_ext = os.path.splitext(os.path.basename(selected_pdf_path))[0]
        self.hybrid_retriever.load_pdf_for_retrieval(pdf_name_without_ext)

        paths = pdf_manager.get_pdf_output_dirs(pdf_name_without_ext)
        self.retrieval_cache = RetrievalCache(
            cache_dir=paths["retrieval_cache_dir"],
            embedding_model=self.embedding_model
        )

        print("Initializing Profile Manager...")
        profile_db_path = config.PROFILE_DB_PATH
        self.profile_manager = ProfileManager(db_path=profile_db_path)

        pdf_filename = os.path.basename(selected_pdf_path)
        summary = get_pdf_summaries().get(pdf_filename, "No specific summary available.")
        self.initial_indexed_doc_summary = f"Accessing docs from '{pdf_filename}'. {summary}"

        print("Building and compiling the agent graph...")
        self.memory = memory
        self.app = self._build_graph()
        print("--- RAG Agent Initialized Successfully ---")

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        # Streamlined Node Definitions
        workflow.add_node("load_user_profile", self.load_user_profile_node)
        workflow.add_node("manage_chat_history", self.manage_chat_history_node)
        workflow.add_node("route_query", self.route_query_node)
        workflow.add_node("transform_query", self.transform_query_node)
        workflow.add_node("retrieve_documents", self.retrieve_documents_node)
        workflow.add_node("grade_and_answer", self.grade_and_answer_node)
        workflow.add_node("rewrite_query", self.rewrite_query_node)
        workflow.add_node("generate_simple_response", self.generate_simple_response_node)
        workflow.add_node("generate_failure_response", self.generate_failure_response_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("ingest_pdf", self.ingest_pdf_node)
        workflow.add_node("update_history_and_profile", self.update_history_and_profile_node)
        workflow.add_node("cleanup_state", self.cleanup_state_node)

        # Streamlined Entry & Routing Edges
        workflow.set_entry_point("load_user_profile")
        workflow.add_edge("load_user_profile", "manage_chat_history")
        workflow.add_edge("manage_chat_history", "route_query")
        workflow.add_conditional_edges(
            "route_query", 
            lambda state: state.query_route,
            {"rag_query": "transform_query", 
             "conversational_query": "generate_simple_response", 
             "web_search_query": "web_search", 
             "ingest_pdf": "ingest_pdf"}
        )
        workflow.add_edge("transform_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "grade_and_answer")

        workflow.add_conditional_edges(
            "grade_and_answer",
            self.should_continue,
            {
                "end_with_answer": "update_history_and_profile",
                "rewrite_query": "rewrite_query",
                "handle_failure": "generate_failure_response",
            },
        )
        workflow.add_edge("rewrite_query", "transform_query")

        workflow.add_conditional_edges(
            "generate_failure_response", 
            self.should_route_from_failure, 
            {"web_search_query": "web_search"}
        )

        # Define terminal edges leading to the end of the graph
        workflow.add_edge("generate_simple_response", "update_history_and_profile")
        workflow.add_edge("web_search", "update_history_and_profile")
        workflow.add_edge("ingest_pdf", "update_history_and_profile")
        workflow.add_edge("update_history_and_profile", "cleanup_state")
        workflow.add_edge("cleanup_state", END)

        return workflow.compile(checkpointer=self.memory)

    def invoke(self, query: str, user_id: str, thread_id: str) -> dict:
        inputs = {
            "initial_query": query,
            "original_query": query,
            "indexed_document_summary": self.initial_indexed_doc_summary
        }
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        final_state = self.app.invoke(inputs, config=config)
        if final_state.get("indexed_document_summary"):
            self.initial_indexed_doc_summary = final_state["indexed_document_summary"]
        return final_state

    def close(self):
        print("--- Closing RAG Agent Connections ---")
        self.profile_manager.close()

    # --- Function definition for each Node ---
    def load_user_profile_node(self, state: AgentState, config: dict) -> dict:
        print("--- Node: Loading User Profile ---")
        user_id = config.get("configurable", {}).get("user_id", None)
        if not user_id:
            print("Warning: No user_id found. Using default profile.")
            return {"personalization_context": {
                "name": "User", "preferences": {"tone": "formal", "format": "bullet points"}, "past_interactions_summary": ""
            }}
        profile = self.profile_manager.get_profile(user_id)
        if profile:
            # Force user identity to match the authenticated session user_id
            profile["name"] = user_id
            print(f"Loaded profile for user: '{user_id}' (Preferences: {profile.get('preferences')})")
            return {"personalization_context": profile}
        else:
            default_profile = {
                "name": user_id, "preferences": {"tone": "formal", "format": "bullet points"}, "past_interactions_summary": ""
            }
            print(f"Created default profile for new user: '{user_id}'")
            return {"personalization_context": default_profile}

    def manage_chat_history_node(self, state: AgentState) -> dict:
        print("--- Node: Managing Chat History ---")
        updated_chat_history = state.chat_history + [HumanMessage(content=state.initial_query)]

        if not updated_chat_history:
            return {"summarized_history": "", "chat_history": updated_chat_history}

        # With Gemma 3's high context, keep up to 16 messages (8 turns) raw without lossy micro-summarization
        if len(updated_chat_history) <= 16:
            summary = "\n".join([f"{msg.type}: {msg.content}" for msg in updated_chat_history])
            return {"summarized_history": summary, "chat_history": updated_chat_history}

        # For long conversations, summarize older exchanges and preserve the recent 6 turns
        recent = updated_chat_history[-6:]
        older = updated_chat_history[:-6]
        summary_prompt = f'''<start_of_turn>user
        Briefly summarize the key facts from this earlier conversation:
        { "\n".join([f'{msg.type}: {msg.content}' for msg in older]) }
        SUMMARY:<end_of_turn><start_of_turn>model'''
        summary_response = self.llm(prompt=summary_prompt, max_tokens=200, stop=["<end_of_turn>"])
        summary = summary_response["choices"][0]["text"].strip()
        formatted_recent = "\n".join([f"{msg.type}: {msg.content}" for msg in recent])
        return {"summarized_history": f"[Prior Context Summary: {summary}]\n{formatted_recent}", "chat_history": updated_chat_history}

    def route_query_node(self, state: AgentState) -> dict:
        print("--- Node: Routing Query (Single-Pass Semantic Router) ---")
        query_raw = state.initial_query.strip()

        # 1. Direct System Command Routing
        if query_raw.lower().startswith("/ingest "):
            pdf_path = query_raw[len("/ingest "):].strip()
            return {"query_route": "ingest_pdf", "original_query": pdf_path} if os.path.exists(pdf_path) else \
                   {"query_route": "conversational_query", "final_answer": f"Error: PDF not found at {pdf_path}"}

        # 2. Single-Pass 3-Way Semantic Intent Classification
        routing_prompt = f'''<start_of_turn>user
[YOUR TASK]
You are an expert query router. Classify the user's message into EXACTLY ONE of three categories:

1. CHITCHAT: Any greeting in any language (e.g., "Hi", "Hello", "Hola", "Namaste", "Bonjour"), social pleasantry, farewell, expression of thanks ("Thank you"), or question about who you are.
2. DOCUMENT_QA: Any question or topic that directly relates to the loaded document's content and themes.
3. WEB_SEARCH: Any question asking for external facts, news, people, entities, or topics not covered by the document.

[LOADED DOCUMENT TOPICS]
{state.indexed_document_summary}

[USER MESSAGE]
"{query_raw}"

[INSTRUCTION]
Respond with ONLY ONE word: CHITCHAT, DOCUMENT_QA, or WEB_SEARCH.

Category:<end_of_turn>
<start_of_turn>model
'''
        try:
            response = self.llm(prompt=routing_prompt, max_tokens=15, temperature=0.0, stop=["<end_of_turn>"])
            decision = response["choices"][0]["text"].strip().upper()
            print(f"Router Semantic Decision: '{decision}'")
        except Exception as e:
            print(f"Router error: {e}. Defaulting to DOCUMENT_QA.")
            decision = "DOCUMENT_QA"

        if "CHITCHAT" in decision:
            return {"query_route": "conversational_query"}
        elif "WEB_SEARCH" in decision:
            return {"query_route": "web_search_query"}
        else:
            return {"query_route": "rag_query"}

    def generate_simple_response_node(self, state: AgentState) -> dict:
        print("--- Node: Generating Simple Conversational Response ---")
        user_name = state.personalization_context.get("name", "there")
        user_msg = state.initial_query.strip()

        prompt = f'''<start_of_turn>user
[YOUR TASK]
You are a friendly, professional on-device AI assistant.
The user ({user_name}) has said: "{user_msg}"

Provide a warm, polite, and concise conversational response (1 to 2 sentences).
Address {user_name} by name.
Politely indicate you are ready to answer questions about the loaded document or search the web.
Do NOT output bullet points, document summaries, or unsolicited technical topics.

[RESPONSE]
<end_of_turn>
<start_of_turn>model
'''
        response = self.llm(prompt=prompt, max_tokens=80, temperature=0.7, stop=["<end_of_turn>"])
        return {"final_answer": response["choices"][0]["text"].strip()}

    def transform_query_node(self, state: AgentState) -> dict:
        print("--- Node: Unified Anaphora Resolution & Query Transformation ---")
        user_query = state.original_query
        chat_context = state.summarized_history.strip() if state.summarized_history else ""

        # Consolidated single-pass prompt for Anaphora Resolution + Multi-Query Generation
        prompt = f'''<start_of_turn>user
[YOUR TASK]
You are a retrieval optimization assistant for an on-device technical search engine.
Analyze the user's latest question in the context of recent chat exchanges.

1. ANAPHORA RESOLUTION: If the question contains pronouns or implicit references (e.g. "he", "she", "it", "they", "this principle", "the author", "that company"), resolve them into explicit entity names based on the conversation history.
2. MULTI-QUERY EXPANSION: Generate the resolved standalone query, followed by 2 distinct affirmative search query variations optimized for keyword (BM25) and semantic (vector) document retrieval.

[RULES]
- Output ONLY the search queries, one per line.
- Do NOT output numbers, bullets, or conversational remarks.

[CHAT HISTORY]
{chat_context if chat_context else "No prior history."}

[LATEST QUESTION]
{user_query}

[EXPANDED SEARCH QUERIES]
<end_of_turn>
<start_of_turn>model
'''
        try:
            response = self.llm(prompt=prompt, max_tokens=150, temperature=0.3, stop=["<end_of_turn>"])
            raw_text = response["choices"][0]["text"].strip()
            lines = [re.sub(r'^\d+[\.\)]\s*|^[-*]\s*', '', l).strip() for l in raw_text.splitlines() if l.strip()]
            generated_queries = [q for q in lines if len(q) > 3][:3]
        except Exception as e:
            print(f"Query transformation error: {e}")
            generated_queries = []

        all_queries = [user_query]
        for q in generated_queries:
            if q.lower() != user_query.lower() and q not in all_queries:
                all_queries.append(q)

        print(f"Generated {len(all_queries)} unified search queries (Primary: '{all_queries[0]}'):")
        for i, q in enumerate(all_queries):
            print(f"  {i+1}. {q}")

        return {"transformed_queries": all_queries}

    def retrieve_documents_node(self, state: AgentState) -> dict:
        print("--- Node: Retrieving Documents ---")
        queries = state.transformed_queries
        current_retrieval_query = queries[0]
        cached_results = None if state.iteration_count > 0 else self.retrieval_cache.check_cache(current_retrieval_query)
        if cached_results:
            print("Cache HIT.")
            retrieved_ids = cached_results
        else:
            print("Cache MISS. Performing hybrid retrieval.")
            retrieved_results = self.hybrid_retriever.retrieve(queries)
            # The new retriever returns a list of (chunk_id, (count, score)) tuples
            retrieved_ids = [result[0] for result in retrieved_results]
            self.retrieval_cache.add_to_cache(current_retrieval_query, retrieved_ids)
        
        retrieved_docs = [chunk for chunk in self.hybrid_retriever.active_all_chunks if chunk["metadata"]["chunk_id"] in retrieved_ids]
        
        # --- OBSERVABILITY: Sort the retrieved_docs based on the reranked order ---
        order_map = {chunk_id: i for i, chunk_id in enumerate(retrieved_ids)}
        retrieved_docs.sort(key=lambda doc: order_map.get(doc['metadata']['chunk_id'], float('inf')))
        print(f"Retrieved {len(retrieved_docs)} documents.")

        # --- OBSERVABILITY: Print top 5 chunks ---
        print("--- Top 5 Retrieved Chunks ---")
        for i, chunk in enumerate(retrieved_docs[:5]):
            print(f"--- Chunk {i+1} ---")
            print(chunk['content'])
            print("--------------------")
        # --- END OBSERVABILITY ---

        return {"retrieved_chunks": retrieved_docs}

    def grade_and_answer_node(self, state: AgentState) -> dict:
        """
        Attempts to answer the question using the retrieved context and grades the relevance.
        This combined node is more efficient and reliable than separate grading and generation steps.
        """
        print("--- Node: Grading and Answering ---")
        context = "\n\n".join([chunk['content'] for chunk in state.retrieved_chunks[:5]])

        # --- OBSERVABILITY: Print full context for LLM ---
        print("--- Context Sent to LLM for Answering ---")
        print(context)
        print("-----------------------------------------")
        # --- END OBSERVABILITY ---

        prompt = f'''<start_of_turn>user
        You are an expert assistant. Your task is to answer the user's question based ONLY on the provided context.
        Consider the user's profile and preferences when formulating your answer to personalize it.
        User Profile: {json.dumps(state.personalization_context, indent=2)}

        After generating a direct answer, you must rate how relevant the provided context was for creating that answer.
        The rating should be a score from 1 to 5, where 1 means "Not at all relevant" and 5 means "Highly relevant and sufficient".

        You MUST respond with a single, valid JSON object containing two keys: "answer" and "relevance_score".
        Do not include any other text or formatting outside of the JSON object.

        QUESTION: {state.initial_query}

        CONTEXT:
        ---
        {context}
        ---

        JSON Response:<end_of_turn>
        <start_of_turn>model
        '''
        
        response = self.llm(prompt=prompt, max_tokens=1024, stop=["<end_of_turn>"])
        response_text = response["choices"][0]["text"].strip()

        try:
            # Find the JSON block in the response
            json_match = re.search(r'```json\n(.*?)```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text

            response_json = json.loads(json_str)
            answer = response_json.get("answer", "")
            score = int(response_json.get("relevance_score", 1))

            if score >= 4:
                print(f"High relevance score ({score}). Using generated answer.")
                return {"final_answer": answer, "is_relevant": "yes"}
            else:
                print(f"Low relevance score ({score}). Triggering rewrite.")
                return {"is_relevant": "no"}

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error decoding JSON or parsing score: {e}")
            print(f"LLM Response was: {response_text}")
            return {"is_relevant": "no"}

    def rewrite_query_node(self, state: AgentState) -> dict:
        print("--- Node: Rewriting Query ---")
        # Create a summary of the current turn for the rewrite prompt
        turn_summary = f"Attempted to answer '{state.initial_query}'. The retrieved context was not sufficient (relevance score < 4)."

        prompt = f'''<start_of_turn>user
        You are a search query rewriter. Your task is to generate a new search query that takes a different approach to find information for the user's question, as previous attempts failed.
        Look at the original question and the summary of the failed attempt.
        Generate a single, new search query that is a variation of the original. It should be concise and focused.
        
        **IMPORTANT**: Your response must ONLY contain the new search query and nothing else. Do not add any rationale, explanation, or formatting.

        Original Question: {state.initial_query}
        Summary of Failed Attempt: {turn_summary}

        New Search Query:<end_of_turn>
        <start_of_turn>model
        '''
        response = self.llm(prompt=prompt, max_tokens=100, stop=["<end_of_turn>"])
        new_query = response["choices"][0]["text"].strip()
        print(f"Generated new query: '{new_query}'")
        return {"original_query": new_query, "iteration_count": state.iteration_count + 1}

    def generate_failure_response_node(self, state: AgentState) -> dict:
        print("--- Node: Generating Graceful Failure Response ---")
        return {"final_answer": "I'm sorry, but I couldn't find a relevant answer in the document. I will now try a web search for you."}

    def web_search_node(self, state: AgentState) -> dict:
        print(f"--- Node: Performing Tiered Web Search for: '{state.original_query}' ---")
        results, engine = hybrid_web_search(state.original_query, max_results=5)

        if not results:
            return {
                "final_answer": f"I attempted to search the web for '{state.original_query}' using both DuckDuckGo and Google Search, but could not find verified information at this time."
            }

        snippets_text = "\n\n".join([
            f"[{i+1}] {item['title']}\nSource: {item['link']}\nSnippet: {item['snippet']}"
            for i, item in enumerate(results)
        ])

        summary_prompt = f'''<start_of_turn>user
[YOUR TASK]
You are a precise, factual research assistant.
Answer the user's question using EXCLUSIVELY the verified search results provided below.

[RULES]
- Ground your answer 100% in the provided search results.
- Do NOT use prior knowledge, unverified assumptions, or invent facts.
- If the search results do not contain enough information to answer the question, clearly state that the search results were inconclusive.

[QUESTION]
{state.original_query}

[VERIFIED WEB SEARCH RESULTS (via {engine})]
{snippets_text}

[ANSWER]
<end_of_turn>
<start_of_turn>model
'''
        try:
            summary_response = self.llm(prompt=summary_prompt, max_tokens=400, temperature=0.2, stop=["<end_of_turn>"])
            answer = summary_response["choices"][0]["text"].strip()

            # Format final answer with search engine badge and source links
            links_list = "\n".join([f"- [{item['title']}]({item['link']})" for item in results[:3] if item.get('link')])
            final_formatted = f"**Web Search Results (via {engine}):**\n\n{answer}\n\n**Sources:**\n{links_list}"
            return {"final_answer": final_formatted}
        except Exception as e:
            print(f"Error during web search synthesis: {e}")
            return {"final_answer": f"Web search retrieved information via {engine}, but an error occurred while generating the summary."}

    def ingest_pdf_node(self, state: AgentState) -> dict:
        print(f"--- Node: Ingesting PDF: {state.original_query} ---")
        pdf_path = state.original_query
        pdf_name_without_ext = os.path.splitext(os.path.basename(pdf_path))[0]
        success, message = pdf_manager.process_and_index_pdf(pdf_path)
        if not success: return {"final_answer": message}
        self.hybrid_retriever.load_pdf_for_retrieval(pdf_name_without_ext)
        paths = pdf_manager.get_pdf_output_dirs(pdf_name_without_ext)
        self.retrieval_cache = RetrievalCache(cache_dir=paths["retrieval_cache_dir"], embedding_model=self.embedding_model)
        pdf_filename = os.path.basename(pdf_path)
        summary = get_pdf_summaries().get(pdf_filename, "No summary available.")
        updated_summary = f"Content from {pdf_filename} is now available. {summary}"
        return {"final_answer": f"Successfully ingested {pdf_filename}.", "indexed_document_summary": updated_summary}

    def update_history_and_profile_node(self, state: AgentState, config: dict) -> dict:
        print("--- Node: Updating History and Profile ---")
        user_id = config.get("configurable", {}).get("user_id")
        state.chat_history.append(AIMessage(content=state.final_answer))
        if not user_id:
            return {"chat_history": state.chat_history}

        # Summarize current turn for past_interactions_summary
        current_turn_summary_prompt = f'''<start_of_turn>user
        Summarize the following conversation turn concisely, focusing on key topics and user intent.
        Chat History: {state.summarized_history}
        User Query: {state.original_query}
        Agent Response: {state.final_answer}
        Summary:<end_of_turn>
        <start_of_turn>model
        '''
        current_turn_summary_response = self.llm(prompt=current_turn_summary_prompt, max_tokens=100, stop=["<end_of_turn>"])
        current_turn_summary = current_turn_summary_response["choices"][0]["text"].strip()

        # Consolidate with existing past_interactions_summary
        existing_summary = state.personalization_context.get("past_interactions_summary", "")
        consolidated_summary_prompt = f'''<start_of_turn>user
        Consolidate the following existing interaction summary with the new turn summary.
        The consolidated summary should be a concise, evolving record of the user's primary interests, key topics discussed, and any recurring themes across all interactions.
        **Ensure that core subject matter and long-term interests are retained and integrated, not replaced, by new conversational details.**
        Aim for a maximum of 400 characters.

        Existing Summary: {existing_summary}
        New Turn Summary: {current_turn_summary}

        Consolidated Summary:<end_of_turn>
        <start_of_turn>model
        '''
        consolidated_summary_response = self.llm(prompt=consolidated_summary_prompt, max_tokens=400, stop=["<end_of_turn>"])
        consolidated_summary = consolidated_summary_response["choices"][0]["text"].strip()

        state.personalization_context["name"] = user_id
        state.personalization_context["past_interactions_summary"] = consolidated_summary

        print(f"--- Profile Saved: User='{user_id}', Preferences={state.personalization_context.get('preferences')}, Summary='{state.personalization_context.get('past_interactions_summary')}' ---")
        self.profile_manager.save_profile(user_id, state.personalization_context)
        return {"chat_history": state.chat_history, "personalization_context": state.personalization_context}

    def cleanup_state_node(self, state: AgentState) -> dict:
        print("--- Node: Cleaning Up State ---")
        return {
            "retrieved_chunks": [], "transformed_queries": [], "is_relevant": "",
            "iteration_count": 0, "current_turn_messages": [], "current_conversation_summary": ""
        }

    def should_continue(self, state: AgentState) -> str:
        if state.is_relevant == "yes":
            return "end_with_answer"
        elif state.iteration_count >= 1:
            return "handle_failure"
        else:
            return "rewrite_query"

    def should_route_from_failure(self, state: AgentState) -> str:
        return "web_search_query"


# --- Gradio UI ---
class RAGInterface:
    def __init__(self, memory: SqliteSaver):
        self.agent = None
        self.user_id = None
        self.thread_id = None
        self.available_pdfs = self._get_available_pdfs()
        self.memory = memory

    def _get_available_pdfs(self) -> List[str]:
        source_dir = config.PDF_SOURCE_DIR
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
            return []
        return sorted([f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")])

    def start_session(self, pdf_filename: str, user_id: str):
        if not pdf_filename or not user_id:
            raise gr.Error("Please select a PDF and enter a User ID.")
        
        print(f"Starting session for user '{user_id}' with PDF '{pdf_filename}'")
        pdf_path = os.path.join(config.PDF_SOURCE_DIR, pdf_filename)
        
        self.agent = RAGAgent(selected_pdf_path=pdf_path, memory=self.memory)
        self.user_id = user_id
        self.thread_id = str(uuid.uuid4())
        
        print(f"New conversation started with Thread ID: {self.thread_id}")
        
        # Generate Mermaid diagram
        try:
            mermaid_markup = self.agent.app.get_graph().draw_mermaid()

            # Create a self-contained HTML document for the iframe.
            iframe_html_doc = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ margin: 0; padding: 10px; box-sizing: border-box; background-color: white; }}
                    .mermaid {{ text-align: center; }}
                </style>
            </head>
            <body>
                <div class="mermaid">
                    {mermaid_markup}
                </div>
                <script type="module">
                    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                    await mermaid.run();
                </script>
            </body>
            </html>
            """

            # Encode the HTML document to Base64
            doc_bytes = iframe_html_doc.encode('utf-8')
            b64_doc = base64.b64encode(doc_bytes).decode('utf-8')
            data_uri = f"data:text/html;base64,{b64_doc}"

            # Use a Data URI in the iframe src for maximum compatibility
            mermaid_html = f'<iframe src="{data_uri}" style="width: 100%; height: 600px; border: none; border-radius: 8px;"></iframe>'

        except Exception as e:
            print(f"Could not generate Mermaid diagram: {e}")
            mermaid_html = "<p>Could not render agent architecture diagram.</p>"

        summary_raw = get_pdf_summaries().get(pdf_filename, "No topic summary available for this PDF.")
        summary_clean = summary_raw.replace("Topics included: ", "").replace("Summary of the PDF selected: ", "")
        formatted_summary = f"**Topics in this document:** {summary_clean}"

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False), # Hide loading message
            f"Chatting with {pdf_filename}",
            [],
            gr.update(value=formatted_summary),
            gr.update(value=mermaid_html)
        )
    
    def end_session(self):
        """
        Resets the session state and returns the UI to the setup screen.
        """
        print(f"Ending session for user '{self.user_id}' with Thread ID: {self.thread_id}")
        self.agent = None
        self.user_id = None
        self.thread_id = None
        
        # Return updates to Gradio components
        return (
            [],                       # Clear the chatbot messages
            gr.update(visible=False), # Hide the chat_screen
            gr.update(visible=True) ,   # Show the setup_screen
            gr.Dropdown(choices=self.available_pdfs, value=None, info="Please select a PDF from the dropdown."), # Reset PDF dropdown
            gr.update(visible=False), # Ensure loading message is hidden when returning to setup
            gr.update(value=""),       # Clear the topic summary
            gr.update(value="")       # Clear the architecture view
        )

    def chat(self, message: str, history: List[dict[str, str]]) -> Tuple[str, List[dict[str, str]]]:
        if self.agent is None:
            raise gr.Error("Agent not initialized. Please start a session first.")
            
        print(f"User '{self.user_id}' | Thread '{self.thread_id}' | Query: '{message}'")
        
        # Append user message to history in Gradio's expected dictionary format
        history.append({"role": "user", "content": message})
        yield "", history # <--- First yield: Clears textbox, shows user message in chatbot

        # Add a "thinking" message from the assistant immediately
        thinking_message_dict = {"role": "assistant", "content": "Thinking... &#129504;"}
        history.append(thinking_message_dict)
        yield "", history # <--- Second yield: Updates chatbot to show "Thinking..." below user's message

        # Now, perform the actual RAG agent invocation (the long-running part)
        final_state = self.agent.invoke(message, self.user_id, self.thread_id)
        response = final_state.get('final_answer', "Sorry, something went wrong.")
        
        print(f"Agent Response: {response}")

        # Replace the "thinking" message with the actual response
        # Check if the last message is indeed our thinking message (by content, as objects might differ)
        if history and history[-1].get("content") == thinking_message_dict["content"]:
            history[-1]["content"] = response # Update content of the last message
        else:
            # Fallback: if for some reason the thinking message isn't last, append the response
            history.append({"role": "assistant", "content": response})

        yield "", history # <--- Third yield: Updates chatbot with the final response

    def launch(self):
        # Define the custom CSS here
        CUSTOM_CSS = """
        /* Keyframes for the pulsing animation */
        @keyframes pulse {
            0% { opacity: 0.5; }
            50% { opacity: 1; }
            100% { opacity: 0.5; }
        }

        /* Styling for the entire loading message block (targeting its elem_id) */
        #loading_status_message {
            background-color: #f0f8ff; /* Light blue background for contrast */
            border: 2px solid #6C5CE7; /* Border matching your theme's primary color */
            border-radius: 8px; /* Rounded corners */
            padding: 15px; /* Internal spacing around content */
            margin-top: 20px; /* Space from elements above it */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
            text-align: center; /* Center align all text within this block */
        }

        /* Apply pulse animation to the specific text span inside the block */
        #loading_status_message .pulsing-text {
            animation: pulse 1.5s infinite ease-in-out;
        }

        /* Optional: Adjust default margins of elements inside the markdown for better spacing */
        #loading_status_message h3,
        #loading_status_message p {
            margin: 0; /* Remove default top/bottom margins */
        }
        #loading_status_message h3 {
            padding-bottom: 5px; /* Small space between heading and paragraph */
        }
        """

        # Define the HTML content for the loading message (without inline styles now)
        LOADING_HTML_CONTENT = f"""
        <h3>
            <span class='pulsing-text'>&#128269; Loading RAG Agent... Please wait.</span>
        </h3>
        <p>This may take a moment as models and data are being prepared.</p>
        """
        with gr.Blocks(theme=gr.themes.Soft(), title="RAG Agent UI", css=CUSTOM_CSS) as demo:
            gr.Markdown("# RAG Agent - Chat with PDF (v1)")

            with gr.Column(visible=True) as setup_screen:
                gr.Markdown("## 1. Configure Your Session")
                pdf_dropdown = gr.Dropdown(self.available_pdfs, label="Select a PDF to Chat With", value=None, info="Please select a PDF from the dropdown...")
                user_id_input = gr.Textbox(label="Enter Your User ID")
                start_button = gr.Button("Start Chat", variant="primary")
                loading_message = gr.Markdown(
                    LOADING_HTML_CONTENT, 
                    visible=False,
                    elem_id="loading_status_message"
                )

            with gr.Column(visible=False) as chat_screen:
                chat_title = gr.Markdown("## Chat")
                
                topic_summary = gr.Markdown(label="Topic Summary")
                
                gr.Markdown("---")

                with gr.Accordion("🤖 Peek Inside the Agent's Brain: Click here to View Workflow", open=False):
                    architecture_view = gr.HTML(value="")
                
                # Add a horizontal line separator
                gr.Markdown("---")

                chatbot = gr.Chatbot([], elem_id="chatbot", type="messages", height=500)
                
                with gr.Row():
                    txt = gr.Textbox(scale=4, show_label=False, placeholder="Type-in your query here (press ENTER key to submit)...", container=False)
                
                txt.submit(self.chat, [txt, chatbot], [txt, chatbot])

                # Add another separator before the final button
                gr.Markdown("---")
                
                end_button = gr.Button("End Chat", variant="primary")

            start_button.click(
                # First, show loading message, then run start_session
                lambda: gr.update(visible=True),
                inputs=None,
                outputs=[loading_message]
            ).then(
                self.start_session,
                inputs=[pdf_dropdown, user_id_input],
                outputs=[setup_screen, chat_screen, loading_message, chat_title, chatbot, topic_summary, architecture_view]
            )

            end_button.click(
                self.end_session,
                inputs=None,
                outputs=[chatbot, chat_screen, setup_screen, pdf_dropdown, loading_message, topic_summary, architecture_view]
            )
        print("Launching Gradio Interface...")
        demo.launch(share=True)

if __name__ == "__main__":
    try:
        with SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as memory:
            ui = RAGInterface(memory=memory)
            ui.launch()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")