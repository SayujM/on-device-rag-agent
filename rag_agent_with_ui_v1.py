import os
import sys
import json
import uuid
from typing import List, Tuple
import re
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
import pdf_manager

# Load environment variables from .env file
load_dotenv()

# --- Static Configuration & Summaries ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "langgraph_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_DB_PATH = os.path.join(CHECKPOINT_DIR, "checkpoints.sqlite")

PDF_SUMMARIES = {
    "ELECTRIC_CHARGES_AND_FIELDS.pdf": "Topics included: Electric charges and their properties, conductors and insulators, charging by induction, Coulomb's Law, the principle of superposition for forces between multiple charges, electric field, electric field lines, electric dipole, electric field due to a point charge, electric field due to an electric dipole (on axial and equatorial lines), torque on an electric dipole in a uniform electric field, continuous charge distribution (linear, surface, and volume charge density), Gauss's Law and its applications (electric field due to an infinitely long straight uniformly charged wire, uniformly charged infinite plane sheet, and uniformly charged thin spherical shell).",
    "The-toyota-way-second-edition-00_chapter_preface.pdf": "Topics included: Introduction to the second edition of 'The Toyota Way', the evolution of Toyota and the book's content since the first edition, the concept of operational excellence as a strategic weapon, the enduring relevance of the Toyota Way principles for various organizations, and a brief overview of the book's structure and updated content, including new insights into Toyota's long-term thinking and adaptability.",
    "The-toyota-way-second-edition-01_chapter_1.pdf": "Topics included: The Toyota Way philosophy, operational excellence, annual net income/loss of major automakers (Toyota, Ford, Volkswagen, General Motors, Honda) from 2004-2018, vehicle quality and long-term reliability ratings, the Toyota Production System (TPS), and scientific thinking as applied to problem-solving and continuous improvement within an organization.",
    "The-toyota-way-second-edition-02_chapter_2.pdf": "Topics included: The Toyota Way 2001 values (Continuous Improvement and Respect for People), the Toyota Production System (TPS) as the operational system, historical context of TPS development (Kiichiro Toyoda, Eiji Toyoda, Taiichi Ohno), the two pillars of TPS: Just-in-Time (JIT) and Jidoka (automation with a human touch), Heijunka (production leveling), Standardized Work, Kaizen (continuous improvement), Lean manufacturing principles, and the evolution of the Toyota Way philosophy.",
    "The-toyota-way-second-edition-03_chapter_3.pdf": "Topics included: Genchi Genbutsu (go and see for yourself) as a Toyota Way principle, practical application of Genchi Genbutsu for problem-solving and decision-making, the importance of understanding the 'gemba' (the actual place where work is done), examples of Toyota's leaders and their commitment to Genchi Genbutsu, the contrast between Toyota's approach and common Western management practices, Nemawashi (consensus building) and its role in decision-making, and A3 reports as a tool for structured problem-solving and communication.",
    "The-toyota-way-second-edition-04_chapter_4.pdf": "Topics included: Creating continuous process flow to bring problems to the surface, the concept of one-piece flow, reducing work-in-process (WIP) inventory, linking processes and people for smooth flow, takt time as the rate of production needed to meet customer demand, and the benefits of flow in identifying waste and improving efficiency.",
    "The-toyota-way-second-edition-05_chapter_5.pdf": "Topics included: Building a culture of stopping to fix problems to get quality right the first time (Jidoka), Andon cords/lights as a system to signal abnormalities, the importance of immediate problem-solving at the source, root cause analysis, and integrating quality control into every step of the process rather than inspecting at the end.",
    "The-toyota-way-second-edition-06_chapter_6.pdf": "Topics included: Standardization as a foundation for continuous improvement and quality, the importance of stable processes, visual management, methods for creating and maintaining standards (e.g., standard work sheets), and how standardization supports problem-solving and training within the Toyota Production System.",
    "The-toyota-way-second-edition-07_chapter_7.pdf": "Topics included: Using pull systems to avoid overproduction, Just-in-Time (JIT) production, the concept of kanban as a pull signal, the dangers of push systems and overproduction, inventory reduction as a way to expose problems, and how pull systems link processes based on actual customer demand.",
    "The-toyuta-way-second-edition-08_chapter_8.pdf": "Topics included: Building quality into the process (Jidoka principle), the concept of 'stop and fix' errors immediately, creating a culture where employees are empowered to halt production for quality issues, the importance of root cause analysis and permanent countermeasures, and examples of Toyota's commitment to quality at the source.",
    "The-toyota-way-second-edition-09_chapter_9.pdf": "Topics included: Use of visual control to support people in decision-making and problem-solving, visual management principles, the importance of displaying information clearly and simply, examples of visual control systems (e.g., kanban, production boards), and how visual aids help identify abnormalities and facilitate immediate corrective action.",
    "The-toyota-way-second-edition-10_chapter_10.pdf": "Topics included: Developing stable and reliable suppliers as an extension of the enterprise, long-term partnerships with suppliers, supplier development and continuous improvement, mutual learning and trust, and integrating suppliers into the Toyota Production System.",
    "The-toyota-way-second-edition-11_chapter_11.pdf": "Topics included: Growing leaders from within who understand the work, live the philosophy, and teach it to others, the importance of developing people through daily work, creating a learning organization, and the role of humility and continuous self-improvement for leaders.",
    "The-toyota-way-second-edition-12_chapter_12.pdf": "Topics included: Developing exceptional people and teams, the importance of continuous learning and growth for employees, creating leaders who live the philosophy, building a culture of respect and challenge, and the role of leadership in fostering a lean environment.",
    "The-toyota-way-second-edition-13_chapter_13.pdf": "Topics included: Respecting and challenging value chain partners (suppliers and dealers), building long-term partnerships based on trust and mutual development, supplier selection and integration into the lean system, and helping partners improve their processes and capabilities to create a stable and efficient value chain.",
    "The-toyota-way-second-edition-14_chapter_14.pdf": "Topics included: Scientific thinking and continuous learning through iterative problem-solving (PDCA cycle), the importance of deep observation and understanding the current condition, setting challenging targets and experimenting to achieve them, and developing a learning culture that embraces failures as opportunities for improvement.",
    "The-toyota-way-second-edition-15_chapter_15.pdf": "Topics included: Hoshin Kanri (policy deployment) for aligning goals and ensuring systematic progress, the importance of long-term vision in planning, the catchball process for consensus building and communication across organizational levels, and how aligned goals drive continuous improvement throughout the enterprise.",
    "The-toyota-way-second-edition-16_chapter_16.pdf": "Topics included: Learning your way to the future through bold strategy and combining large leaps with small steps, the balance between incremental innovation and disruptive innovation, Toyota's strategic approach to new technologies and market shifts, the importance of a long-term vision in guiding innovation efforts, and examples of strategic breakthroughs and challenges.",
    "The-toyota-way-second-edition-17_chapter_conclusion.pdf": "Topics included: Sustaining a lean transformation and the Toyota Way principles, the challenge of integrating mechanistic and organic approaches to lean, the importance of leadership commitment and consistent effort, building a learning enterprise for continuous adaptation and improvement, and summary of key takeaways from The Toyota Way philosophy."
}

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
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        gguf_model_path = os.path.join(self.base_dir, "gemma-3n-E4B-it-Q4_K_M.gguf")
        chroma_db_dir = os.path.join(self.base_dir, "chroma_db")
        embedding_model_name = "all-MiniLM-L6-v2"

        print("Loading Query Transformer and Embedding Model...")
        self.query_transformer = QueryTransformer(model_path=gguf_model_path)
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
        profile_db_path = os.path.join(self.base_dir, "user_profiles.db")
        self.profile_manager = ProfileManager(db_path=profile_db_path)

        pdf_filename = os.path.basename(selected_pdf_path)
        summary = PDF_SUMMARIES.get(pdf_filename, "No specific summary available.")
        self.initial_indexed_doc_summary = f"Accessing docs from '{pdf_filename}'. {summary}"

        print("Building and compiling the agent graph...")
        self.memory = memory
        self.app = self._build_graph()
        print("--- RAG Agent Initialized Successfully ---")

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        # Node definitions
        workflow.add_node("load_user_profile", self.load_user_profile_node)
        workflow.add_node("add_initial_query_to_history", self.add_initial_query_to_history_node)
        workflow.add_node("summarize_history", self.summarize_history_node)
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

        workflow.set_entry_point("load_user_profile")
        # Edge definitions
        workflow.add_edge("load_user_profile", "add_initial_query_to_history")
        workflow.add_edge("add_initial_query_to_history", "summarize_history")
        workflow.add_edge("summarize_history", "route_query")
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
                "name": "Default User", "preferences": {"tone": "formal"}, "past_interactions_summary": ""
            }}
        profile = self.profile_manager.get_profile(user_id)
        if profile:
            print(f"Loaded profile for user: {user_id}")
            return {"personalization_context": profile}
        else:
            print(f"No profile for new user: {user_id}. Creating default.")
            return {"personalization_context": {
                "name": user_id, "preferences": {"tone": "formal"}, "past_interactions_summary": ""
            }}

    def add_initial_query_to_history_node(self, state: AgentState) -> dict:
        print("--- Node: Adding Initial Query to History ---")
        updated_chat_history = state.chat_history + [HumanMessage(content=state.initial_query)]
        return {"chat_history": updated_chat_history}

    def summarize_history_node(self, state: AgentState) -> dict:
        print("--- Node: Summarizing History ---")
        if not state.chat_history: return {"summarized_history": ""}
        if len(state.chat_history) <= 6:
            summary = "\n".join([f"{msg.type}: {msg.content}" for msg in state.chat_history])
            return {"summarized_history": summary}
        recent = state.chat_history[-4:]
        older = state.chat_history[:-4]
        summary_prompt = f'''<start_of_turn>user
        Briefly summarize the following conversation:
        { "\n".join([f'{msg.type}: {msg.content}' for msg in older]) }
        SUMMARY:<end_of_turn><start_of_turn>model'''
        summary_response = self.llm(prompt=summary_prompt, max_tokens=200, stop=["<end_of_turn>"])
        summary = summary_response["choices"][0]["text"].strip()
        formatted_recent = " ".join([f"{msg.type}: {msg.content}" for msg in recent])
        return {"summarized_history": f"[Summary: {summary}] [Recent:] {formatted_recent}"}

    def route_query_node(self, state: AgentState) -> dict:
        print("--- Node: Routing Query ---")
        if state.initial_query.lower().startswith("/ingest "):
            pdf_path = state.initial_query[len("/ingest "):].strip()
            return {"query_route": "ingest_pdf", "original_query": pdf_path} if os.path.exists(pdf_path) else \
                   {"query_route": "conversational_query", "final_answer": f"Error: PDF not found at {pdf_path}"}

        relevance_prompt = f'''<start_of_turn>user
        Is the user's question likely answerable by the document summary?
        Respond with 'yes', 'no', or 'maybe'.
        Summary: {state.indexed_document_summary}
        Question: {state.original_query}
        Answer:<end_of_turn><start_of_turn>model'''
        relevance_response = self.llm(prompt=relevance_prompt, max_tokens=10, stop=["<end_of_turn>"])
        relevance_grade = relevance_response["choices"][0]["text"].strip().lower()
        print(f"Doc Relevance: '{relevance_grade}'")
        if "yes" in relevance_grade or "maybe" in relevance_grade:
            return {"query_route": "rag_query"}

        conversational_check_prompt = f'''<start_of_turn>user
        Is the following a simple conversational question or does it need a web search?
        Respond 'conversational' or 'needs_web_search'.
        Question: {state.initial_query}
        Answer:<end_of_turn><start_of_turn>model'''
        conv_check_response = self.llm(prompt=conversational_check_prompt, max_tokens=20, stop=["<end_of_turn>"])
        conv_type = conv_check_response["choices"][0]["text"].strip().lower()
        print(f"Conversational Type: '{conv_type}'")
        return {"query_route": "conversational_query"} if "conversational" in conv_type else {"query_route": "web_search_query"}

    def generate_simple_response_node(self, state: AgentState) -> dict:
        print("--- Node: Generating Simple Conversational Response ---")
        prompt = f'''<start_of_turn>user
        Use the chat history to provide a friendly response.
        History: {state.summarized_history}
        User Message: {state.original_query}
        Response:<end_of_turn><start_of_turn>model'''
        response = self.llm(prompt=prompt, max_tokens=200, stop=["<end_of_turn>"])
        return {"final_answer": response["choices"][0]["text"].strip()}

    def transform_query_node(self, state: AgentState) -> dict:
        print("--- Node: Transforming Query ---")
        contextualize_prompt = f'''<start_of_turn>user
        Given the chat history and user profile, generate a standalone search query from the user's latest question.
        History: {state.summarized_history}
        Profile: {json.dumps(state.personalization_context, indent=2)}
        Question: {state.original_query}
        Standalone Query:<end_of_turn><start_of_turn>model'''
        ctx_query_response = self.llm(prompt=contextualize_prompt, max_tokens=100, stop=["<end_of_turn>"])
        contextualized_query = ctx_query_response["choices"][0]["text"].strip()
        print(f"Contextualized Query: {contextualized_query}")
        transformed_queries = self.query_transformer.transform_query(contextualized_query)
        all_queries = [contextualized_query] + transformed_queries
        print(f"Generated {len(all_queries)} queries.")
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
        print("--- Node: Performing Web Search ---")
        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("SEARCH_ENGINE_ID")
        if not api_key or not cse_id:
            return {"final_answer": "Web search is not configured."}
        try:
            service = build("customsearch", "v1", developerKey=api_key)
            res = service.cse().list(q=state.initial_query, cx=cse_id, num=3).execute()
            if "items" not in res:
                return {"final_answer": f"I couldn't find info for '{state.initial_query}' on the web."}
            snippets = "\n\n".join([item.get('snippet', '') for item in res.get("items", [])])
            summary_prompt = f'''<start_of_turn>user
            Summarize these web search results to answer: '{state.initial_query}'.
            Results: {snippets}
            Summary:<end_of_turn><start_of_turn>model
            '''
            summary_response = self.llm(prompt=summary_prompt, max_tokens=500, stop=["<end_of_turn>"])
            answer = summary_response["choices"][0]["text"].strip()
            return {"final_answer": f"Here's what I found on the web:\n\n{answer}"}
        except Exception as e:
            print(f"Error during web search: {e}")
            return {"final_answer": "An error occurred during the web search."}

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
        summary = PDF_SUMMARIES.get(pdf_filename, "No summary available.")
        updated_summary = f"Content from {pdf_filename} is now available. {summary}"
        return {"final_answer": f"Successfully ingested {pdf_filename}.", "indexed_document_summary": updated_summary}

    def update_history_and_profile_node(self, state: AgentState, config: dict) -> dict:
        print("--- Node: Updating History and Profile ---")
        user_id = config.get("configurable", {}).get("user_id")
        state.chat_history.append(AIMessage(content=state.final_answer))
        if not user_id:
            return {"chat_history": state.chat_history}
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
        source_dir = os.path.join(BASE_DIR, "pdf_files", "source")
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
            return []
        return sorted([f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")])

    def start_session(self, pdf_filename: str, user_id: str):
        if not pdf_filename or not user_id:
            raise gr.Error("Please select a PDF and enter a User ID.")
        
        print(f"Starting session for user '{user_id}' with PDF '{pdf_filename}'")
        pdf_path = os.path.join(BASE_DIR, "pdf_files", "source", pdf_filename)
        
        self.agent = RAGAgent(selected_pdf_path=pdf_path, memory=self.memory)
        self.user_id = user_id
        self.thread_id = str(uuid.uuid4())
        
        print(f"New conversation started with Thread ID: {self.thread_id}")
        
        summary_raw = PDF_SUMMARIES.get(pdf_filename, "No topic summary available for this PDF.")
        summary_clean = summary_raw.replace("Topics included: ", "")
        formatted_summary = f"**Topics in this document:** {summary_clean}"

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False), # Hide loading message
            f"Chatting with {pdf_filename}",
            [],
            gr.update(value=formatted_summary)
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
            gr.update(value="")       # Clear the topic summary
        )

    def chat(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        if self.agent is None:
            raise gr.Error("Agent not initialized. Please start a session first.")
            
        print(f"User '{self.user_id}' | Thread '{self.thread_id}' | Query: '{message}'")
        
        history.append([message, None])
        yield "", history # <--- First yield: Clears textbox, shows user message in chatbot

        # Add a "thinking" message from the assistant immediately
        history[-1][1] = "Thinking... &#129504;" # Assign a temporary "Thinking..." message
        yield "", history # <--- Second yield: Updates chatbot to show "Thinking..." below user's message

        # Now, perform the actual RAG agent invocation (the long-running part)
        final_state = self.agent.invoke(message, self.user_id, self.thread_id)
        response = final_state.get('final_answer', "Sorry, something went wrong.")
        
        print(f"Agent Response: {response}")
        history[-1][1] = response
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
                chatbot = gr.Chatbot([], elem_id="chatbot", type="messages", height=500)
                with gr.Row():
                    txt = gr.Textbox(scale=4, show_label=False, placeholder="Type-in your query here (press ENTER key to submit)...", container=False)
                txt.submit(self.chat, [txt, chatbot], [txt, chatbot])
                end_button = gr.Button("End Chat", variant="primary")

            start_button.click(
                # First, show loading message, then run start_session
                lambda: gr.update(visible=True),
                inputs=None,
                outputs=[loading_message]
            ).then(
                self.start_session,
                inputs=[pdf_dropdown, user_id_input],
                outputs=[setup_screen, chat_screen, loading_message, chat_title, chatbot, topic_summary]
            )

            end_button.click(
                self.end_session,
                inputs=None,
                outputs=[chatbot, chat_screen, setup_screen, pdf_dropdown, loading_message, topic_summary]
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