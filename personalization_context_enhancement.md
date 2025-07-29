# Personalization Context Enhancement Plan

## 1. Current Drawbacks in Handling `personalization_context`

The existing RAG agent workflow, as implemented in `rag_agent_with_ui_v2.py`, has the following limitations regarding the `personalization_context`:

*   **Static Profile within a Turn:** The `personalization_context` is loaded at the beginning of a user session/turn via `load_user_profile_node` and remains unchanged throughout that turn.
*   **No Conversational Update Logic:** There is no explicit mechanism within any of the agent's nodes to detect, parse, and apply user-stated preferences (e.g., "change my tone to casual", "use bullet points") to the `state.personalization_context` based on conversational input.
*   **No Dynamic `past_interactions_summary` Update:** The `past_interactions_summary` field within `personalization_context` is not dynamically updated or summarized based on the content of the current conversation. While it might contain pre-existing data from previous sessions (if manually populated or updated by an older agent version), the current agent does not actively learn and consolidate new interaction summaries into this field.
*   **Saving Unchanged Data:** Consequently, when `update_history_and_profile_node` saves the profile at the end of a turn, it saves the same `personalization_context` that was loaded, as no intermediate node has modified its content based on the ongoing dialogue.

In essence, the agent can *read* and *use* the `personalization_context` for informing LLM prompts, but it cannot *learn* or *update* its content through conversational interaction.

## 2. Detailed Logic for Improvement

The goal is to enable the agent to dynamically update the `personalization_context` based on conversational cues, thereby allowing it to learn user preferences and maintain an evolving summary of past interactions.

### Revised Proposed Solution:

**A. Introduce a new `extract_and_update_preferences_node`:**

*   **Objective:** To detect and apply user-stated preferences (e.g., tone, formatting, name) to the `personalization_context` as soon as they are expressed, regardless of the query type.
*   **Placement:** This new node will be placed early in the graph, specifically **after `summarize_history_node` and before `route_query_node`**. This ensures that all user queries (conversational, RAG, web search, ingest) pass through this node, allowing for universal preference detection.
*   **Mechanism:**
    *   An LLM call will be introduced within this new node.
    *   **Input to LLM:** The user's `state.original_query` (latest input) and the current `state.personalization_context`.
    *   **LLM Prompt:** Will instruct the LLM to analyze the input and identify any expressed preference changes.
    *   **Expected LLM Output:** A structured JSON object containing identified preference changes (e.g., `{"preferences": {"tone": "casual"}, "name": "John Doe"}`). If no changes are detected, an empty JSON object `{}`.
    *   **Processing LLM Output:** The agent will parse this JSON output. If valid preferences are found, they will be merged into the `state.personalization_context['preferences']` dictionary. The top-level `name` key will also be updated if specified.
    *   **Benefit:** Ensures that any stated preferences are immediately incorporated into the `personalization_context` for the current response and subsequent interactions within the same session, regardless of the query's primary intent.

**B. Enhance `update_history_and_profile_node` for `past_interactions_summary`:**

*   **Objective:** To maintain a concise, evolving summary of the user's overall interests and key topics discussed across all interactions.
*   **Mechanism:**
    *   An LLM call will be introduced within the `update_history_and_profile_node` (which is already responsible for saving the profile).
    *   **Input to LLM:** The `state.chat_history` (or a summary of the current turn's messages) and the `existing` `state.personalization_context['past_interactions_summary']`.
    *   **LLM Prompt:** Will instruct the LLM to generate an updated, consolidated summary that integrates the new information from the current conversation into the existing summary.
    *   **Expected LLM Output:** A concise string representing the updated `past_interactions_summary`.
    *   **Processing LLM Output:** The generated summary will update `state.personalization_context['past_interactions_summary']`.
    *   **Benefit:** Allows the agent to build a persistent, evolving understanding of the user's interests and past topics, which can then be leveraged by other nodes (like `transform_query_node` and `grade_and_answer_node`) for better contextualization and personalization in future turns and sessions.

## 3. Assumptions and Hypotheses

The successful implementation of this enhancement relies on the following assumptions and hypotheses:

*   **LLM's Ability to Extract Preferences:**
    *   **Hypothesis:** The chosen LLM (Gemma-3n-E4B-it-Q4_K_M.gguf) will be capable of reliably identifying and extracting user preferences (e.g., tone, formatting, name) from natural language queries and outputting them in the specified JSON format.
    *   **Assumption:** The prompt engineering for preference extraction will be effective in guiding the LLM to produce consistent and accurate results.
    *   **New Assumption:** The LLM can accurately extract preferences even when they are embedded within a query that has another primary intent (e.g., a RAG query).

*   **LLM's Ability to Summarize and Consolidate `past_interactions_summary`:**
    *   **Hypothesis:** The LLM can effectively summarize the current conversation and integrate it into an existing `past_interactions_summary` while maintaining conciseness and relevance.
    *   **Assumption:** The LLM can adhere to specified token/character limits for the `past_interactions_summary`, prioritizing and compressing information as needed to prevent context bloat.

*   **Structure of `preferences` and `past_interactions_summary`:**
    *   **Hypothesis:** The defined structure for `personalization_context['preferences']` (e.g., `tone`, `formatting`, `language`, `level_of_detail`) is sufficient to capture the most common and impactful user preferences.
    *   **Assumption:** The `past_interactions_summary` as a single string is an adequate representation for long-term user interests, and a more complex structured data type is not immediately necessary.

*   **Merging Logic for Preferences:**
    *   **Hypothesis:** The Python dictionary `update()` method (or similar merging logic) will correctly integrate new preferences into the existing `preferences` dictionary without unintended side effects.

*   **Performance Impact:**
    *   **Hypothesis:** The addition of two new LLM calls per turn (one for preference extraction, one for summary update) will not significantly degrade the agent's response time to an unacceptable level.
    *   **Assumption:** The LLM calls are efficient enough for interactive use.

*   **Robustness to LLM Output Errors:**
    *   **Hypothesis:** The parsing of LLM-generated JSON for preferences will be robust to potential malformed output, with appropriate error handling to prevent crashes.

This document will serve as our reference as we proceed with the implementation.
