import operator
import logging
from typing import Annotated, Any, Dict, Optional
from typing_extensions import TypedDict, Literal
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
from dotenv import load_dotenv
from datetime import datetime
import re

# Load environment variables
load_dotenv()

from backend.memory.context_manager import ContextManager
from backend.memory.conversation_manager import ConversationManager
from utils import (
    call_llm,
    get_search_results,
    get_search_query_llm,
    call_thinking_llm,
    get_thinking_llm,
)
from config import TaskType
from prompt import (
    generate_search_queries_prompt,
    refine_search_queries_prompt,
    final_report_prompt,
)

# Constants
MAX_QUERY_GENERATIONS = 3
MAX_RESULT_TO_FETCH = 1
MAX_QUERY_REFINEMENTS = 2


# State definition for langgraph agent workflow
class DeepResearchState(TypedDict):
    user_query: str
    search_queries: Annotated[list, operator.add]
    search_results: Annotated[list, operator.add]
    individual_page_summaries: Annotated[list, operator.add]
    report_markdown: str
    query_generation_count: int
    claim_confidences: list
    reasoning_trace: Annotated[list, operator.add]
    analysis_summary: str
    analysis_results: Dict[str, Any]
    conversation_id: Optional[int]
    conversation_context: str
    is_followup: bool
    search_session_history: list


logger = logging.getLogger(__name__)
_context_manager = ContextManager()
_conversation_manager = ConversationManager()


def _ensure_conversation_id(conversation_id: Optional[int], user_query: str) -> int:
    if conversation_id:
        return conversation_id
    title = user_query.strip()[:200] or "Deep research conversation"
    conversation = _conversation_manager.create_conversation(title=title)
    return conversation.id


def _store_user_query(conversation_id: int, user_query: str) -> None:
    try:
        _conversation_manager.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_query,
            message_type="user_query",
        )
    except Exception as exc:
        logger.warning("Failed to store user query in conversation %s: %s", conversation_id, exc)


def _store_assistant_report(conversation_id: int, report: str) -> None:
    if not report:
        return
    try:
        _conversation_manager.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=report,
            message_type="report",
        )
    except Exception as exc:
        logger.warning("Failed to store assistant report in conversation %s: %s", conversation_id, exc)


def _store_reasoning_trace(conversation_id: int, reasoning_trace: list) -> None:
    if not reasoning_trace:
        return
    try:
        trace_content = "\n".join(f"{idx+1}. {step}" for idx, step in enumerate(reasoning_trace))
        _conversation_manager.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=trace_content,
            message_type="reasoning_trace",
        )
    except Exception as exc:
        logger.warning("Failed to store reasoning trace for conversation %s: %s", conversation_id, exc)


def _record_search_session(
    conversation_id: int, search_query: str, results: list
) -> None:
    titles = [result.get("title", "Untitled") for result in results[:3]]
    summary = " | ".join(titles) if titles else "No results retrieved."
    try:
        _conversation_manager.record_search_session(
            conversation_id=conversation_id,
            search_query=search_query,
            result_count=len(results),
            summary=summary,
        )
    except Exception as exc:
        logger.warning("Failed to record search session for conversation %s: %s", conversation_id, exc)


def _is_follow_up(conversation_id: int) -> bool:
    if not conversation_id:
        return False
    try:
        messages = _conversation_manager.get_messages(conversation_id, limit=50)
    except Exception:
        return False
    return any(message.role == "user" for message in messages)


def _format_search_session_history(sessions: list) -> str:
    if not sessions:
        return ""
    formatted = []
    for session in sessions:
        title = session.search_query if hasattr(session, "search_query") else session.get("search_query")
        summary = session.summary if hasattr(session, "summary") else session.get("summary")
        result_count = session.result_count if hasattr(session, "result_count") else session.get("result_count", 0)
        formatted.append(f"{title} ({result_count} results) - {summary or 'No summary'}")
    return "\n".join(formatted)


def _get_recent_search_sessions(conversation_id: int, limit: int = 5) -> list:
    try:
        return _conversation_manager.get_search_sessions(conversation_id, limit=limit)
    except Exception as e:
        logger.debug("Unable to fetch search sessions: %s", e)
        return []


def query_planner(state: DeepResearchState):
    """
    Enhanced query planner using the specialized SearchQueryLLM.

    This function now uses the dedicated search query generation model
    optimized for creating effective search queries.
    """
    user_query = state["user_query"]
    query_generation_count = state.get("query_generation_count", 0)
    conversation_id = _ensure_conversation_id(state.get("conversation_id"), user_query)
    conversation_context = state.get("conversation_context", "")

    built_context = _context_manager.build_context(conversation_id, user_query)
    if built_context:
        conversation_context = built_context

    is_followup = _is_follow_up(conversation_id)
    recent_sessions = _get_recent_search_sessions(conversation_id, limit=5)
    sessions_context = _format_search_session_history(recent_sessions)

    recent_sessions = _get_recent_search_sessions(conversation_id, limit=5)
    sessions_context = _format_search_session_history(recent_sessions)

    current_date = datetime.today().strftime("%d %B %Y")

    # Use the specialized SearchQueryLLM for better query generation
    search_query_llm = get_search_query_llm()

    try:
        # Generate queries using the specialized model with context
        context_segments = [
            f"Current date: {current_date}.",
            "Generate diverse queries to research this topic thoroughly.",
        ]
        if conversation_context:
            context_segments.append(f"Conversation context:\n{conversation_context}")
        if sessions_context:
            context_segments.append(f"Recent search sessions:\n{sessions_context}")
        if is_followup:
            context_segments.append("This query builds on the existing conversation; treat it as a follow-up and avoid repeating earlier research unless explicitly requested.")
        if sessions_context:
            context_segments.append(f"Recent search sessions:\n{sessions_context}")
        context = " ".join(context_segments)
        search_queries = search_query_llm.generate_initial_queries(
            user_query=user_query, max_queries=MAX_QUERY_GENERATIONS, context=context
        )

        # Validate and score each query
        validated_queries = []
        for query in search_queries:
            validation = search_query_llm.validate_query(query)
            if (
                validation["is_valid"] and validation["score"] > 0.3
            ):  # Minimum quality threshold
                validated_queries.append(query)

        # If validation filtered too many queries, use original ones
        if len(validated_queries) < 2:
            validated_queries = search_queries

        search_queries = validated_queries[:MAX_QUERY_GENERATIONS]

    except Exception as e:
        # Fallback to original method if SearchQueryLLM fails
        search_queries_response = call_llm(
            generate_search_queries_prompt.format(
                user_query=user_query,
                current_date=current_date,
                MAX_QUERY_GENERATIONS=MAX_QUERY_GENERATIONS,
            ),
            task_type=TaskType.SEARCH_QUERY_GENERATION,
        )

        search_queries = [
            q.strip().strip("'\"").strip("-").strip()
            for q in search_queries_response.split("\n")
        ]

    # Guard rails to force limit the number of queries in case llm decides to generate more
    search_queries = search_queries[:MAX_QUERY_GENERATIONS]

    consolidated_search_results = []
    for search_query in search_queries:
        search_results_per_query = get_search_results(
            search_query,
            MAX_RESULT_TO_FETCH,
            previous_search_results=consolidated_search_results,
        )
        consolidated_search_results += search_results_per_query
        _record_search_session(conversation_id, search_query, search_results_per_query)

    _store_user_query(conversation_id, user_query)

    return {
        "search_queries": search_queries,
        "search_results": consolidated_search_results,
        "query_generation_count": query_generation_count + 1,
        "conversation_id": conversation_id,
        "conversation_context": conversation_context,
        "is_followup": is_followup,
    }


def should_refine_query(
    state: DeepResearchState,
) -> Command[Literal["should_refine_query", "final_report_generator"]]:
    """
    Enhanced query refinement using the specialized SearchQueryLLM.

    This function now uses the dedicated search query model to generate
    more intelligent refinements based on search results.
    """
    user_query = state["user_query"]
    search_queries = state["search_queries"]
    search_results = state["search_results"]
    query_generation_count = state.get("query_generation_count", 0)
    conversation_id = state.get("conversation_id")
    conversation_context = state.get("conversation_context", "")

    current_date = datetime.today().strftime("%d %B %Y")

    if query_generation_count >= MAX_QUERY_REFINEMENTS:
        return Command(update={}, goto="final_report_generator")

    # Prepare search results summary for the SearchQueryLLM
    search_results_summary = "\n===========\n".join(
        f"Title: {r['title']}\nContent: {r['text_content'][:500]}..."  # Truncate for better processing
        for r in search_results[:5]  # Use only top 5 results for refinement
    )

    search_query_llm = get_search_query_llm()

    try:
        # Use the specialized SearchQueryLLM for intelligent query refinement
        refined_search_queries = search_query_llm.refine_queries(
            original_query=user_query,
            previous_queries=search_queries,
            search_results_summary=search_results_summary,
            max_queries=MAX_QUERY_GENERATIONS,
        )

        # Validate refined queries
        validated_refined_queries = []
        for query in refined_search_queries:
            validation = search_query_llm.validate_query(query)
            if (
                validation["is_valid"] and validation["score"] > 0.4
            ):  # Higher threshold for refinements
                validated_refined_queries.append(query)

        # If no good refined queries, skip refinement
        if not validated_refined_queries:
            return Command(update={}, goto="final_report_generator")

        refined_search_queries = validated_refined_queries[:MAX_QUERY_GENERATIONS]

    except Exception as e:
        # Fallback to original method if SearchQueryLLM fails
        refined_search_queries_response = call_llm(
            refine_search_queries_prompt.format(
                user_query=user_query,
                current_date=current_date,
                search_queries="\n".join(search_queries),
                search_results=search_results_summary,
                MAX_QUERY_GENERATIONS=MAX_QUERY_GENERATIONS,
            ),
            task_type=TaskType.SEARCH_QUERY_GENERATION,
        )

        refined_search_queries = [
            q.strip().strip("'\"").strip("-").strip()
            for q in refined_search_queries_response.split("\n")
            if q.lower() != "none"
        ]

    # Guard rails to force limit the number of queries
    refined_search_queries = refined_search_queries[:MAX_QUERY_GENERATIONS]

    consolidated_search_results = []
    for refined_search_query in refined_search_queries:
        search_results_per_query = get_search_results(
            refined_search_query,
            MAX_RESULT_TO_FETCH,
            search_results + consolidated_search_results,
        )
        consolidated_search_results += search_results_per_query

    is_followup = state.get("is_followup", False)
    if refined_search_queries:
        return Command(
            update={
                "search_queries": refined_search_queries,
                "search_results": consolidated_search_results,
                "query_generation_count": query_generation_count + 1,
                "conversation_id": conversation_id,
                "conversation_context": conversation_context,
                "is_followup": is_followup,
            },
            goto="should_refine_query",
        )
    else:
        return Command(update={}, goto="final_report_generator")


def _summarize_search_result(result: Dict[str, Any]) -> str:
    """Create a short summary string for a search result."""
    title = result.get("title", "Untitled Source")
    href = result.get("href", "")
    raw_content = (result.get("text_content") or result.get("body") or "").strip()
    cleaned_content = re.sub(r"\s+", " ", raw_content)
    if len(cleaned_content) > 220:
        cleaned_content = cleaned_content[:220].rsplit(" ", 1)[0] + "..."
    snippet = cleaned_content or "Summary not available."
    reference = f"[{title}]({href})" if href else title
    return f"{reference}: {snippet}"


def final_report_generator(state: DeepResearchState):
    """
    Enhanced final report generator using the specialized ThinkingLLM.

    This function now uses the dedicated thinking model for better
    report generation and reasoning.
    """
    user_query = state["user_query"]
    search_results = state["search_results"]

    search_results_str = "\n===========\n".join(
        f"Title: {r.get('title', 'Untitled')}\nLink:{r.get('href', '')}\nContent: {r.get('text_content', '')}"
        for r in search_results
    )

    search_session_history = []
    conversation_id = state.get("conversation_id")
    if conversation_id:
        sessions = _conversation_manager.get_search_sessions(conversation_id, limit=5)
        search_session_history = [
            _conversation_manager.search_session_to_dict(session) for session in sessions
        ]

    session_context = ""
    if search_session_history:
        session_context = "\n".join(
            f"{entry['created_at']}: {entry['search_query']} ({entry['result_count']} results) - {entry['summary'] or 'No summary'}"
            for entry in search_session_history
        )
        if session_context:
            search_results_str = f"Previous search sessions:\n{session_context}\n\n{search_results_str}"

    thinking_llm = None
    try:
        thinking_llm = get_thinking_llm()
    except Exception as e:
        logger.error("Failed to initialize ThinkingLLM: %s", e)

    analysis_result: Dict[str, Any] = {}
    analysis_summary = ""
    if thinking_llm:
        try:
            analysis_result = thinking_llm.analyze_research_findings(
                research_data=search_results_str,
                user_query=user_query,
                context="Search Results:\n" + search_results_str,
            )
            analysis_summary = analysis_result.get("analysis", "").strip()
        except Exception as e:
            logger.warning("Failed to analyze research findings: %s", e)

    final_report = ""
    if thinking_llm:
        try:
            final_report = thinking_llm.generate_comprehensive_report(
                user_query=user_query,
                research_findings=search_results_str,
                analysis_results=analysis_result or None,
            )
        except Exception as e:
            logger.warning("ThinkingLLM report generation failed: %s", e)

    if not final_report:
        try:
            final_report = call_thinking_llm(
                prompt=final_report_prompt.format(
                    user_query=user_query, search_results=search_results_str
                ),
                task="report",
                context=f"User Query: {user_query}",
            )
        except Exception as e:
            logger.error(
                "Fallback thinking report generation failed: %s", e
            )
            final_report = (
                "Report generation failed due to repeated errors. "
                "Please try again later."
            )

    if not analysis_summary and final_report:
        analysis_summary = final_report.split("\n")[0].strip()

    reasoning_trace: list = []
    if thinking_llm:
        try:
            reasoning_result = thinking_llm.reason_step_by_step(
                problem=user_query,
                context=search_results_str,
            )
            reasoning_trace = reasoning_result.get("reasoning_steps", []) or []
        except Exception as e:
            logger.warning("Failed to generate reasoning trace: %s", e)

    confidence_scores = []
    if thinking_llm:
        try:
            confidence_scores = thinking_llm.assess_claim_confidence(
                report_markdown=final_report,
                search_results=search_results,
                user_query=user_query,
            )
        except Exception as e:
            logger.warning("Failed to generate confidence scores: %s", e)

    if state.get("conversation_id"):
        _store_assistant_report(state["conversation_id"], final_report)
        _store_reasoning_trace(state["conversation_id"], reasoning_trace)

    summaries = [
        _summarize_search_result(result)
        for result in search_results[:5]
    ]

    return {
        "individual_page_summaries": summaries,
        "report_markdown": final_report,
        "claim_confidences": confidence_scores,
        "reasoning_trace": reasoning_trace,
        "analysis_summary": analysis_summary,
        "analysis_results": analysis_result or {},
        "search_session_history": search_session_history,
        "is_followup": state.get("is_followup", False),
    }


def build_graph():
    graph_builder = StateGraph(DeepResearchState)

    graph_builder.add_node(query_planner)
    graph_builder.add_node(should_refine_query)
    graph_builder.add_node(final_report_generator)

    graph_builder.add_edge(START, "query_planner")
    graph_builder.add_edge("query_planner", "should_refine_query")
    graph_builder.add_edge("final_report_generator", END)

    graph = graph_builder.compile()

    # Generate the graph image and save it
    graph_image_path = "graph.png"
    with open(graph_image_path, "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

    return graph


if __name__ == "__main__":
    graph = build_graph()
    for chunk in graph.stream(
        {
            "user_query": "have we reached agi with manus ai?",
        }
    ):
        print(chunk)
