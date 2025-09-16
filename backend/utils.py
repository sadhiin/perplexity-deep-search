import os
import logging
from typing import Optional, List, Dict, Any
from groq import Groq
import bs4, requests
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# Import our new model management system
from config import TaskType, get_config
from models.model_manager import ModelManager
from models.search_query_llm import SearchQueryLLM
from models.thinking_llm import ThinkingLLM

load_dotenv()

# Initialize components
logger = logging.getLogger(__name__)
_model_manager = None
_search_query_llm = None
_thinking_llm = None

# Backward compatibility - maintain the original Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def get_model_manager() -> ModelManager:
    """Get or create the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def get_search_query_llm() -> SearchQueryLLM:
    """Get or create the global search query LLM instance."""
    global _search_query_llm
    if _search_query_llm is None:
        _search_query_llm = SearchQueryLLM(get_model_manager())
    return _search_query_llm


def get_thinking_llm() -> ThinkingLLM:
    """Get or create the global thinking LLM instance."""
    global _thinking_llm
    if _thinking_llm is None:
        _thinking_llm = ThinkingLLM(get_model_manager())
    return _thinking_llm


def call_llm(prompt: str, model: str = "llama-3.3-70b-versatile", task_type: Optional[TaskType] = None) -> str:
    """
    Call an LLM with the given prompt.

    This function provides backward compatibility while supporting the new multi-model system.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (for backward compatibility)
        task_type: Optional task type for intelligent model selection

    Returns:
        String response from the LLM
    """
    try:
        # If task_type is specified, use the new model management system
        if task_type:
            model_manager = get_model_manager()
            llm_model = model_manager.get_model_for_task(task_type)

            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=prompt)]
            response = llm_model.invoke(messages)
            return response.content

        # Backward compatibility: use the original Groq client
        if not client.api_key:
            logger.warning("GROQ_API_KEY not found, falling back to new model system")
            # Fallback to new system
            model_manager = get_model_manager()
            available_models = model_manager.list_available_models()
            if available_models:
                llm_model = model_manager.get_model(available_models[0])
                from langchain_core.messages import SystemMessage
                messages = [SystemMessage(content=prompt)]
                response = llm_model.invoke(messages)
                return response.content
            else:
                raise Exception("No available models found")

        response = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        # Try fallback with new model system
        try:
            model_manager = get_model_manager()
            available_models = model_manager.list_available_models()
            if available_models:
                fallback_model = model_manager.get_model(available_models[0])
                from langchain_core.messages import SystemMessage
                messages = [SystemMessage(content=prompt)]
                response = fallback_model.invoke(messages)
                return response.content
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            raise Exception(f"All LLM calls failed. Original error: {e}, Fallback error: {fallback_error}")


def call_search_query_llm(user_query: str, max_queries: int = 3, context: Optional[str] = None) -> List[str]:
    """
    Generate search queries using the specialized search query LLM.

    Args:
        user_query: The user's research question
        max_queries: Maximum number of queries to generate
        context: Optional context for query generation

    Returns:
        List of search query strings
    """
    try:
        search_llm = get_search_query_llm()
        return search_llm.generate_initial_queries(user_query, max_queries, context)
    except Exception as e:
        logger.error(f"Error generating search queries: {e}")
        # Simple fallback
        return [user_query]


def call_thinking_llm(
    prompt: str,
    task: str = "analysis",
    context: Optional[str] = None
) -> str:
    """
    Call the thinking LLM for complex reasoning tasks.

    Args:
        prompt: The prompt or question
        task: Type of task (analysis, report, chat)
        context: Optional additional context

    Returns:
        Response from the thinking LLM
    """
    try:
        thinking_llm = get_thinking_llm()

        if task == "analysis":
            result = thinking_llm.analyze_research_findings(prompt, prompt, context)
            return result.get("analysis", "Analysis failed")
        elif task == "report":
            return thinking_llm.generate_comprehensive_report(prompt, prompt)
        elif task == "chat":
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            return thinking_llm.generate_chat_response(messages, context)
        else:
            # Default to step-by-step reasoning
            result = thinking_llm.reason_step_by_step(prompt, context)
            return result.get("full_reasoning", "Reasoning failed")

    except Exception as e:
        logger.error(f"Error with thinking LLM: {e}")
        # Fallback to regular LLM call
        return call_llm(prompt, task_type=TaskType.THINKING_REASONING)


def get_search_results(query, max_results, previous_search_results=[]):
    # Extract hrefs from previous_search_results for comparison
    visited_hrefs = {result["href"] for result in previous_search_results}

    # Retrieve max_results + len(previous_search_results) to account for duplicates
    retrieved_search_results = DDGS().text(
        query, max_results=max_results + len(previous_search_results)
    )

    # Filter out already visited pages - worst case we will still get max_results if all are visited
    new_search_results = [
        result
        for result in retrieved_search_results
        if result["href"] not in visited_hrefs
    ]

    # Limit the max results after filtering to max_results and some minor room - this is needed as we get many results if len of previous_search_results is high
    new_search_results = new_search_results[: max_results + 1]

    # Can make this loop execute in parallel for faster results
    for retrieved_search_result in new_search_results:
        try:
            response = requests.get(
                retrieved_search_result["href"], headers={"User-Agent": "Mozilla/5.0"}
            )
            soup = bs4.BeautifulSoup(response.text, "lxml")
            text_content = soup.body.get_text(" ", strip=True)
            # Limit the text content in case we encounter a large page
            retrieved_search_result["text_content"] = text_content[:2000]
        except:
            retrieved_search_result["text_content"] = retrieved_search_result["body"]

    return new_search_results