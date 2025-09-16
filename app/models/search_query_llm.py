"""
Search Query LLM - Specialized model for generating and refining search queries.

This module provides a dedicated interface for search query generation with
optimized prompting, fast response times, and query validation.
"""

import logging
from typing import List, Optional, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from config import TaskType
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class SearchQueryLLM:
    """
    Specialized LLM interface for search query generation and refinement.

    This class is optimized for fast, efficient search query generation using
    lightweight models like GPT-4o-mini or Llama-3.1-8B.
    """

    SEARCH_QUERY_SYSTEM_PROMPT = """You are a search query generation specialist. Your task is to create effective search queries that will retrieve the most relevant information for research purposes.

Guidelines for generating search queries:
1. Be specific and focused - avoid overly broad terms
2. Use relevant keywords and synonyms
3. Consider different perspectives and aspects of the topic
4. Prioritize recent information when applicable
5. Generate queries that complement each other rather than duplicate
6. Keep queries concise but descriptive
7. Use quoted phrases for exact matches when needed
8. Consider domain-specific terminology

Output format: Return each search query on a new line, no numbering or formatting."""

    QUERY_REFINEMENT_SYSTEM_PROMPT = """You are a search query refinement specialist. Based on initial search results, generate refined queries to fill information gaps and explore different angles.

Your refinement strategy should:
1. Identify missing information from the initial results
2. Generate queries for unexplored aspects
3. Use different terminology or synonyms
4. Target specific data, statistics, or recent developments
5. Avoid duplicating successful queries from previous searches
6. Focus on filling knowledge gaps

Output format: Return each refined search query on a new line, no numbering or formatting."""

    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        Initialize the search query LLM.

        Args:
            model_manager: Optional model manager instance
        """
        self.model_manager = model_manager or ModelManager()
        self.model: BaseChatModel = self.model_manager.get_model_for_task(
            TaskType.SEARCH_QUERY_GENERATION
        )

    def generate_initial_queries(
        self,
        user_query: str,
        max_queries: int = 3,
        context: Optional[str] = None
    ) -> List[str]:
        """
        Generate initial search queries for a user's research question.

        Args:
            user_query: The user's original research question
            max_queries: Maximum number of queries to generate
            context: Optional context to help with query generation

        Returns:
            List of search query strings
        """
        try:
            prompt = f"""Research Question: {user_query}

Generate {max_queries} diverse and effective search queries to research this topic thoroughly."""

            if context:
                prompt += f"\n\nAdditional Context: {context}"

            messages = [
                SystemMessage(content=self.SEARCH_QUERY_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            response = self.model.invoke(messages)
            queries = self._parse_query_response(response.content, max_queries)

            logger.info(f"Generated {len(queries)} initial search queries")
            return queries

        except Exception as e:
            logger.error(f"Error generating initial queries: {e}")
            # Fallback to basic query generation
            return [user_query]

    def refine_queries(
        self,
        original_query: str,
        previous_queries: List[str],
        search_results_summary: str,
        max_queries: int = 3
    ) -> List[str]:
        """
        Generate refined search queries based on initial results.

        Args:
            original_query: The original user query
            previous_queries: Previously generated queries
            search_results_summary: Summary of results from previous queries
            max_queries: Maximum number of refined queries to generate

        Returns:
            List of refined search query strings
        """
        try:
            prompt = f"""Original Research Question: {original_query}

Previous Search Queries:
{chr(10).join(f"- {query}" for query in previous_queries)}

Search Results Summary:
{search_results_summary}

Based on the results so far, generate {max_queries} refined search queries to fill information gaps and explore different angles."""

            messages = [
                SystemMessage(content=self.QUERY_REFINEMENT_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            response = self.model.invoke(messages)
            refined_queries = self._parse_query_response(response.content, max_queries)

            logger.info(f"Generated {len(refined_queries)} refined search queries")
            return refined_queries

        except Exception as e:
            logger.error(f"Error refining queries: {e}")
            # Fallback to variations of original query
            return [f"{original_query} recent developments", f"{original_query} 2024"]

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate and assess the quality of a search query.

        Args:
            query: The search query to validate

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "score": 0.0,
            "issues": [],
            "suggestions": []
        }

        # Basic validation rules
        if len(query.strip()) < 3:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Query too short")
            return validation_result

        if len(query) > 200:
            validation_result["issues"].append("Query might be too long")
            validation_result["suggestions"].append("Consider splitting into multiple queries")

        # Score based on various factors
        score = 50  # Base score

        # Length scoring
        if 10 <= len(query) <= 100:
            score += 20

        # Keyword diversity (simple check for multiple words)
        words = query.split()
        if len(words) >= 3:
            score += 15

        # Contains specific terms
        specific_terms = ["recent", "2024", "2023", "latest", "current"]
        if any(term in query.lower() for term in specific_terms):
            score += 10

        # Avoid overly common words
        common_words = ["the", "and", "or", "but", "very", "really"]
        if not all(word.lower() in common_words for word in words[:3]):
            score += 5

        validation_result["score"] = min(score, 100) / 100.0

        return validation_result

    def _parse_query_response(self, response: str, max_queries: int) -> List[str]:
        """
        Parse the LLM response to extract search queries.

        Args:
            response: Raw response from the LLM
            max_queries: Maximum number of queries to return

        Returns:
            List of cleaned search queries
        """
        # Split by lines and clean up
        lines = response.strip().split('\n')
        queries = []

        for line in lines:
            # Remove common prefixes and numbering
            cleaned = line.strip()

            # Remove numbering (1., 2., etc.)
            import re
            cleaned = re.sub(r'^\d+\.\s*', '', cleaned)

            # Remove bullet points
            cleaned = re.sub(r'^[-•*]\s*', '', cleaned)

            # Remove quotes if they wrap the entire query
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]

            # Skip empty lines and very short queries
            if len(cleaned.strip()) >= 3:
                queries.append(cleaned.strip())

        # Return up to max_queries
        return queries[:max_queries]

    def generate_follow_up_queries(
        self,
        main_topic: str,
        discovered_subtopics: List[str],
        max_queries: int = 3
    ) -> List[str]:
        """
        Generate follow-up queries based on discovered subtopics.

        Args:
            main_topic: The main research topic
            discovered_subtopics: Subtopics discovered during research
            max_queries: Maximum number of follow-up queries

        Returns:
            List of follow-up search queries
        """
        try:
            subtopics_text = ", ".join(discovered_subtopics)

            prompt = f"""Main Research Topic: {main_topic}

Discovered Subtopics: {subtopics_text}

Generate {max_queries} targeted follow-up search queries to explore these subtopics more deeply."""

            messages = [
                SystemMessage(content=self.SEARCH_QUERY_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            response = self.model.invoke(messages)
            follow_up_queries = self._parse_query_response(response.content, max_queries)

            logger.info(f"Generated {len(follow_up_queries)} follow-up queries")
            return follow_up_queries

        except Exception as e:
            logger.error(f"Error generating follow-up queries: {e}")
            # Fallback to subtopic-based queries
            return [f"{main_topic} {subtopic}" for subtopic in discovered_subtopics[:max_queries]]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model being used."""
        return self.model_manager.get_model_info(
            self.model_manager.config.get_model_for_task(TaskType.SEARCH_QUERY_GENERATION)
        )