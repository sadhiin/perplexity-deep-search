"""
Thinking LLM - Specialized model for deep reasoning and analysis.

This module provides a dedicated interface for complex reasoning tasks using
models like DeepSeek R1, with support for chain-of-thought reasoning and
structured thinking processes.
"""

import logging
import json
from typing import List, Optional, Dict, Any, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from backend.config import TaskType
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class ThinkingLLM:
    """
    Specialized LLM interface for deep reasoning and analysis.

    This class is optimized for complex reasoning tasks using models like
    DeepSeek R1, with support for chain-of-thought reasoning and structured analysis.
    """

    THINKING_SYSTEM_PROMPT = """You are a deep reasoning AI assistant specialized in thorough analysis and structured thinking. Your approach should be:

1. ANALYTICAL: Break down complex topics into components
2. COMPREHENSIVE: Consider multiple perspectives and angles
3. EVIDENCE-BASED: Ground reasoning in facts and data
4. STRUCTURED: Present findings in clear, logical organization
5. CRITICAL: Question assumptions and identify limitations
6. INSIGHTFUL: Provide meaningful conclusions and implications

When analyzing information:
- Start with key insights and main conclusions
- Support claims with evidence from the provided information
- Identify patterns, relationships, and contradictions
- Consider alternative interpretations where relevant
- Highlight areas of uncertainty or missing information
- Provide actionable insights when possible

Your goal is to provide deep, thoughtful analysis that goes beyond surface-level summary."""

    REPORT_GENERATION_SYSTEM_PROMPT = """You are an expert research analyst tasked with creating comprehensive, well-structured reports. Your reports should be:

STRUCTURE:
- Clear executive summary with key findings
- Logical organization with appropriate headings
- Evidence-based conclusions with supporting data
- Identification of knowledge gaps or limitations

ANALYSIS DEPTH:
- Multi-perspective examination of the topic
- Synthesis of information from multiple sources
- Critical evaluation of evidence quality
- Pattern recognition across different data points

PRESENTATION:
- Professional, accessible language
- Appropriate use of bullet points and formatting
- Clear transitions between sections
- Balanced coverage of different aspects

Focus on providing actionable insights and drawing meaningful connections between different pieces of information."""

    CHAT_RESPONSE_SYSTEM_PROMPT = """You are a thoughtful, knowledgeable AI assistant engaged in conversation. Your responses should be:

CONVERSATIONAL QUALITIES:
- Natural and engaging while maintaining depth
- Responsive to the specific question or context
- Building on previous conversation when relevant
- Acknowledging uncertainty when appropriate

THINKING APPROACH:
- Consider the user's perspective and needs
- Provide reasoning behind your responses
- Offer multiple viewpoints when relevant
- Ask clarifying questions when helpful

INFORMATION HANDLING:
- Draw from provided context and research
- Distinguish between different types of information
- Provide appropriate level of detail for the context
- Suggest follow-up topics or questions when useful

Maintain a balance between being informative and conversational, ensuring responses are both accurate and engaging."""

    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        Initialize the thinking LLM.

        Args:
            model_manager: Optional model manager instance
        """
        self.model_manager = model_manager or ModelManager()

    def _get_model(self, task_type: TaskType = TaskType.THINKING_REASONING) -> BaseChatModel:
        """Fetch a model targeted for the given thinking task."""
        return self.model_manager.get_model_for_task(task_type)

    def analyze_research_findings(
        self,
        research_data: str,
        user_query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform deep analysis of research findings.

        Args:
            research_data: Raw research data to analyze
            user_query: Original user query for context
            context: Optional additional context

        Returns:
            Dictionary containing structured analysis results
        """
        try:
            prompt = f"""Original Query: {user_query}

Research Data to Analyze:
{research_data}"""

            if context:
                prompt += f"\n\nAdditional Context: {context}"

            prompt += """

Please provide a comprehensive analysis including:
1. Key insights and main findings
2. Important patterns or relationships
3. Critical evaluation of the information
4. Areas of uncertainty or gaps
5. Implications and actionable insights

Structure your response with clear headings and bullet points for readability."""

            messages = [
                SystemMessage(content=self.THINKING_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            model = self._get_model(TaskType.THINKING_REASONING)
            response = model.invoke(messages)

            # Parse the structured response
            analysis = self._parse_analysis_response(response.content)

            logger.info("Completed deep analysis of research findings")
            return analysis

        except Exception as e:
            logger.error(f"Error in research analysis: {e}")
            return {
                "analysis": response.content if 'response' in locals() else "Analysis failed",
                "key_insights": [],
                "limitations": ["Analysis encountered errors"],
                "confidence": 0.5
            }

    def generate_comprehensive_report(
        self,
        user_query: str,
        research_findings: str,
        analysis_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a comprehensive report based on research and analysis.

        Args:
            user_query: Original user query
            research_findings: Collected research data
            analysis_results: Optional analysis results from previous step

        Returns:
            Comprehensive markdown report
        """
        try:
            prompt = f"""Research Question: {user_query}

Research Findings:
{research_findings}"""

            if analysis_results:
                prompt += f"\n\nPrevious Analysis Results:\n{analysis_results.get('analysis', '')}"

            prompt += """

Create a comprehensive, well-structured report in markdown format that:
1. Provides an executive summary of key findings
2. Presents detailed analysis organized by themes
3. Includes evidence-based conclusions
4. Identifies limitations and areas for further research
5. Offers actionable insights where applicable

Use proper markdown formatting with headings, bullet points, and emphasis where appropriate."""

            messages = [
                SystemMessage(content=self.REPORT_GENERATION_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            model = self._get_model(TaskType.REPORT_GENERATION)
            response = model.invoke(messages)

            logger.info("Generated comprehensive report")
            return response.content

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"# Research Report\n\nError generating report: {e}\n\nOriginal query: {user_query}"

    def generate_chat_response(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        context: Optional[str] = None
    ) -> str:
        """
        Generate a thoughtful chat response with reasoning.

        Args:
            messages: Conversation history
            context: Optional additional context from research

        Returns:
            Chat response string
        """
        try:
            # Prepare the conversation with system prompt
            conversation_messages = [SystemMessage(content=self.CHAT_RESPONSE_SYSTEM_PROMPT)]

            # Add context if available
            if context:
                context_message = SystemMessage(
                    content=f"Additional Context from Research:\n{context}"
                )
                conversation_messages.append(context_message)

            # Add conversation history
            conversation_messages.extend(messages)

            model = self._get_model(TaskType.CHAT_RESPONSE)
            response = model.invoke(conversation_messages)

            logger.info("Generated chat response with reasoning")
            return response.content

        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return "I apologize, but I encountered an error while processing your request. Could you please try again?"

    def evaluate_information_quality(
        self,
        information: str,
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the quality and reliability of information.

        Args:
            information: Information to evaluate
            criteria: Optional specific criteria to evaluate against

        Returns:
            Dictionary with quality assessment
        """
        try:
            default_criteria = [
                "Accuracy and factual correctness",
                "Source reliability and credibility",
                "Completeness and comprehensiveness",
                "Recency and relevance",
                "Objectivity and bias assessment"
            ]

            eval_criteria = criteria or default_criteria
            criteria_text = "\n".join(f"- {criterion}" for criterion in eval_criteria)

            prompt = f"""Information to Evaluate:
{information}

Evaluation Criteria:
{criteria_text}

Please provide a detailed quality assessment including:
1. Overall quality score (1-10)
2. Strengths of the information
3. Weaknesses or limitations
4. Reliability assessment
5. Recommendations for improvement

Be critical but fair in your evaluation."""

            messages = [
                SystemMessage(content=self.THINKING_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            model = self._get_model(TaskType.THINKING_REASONING)
            response = model.invoke(messages)

            # Parse evaluation results
            evaluation = self._parse_evaluation_response(response.content)

            logger.info("Completed information quality evaluation")
            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating information quality: {e}")
            return {
                "quality_score": 5.0,
                "strengths": [],
                "weaknesses": ["Evaluation failed"],
                "reliability": "Unknown",
                "recommendations": ["Re-evaluate manually"]
            }

    def reason_step_by_step(
        self,
        problem: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform step-by-step reasoning on a complex problem.

        Args:
            problem: Problem or question to reason through
            context: Optional additional context

        Returns:
            Dictionary with reasoning steps and conclusion
        """
        try:
            prompt = f"""Problem to Solve: {problem}"""

            if context:
                prompt += f"\n\nContext: {context}"

            prompt += """

Please work through this step-by-step using clear reasoning:

1. First, break down the problem into its key components
2. Identify what information we have and what we need
3. Consider different approaches or perspectives
4. Work through the logic systematically
5. Arrive at a well-reasoned conclusion
6. Identify any assumptions or limitations

Show your thinking process clearly at each step."""

            messages = [
                SystemMessage(content=self.THINKING_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            model = self._get_model(TaskType.THINKING_REASONING)
            response = model.invoke(messages)

            # Parse reasoning steps
            reasoning = self._parse_reasoning_response(response.content)

            logger.info("Completed step-by-step reasoning")
            return reasoning

        except Exception as e:
            logger.error(f"Error in step-by-step reasoning: {e}")
            return {
                "reasoning_steps": [f"Error occurred: {e}"],
                "conclusion": "Unable to complete reasoning",
                "confidence": 0.0,
                "assumptions": [],
                "limitations": ["Reasoning process failed"]
            }

    def assess_claim_confidence(
        self,
        report_markdown: str,
        search_results: List[Dict[str, Any]],
        user_query: Optional[str] = None,
        max_claims: int = 6,
    ) -> List[Dict[str, Any]]:
        """Score confidence for major claims in a report."""
        try:
            if not report_markdown.strip():
                return []

            evidence_chunks = []
            for result in search_results:
                title = result.get("title", "Unknown Source")
                link = result.get("href", "")
                snippet = (result.get("text_content") or "")[:400]
                evidence_chunks.append(f"{title} ({link})\n{snippet}")

            evidence_summary = "\n\n".join(evidence_chunks[:8])

            prompt = f"""You are verifying the reliability of a research report.

Original Query: {user_query or 'Unknown'}

Research Report:
{report_markdown}

Supporting Evidence:
{evidence_summary}

Task: Extract up to {max_claims} of the most important claims or findings from the report and rate confidence that each claim is fully supported by the evidence. Only use the provided evidence and report content.

Return your analysis strictly as valid JSON formatted as:
[
  {{
    "claim": "short statement",
    "confidence_score": 0.0-1.0,
    "evidence": ["cite key supporting facts"],
    "rationale": "why this score"
  }}
]

- Use decimal confidence scores between 0 and 1.
- Include at least one piece of supporting evidence per claim when possible.
- Focus on the most decision-relevant claims.
"""

            messages = [
                SystemMessage(content=self.THINKING_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            model = self._get_model(TaskType.THINKING_REASONING)
            response = model.invoke(messages)

            return self._parse_confidence_scores(response.content)

        except Exception as e:
            logger.error(f"Error assessing claim confidence: {e}")
            return []

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse structured analysis response."""
        # Simple parsing - in production, could use more sophisticated NLP
        return {
            "analysis": response,
            "key_insights": self._extract_bullet_points(response, "insights"),
            "limitations": self._extract_bullet_points(response, "limitations"),
            "confidence": 0.8  # Default confidence
        }

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse quality evaluation response."""
        import re

        # Extract quality score
        score_match = re.search(r'score.*?(\d+(?:\.\d+)?)', response.lower())
        quality_score = float(score_match.group(1)) if score_match else 5.0

        return {
            "quality_score": quality_score,
            "evaluation": response,
            "strengths": self._extract_bullet_points(response, "strengths"),
            "weaknesses": self._extract_bullet_points(response, "weaknesses"),
            "reliability": "Moderate",  # Default
            "recommendations": self._extract_bullet_points(response, "recommendations")
        }

    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """Parse step-by-step reasoning response."""
        # Extract numbered steps
        import re
        steps = re.findall(r'\d+\.\s*([^0-9]+?)(?=\d+\.|$)', response, re.DOTALL)

        return {
            "reasoning_steps": [step.strip() for step in steps],
            "full_reasoning": response,
            "confidence": 0.8,
            "assumptions": self._extract_bullet_points(response, "assumptions"),
            "limitations": self._extract_bullet_points(response, "limitations")
        }

    def _parse_confidence_scores(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON confidence scores from the model response."""
        try:
            json_text = self._extract_json_block(response)
            data = json.loads(json_text)
        except Exception as e:
            logger.error(f"Failed to parse confidence scores: {e}")
            return []

        if isinstance(data, dict):
            data = data.get("claims") or data.get("items") or [data]

        normalized_scores = []
        for entry in data or []:
            if not isinstance(entry, dict):
                continue
            claim = entry.get("claim") or entry.get("statement") or ""
            if not claim:
                continue
            score = entry.get("confidence_score", entry.get("confidence"))
            score = self._normalize_confidence_score(score)
            level = entry.get("confidence_level") or self._score_to_level(score)
            evidence = entry.get("evidence") or []
            if isinstance(evidence, str):
                evidence = [evidence]
            rationale = entry.get("rationale") or entry.get("justification") or ""

            normalized_scores.append({
                "claim": claim.strip(),
                "confidence_score": score,
                "confidence_level": level,
                "evidence": [e.strip() for e in evidence if e],
                "rationale": rationale.strip(),
            })

        return normalized_scores

    def _extract_bullet_points(self, text: str, section: str) -> List[str]:
        """Extract bullet points from a specific section."""
        import re

        # Find section and extract bullet points
        section_pattern = rf'{section}.*?:\s*(.*?)(?=\n\n|\n[A-Z]|$)'
        section_match = re.search(section_pattern, text, re.IGNORECASE | re.DOTALL)

        if not section_match:
            return []

        section_text = section_match.group(1)
        bullets = re.findall(r'[-•*]\s*([^\n]+)', section_text)

        return [bullet.strip() for bullet in bullets]

    def _extract_json_block(self, text: str) -> str:
        """Extract the first JSON array/object from text."""
        import re

        code_block = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        start = text.find("[")
        if start != -1:
            end = text.rfind("]")
            if end != -1 and end > start:
                return text[start:end + 1]

        start = text.find("{")
        if start != -1:
            end = text.rfind("}")
            if end != -1 and end > start:
                return text[start:end + 1]

        raise ValueError("No JSON content found in response")

    def _normalize_confidence_score(self, value: Any) -> float:
        """Normalize arbitrary input into 0-1 range."""
        try:
            score = float(value)
        except Exception:
            return 0.5

        if score > 1:
            score = score / 100 if score <= 100 else 1.0
        if score < 0:
            score = 0.0
        return min(score, 1.0)

    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to descriptive label."""
        if score >= 0.75:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current thinking model."""
        return self.model_manager.get_model_info(
            self.model_manager.config_manager.config.get_model_for_task(
                TaskType.THINKING_REASONING
            )
        )
