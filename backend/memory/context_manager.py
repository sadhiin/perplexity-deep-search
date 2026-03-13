from typing import List, Optional

from backend.models.thinking_llm import ThinkingLLM
from backend.memory.conversation_manager import ConversationManager


class ContextManager:
    """Maintains sliding-window context for a conversation."""

    def __init__(
        self,
        conversation_manager: Optional[ConversationManager] = None,
        max_history: int = 20,
        max_chars: int = 1800,
        summarizer: Optional[ThinkingLLM] = None,
    ):
        self.conversation_manager = conversation_manager or ConversationManager()
        self.max_history = max_history
        self.max_chars = max_chars
        self.summarizer = summarizer or ThinkingLLM()

    def build_context(
        self, conversation_id: int, user_query: Optional[str] = None
    ) -> str:
        """Return a concatenated context string capped by message count and character budget."""
        messages = self.conversation_manager.get_messages(
            conversation_id, limit=200, offset=0
        )
        if not messages:
            return ""

        selected_messages = self._select_relevant_messages(messages, user_query)
        context_lines = []
        total_chars = 0
        for message in selected_messages:
            line = f"{message.role}: {message.content.strip()}"
            if not line:
                continue
            if total_chars + len(line) > self.max_chars:
                break
            context_lines.append(line)
            total_chars += len(line)

        context_lines = list(reversed(context_lines))
        needs_summary = len(selected_messages) > self.max_history or total_chars >= self.max_chars
        if needs_summary:
            summary_line = self._summarize_messages(selected_messages)
            if summary_line:
                context_lines.insert(0, f"Conversation summary: {summary_line}")
        return "\n".join(context_lines)

    def _summarize_messages(self, messages: List) -> str:
        if not messages:
            return ""

        summary_messages = messages[: max(1, len(messages) // 2)]
        research_data = "\n".join(
            f"{message.role}: {message.content.strip()}"
            for message in summary_messages
            if message.content.strip()
        )
        if not research_data:
            return ""

        try:
            analysis = self.summarizer.analyze_research_findings(
                research_data=research_data,
                user_query="Conversation history summary",
            )
            summary_text = analysis.get("analysis", "").strip()
            if summary_text:
                return summary_text.split("\n")[0]
        except Exception:
            pass

        return self._fallback_summary(summary_messages)

    def _fallback_summary(self, messages: List) -> str:
        lines = []
        for message in messages[-3:]:
            content_snippet = message.content.strip().split(".")[0]
            if content_snippet:
                lines.append(f"{message.role}: {content_snippet.strip()}")
        return " | ".join(lines)

    def _select_relevant_messages(
        self, messages, user_query: Optional[str]
    ) -> List:
        """Score and select messages to respect the sliding window."""
        scored = []
        for message in messages:
            score = self._score_message_relevance(message.content, user_query)
            scored.append((score, message))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [item[1] for item in scored[: self.max_history]]
        selected.sort(key=lambda message: message.created_at)
        return selected

    def _score_message_relevance(
        self, message_content: str, user_query: Optional[str]
    ) -> float:
        """Simple keyword overlap scoring to keep relevant context."""
        if not message_content.strip():
            return 0.0
        if not user_query:
            return 1.0
        lower_content = message_content.lower()
        lower_query = user_query.lower()
        overlap = sum(
            1
            for token in lower_query.split()
            if token and token in lower_content
        )
        return 1.0 + overlap / (len(lower_query.split()) or 1)
