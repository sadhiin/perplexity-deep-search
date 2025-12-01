from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from backend.database import (
    Conversation,
    Message,
    SearchSession,
    get_db_session,
    init_db,
)
from backend.models.thinking_llm import ThinkingLLM


class ConversationManager:
    """Handles persistence of conversations, messages, and search sessions."""

    def __init__(
        self,
        summarizer: Optional[ThinkingLLM] = None,
        summary_threshold: int = 6,
    ) -> None:
        init_db()
        self.summarizer = summarizer or ThinkingLLM()
        self.summary_threshold = summary_threshold

    def create_conversation(
        self, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        metadata = metadata or {}
        with get_db_session() as session:
            conversation = Conversation(title=title, metadata=metadata)
            session.add(conversation)
            session.flush()
            session.refresh(conversation)
            return conversation

    def list_conversations(
        self, limit: int = 20, offset: int = 0
    ) -> List[Conversation]:
        with get_db_session() as session:
            statement = select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit).offset(offset)
            return session.scalars(statement).all()

    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        with get_db_session() as session:
            statement = select(Conversation).where(Conversation.id == conversation_id)
            return session.scalar(statement)

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        message_type: str = "research",
    ) -> Message:
        with get_db_session() as session:
            conversation = session.get(Conversation, conversation_id)
            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                message_type=message_type,
            )
            session.add(message)
            session.flush()
            session.refresh(message)
            self._maybe_update_summary(session, conversation)
            return message

    def get_messages(
        self, conversation_id: int, limit: int = 50, offset: int = 0
    ) -> List[Message]:
        with get_db_session() as session:
            statement = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
                .offset(offset)
            )
            return session.scalars(statement).all()

    def record_search_session(
        self,
        conversation_id: int,
        search_query: str,
        result_count: int = 0,
        summary: Optional[str] = None,
    ) -> SearchSession:
        with get_db_session() as session:
            conversation = session.get(Conversation, conversation_id)
            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")
            session_record = SearchSession(
                conversation_id=conversation_id,
                search_query=search_query,
                result_count=result_count,
                summary=summary,
            )
            session.add(session_record)
            session.flush()
            session.refresh(session_record)
            return session_record

    def summarize_conversation(
        self, conversation_id: int, limit: int = 10
    ) -> Optional[str]:
        with get_db_session() as session:
            conversation = session.get(Conversation, conversation_id)
            if not conversation:
                return None
            messages = self._fetch_recent_messages(session, conversation_id, limit=limit)
            if not messages:
                return None
            summary = self._generate_summary(conversation, messages)
            if summary:
                conversation.metadata = conversation.metadata or {}
                conversation.metadata["summary"] = summary
                session.add(conversation)
            return summary

    def get_search_sessions(
        self, conversation_id: int, limit: int = 20, offset: int = 0
    ) -> List[SearchSession]:
        with get_db_session() as session:
            statement = (
                select(SearchSession)
                .where(SearchSession.conversation_id == conversation_id)
                .order_by(SearchSession.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return session.scalars(statement).all()

    def _maybe_update_summary(
        self, session, conversation: Conversation
    ) -> None:
        if not self.summary_threshold:
            return
        messages = self._fetch_recent_messages(
            session, conversation.id, limit=self.summary_threshold
        )
        if len(messages) < self.summary_threshold:
            return
        summary = self._generate_summary(conversation, messages)
        if summary:
            conversation.metadata = conversation.metadata or {}
            conversation.metadata["summary"] = summary
            session.add(conversation)

    def _fetch_recent_messages(
        self, session, conversation_id: int, limit: int = 20
    ) -> List[Message]:
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        return list(reversed(session.scalars(statement).all()))

    def _generate_summary(
        self, conversation: Conversation, messages: List[Message]
    ) -> Optional[str]:
        if not messages:
            return None
        research_data = "\n".join(
            f"{message.role}: {message.content.strip()}" for message in messages
        )
        summary = None
        try:
            analysis = self.summarizer.analyze_research_findings(
                research_data=research_data,
                user_query=conversation.title or "Conversation summary",
            )
            candidate = (analysis.get("analysis") or "").strip()
            if candidate:
                summary = candidate.split("\n")[0]
        except Exception:
            pass
        if not summary:
            summary = self._fallback_summary(messages)
        return summary

    @staticmethod
    def conversation_to_dict(conversation: Conversation) -> Dict[str, Any]:
        return {
            "id": conversation.id,
            "title": conversation.title,
            "metadata": conversation.metadata or {},
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
        }

    @staticmethod
    def message_to_dict(message: Message) -> Dict[str, Any]:
        return {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "message_type": message.message_type,
            "created_at": message.created_at.isoformat(),
        }

    @staticmethod
    def search_session_to_dict(session_record: SearchSession) -> Dict[str, Any]:
        return {
            "id": session_record.id,
            "search_query": session_record.search_query,
            "result_count": session_record.result_count,
            "summary": session_record.summary,
            "created_at": session_record.created_at.isoformat(),
        }
