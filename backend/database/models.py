from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from .connection import Base


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_title", "title"),
        Index("ix_conversations_updated_at", "updated_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(256), nullable=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    search_sessions = relationship("SearchSession", back_populates="conversation", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"Conversation(id={self.id}, title={self.title})"



class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
        Index("ix_messages_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    message_type = Column(String(32), nullable=False, default="research")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        return f"Message(id={self.id}, role={self.role})"


class SearchSession(Base):
    __tablename__ = "search_sessions"
    __table_args__ = (
        Index("ix_search_sessions_conversation_id", "conversation_id"),
        Index("ix_search_sessions_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    search_query = Column(String(512), nullable=False)
    result_count = Column(Integer, default=0)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="search_sessions")

    def __repr__(self) -> str:
        return f"SearchSession(id={self.id}, query={self.search_query})"
