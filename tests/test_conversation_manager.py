import importlib
import os
from pathlib import Path

import pytest


def _fresh_conversation_manager(tmp_path: Path, summary_threshold: int = 3):
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'memory.db'}"

    import backend.database.connection as db_connection

    importlib.reload(db_connection)

    import backend.memory.conversation_manager as manager_module

    importlib.reload(manager_module)

    return manager_module.ConversationManager(summary_threshold=summary_threshold)


class DummySummarizer:
    def analyze_research_findings(
        self, research_data: str, user_query: str, context: str = ""
    ):
        return {"analysis": "Dummy summary line 1\nAdditional context"}


class FailingSummarizer:
    def analyze_research_findings(
        self, research_data: str, user_query: str, context: str = ""
    ):
        raise RuntimeError("LLM failed to summarize")


def test_conversation_summary_triggers_after_threshold(tmp_path):
    manager = _fresh_conversation_manager(tmp_path, summary_threshold=3)
    manager.summarizer = DummySummarizer()

    conversation = manager.create_conversation(title="Testing summary")

    for idx in range(3):
        manager.add_message(
            conversation_id=conversation.id,
            role="user",
            content=f"Test message {idx}",
            message_type="research",
        )

    refreshed = manager.get_conversation(conversation.id)
    assert refreshed is not None
    assert refreshed.metadata.get("summary") == "Dummy summary line 1"


def test_conversation_summary_falls_back_when_llm_fails(tmp_path):
    manager = _fresh_conversation_manager(tmp_path, summary_threshold=2)
    manager.summarizer = FailingSummarizer()

    conversation = manager.create_conversation(title="Fallback summary")
    manager.add_message(conversation.id, "user", "First line is important.")
    manager.add_message(conversation.id, "assistant", "Assistant reply adds info.")

    refreshed = manager.get_conversation(conversation.id)
    assert refreshed is not None
    assert "First line is important" in refreshed.metadata.get("summary", "")
