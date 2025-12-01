from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from backend.memory import ConversationManager

chat_router = APIRouter()
conversation_manager = ConversationManager()


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MessageCreateRequest(BaseModel):
    role: str
    content: str
    message_type: str = "research"


class SearchSessionCreateRequest(BaseModel):
    search_query: str
    result_count: int = 0
    summary: Optional[str] = None


@chat_router.post("/conversations")
def create_conversation(payload: ConversationCreateRequest):
    conversation = conversation_manager.create_conversation(
        title=payload.title, metadata=payload.metadata
    )
    return {"conversation": conversation_manager.conversation_to_dict(conversation)}


@chat_router.get("/conversations")
def list_conversations():
    conversations = conversation_manager.list_conversations()
    serialized = [
        conversation_manager.conversation_to_dict(conversation)
        for conversation in conversations
    ]
    return {"conversations": serialized}


@chat_router.post("/conversations/{conversation_id}/messages")
def add_message(conversation_id: int, payload: MessageCreateRequest):
    try:
        message = conversation_manager.add_message(
            conversation_id=conversation_id,
            role=payload.role,
            content=payload.content,
            message_type=payload.message_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"message": conversation_manager.message_to_dict(message)}


@chat_router.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: int, limit: int = 50, offset: int = 0):
    messages = conversation_manager.get_messages(
        conversation_id=conversation_id, limit=limit, offset=offset
    )
    return {
        "messages": [conversation_manager.message_to_dict(message) for message in messages]
    }


@chat_router.post("/conversations/{conversation_id}/search-sessions")
def record_search_session(conversation_id: int, payload: SearchSessionCreateRequest):
    try:
        session_record = conversation_manager.record_search_session(
            conversation_id=conversation_id,
            search_query=payload.search_query,
            result_count=payload.result_count,
            summary=payload.summary,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "search_session": conversation_manager.search_session_to_dict(session_record)
    }


@chat_router.get("/conversations/{conversation_id}/search-sessions")
def get_search_sessions(conversation_id: int, limit: int = 20, offset: int = 0):
    sessions = conversation_manager.get_search_sessions(
        conversation_id=conversation_id, limit=limit, offset=offset
    )
    return {
        "search_sessions": [
            conversation_manager.search_session_to_dict(session_record)
            for session_record in sessions
        ]
    }


@chat_router.get("/metrics")
def get_metrics():
    try:
        metrics = conversation_manager.get_metrics()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to collect metrics: {exc}")
    return {"metrics": metrics}


@chat_router.get("/conversations/{conversation_id}/export")
def export_conversation(
    conversation_id: int,
    format: str = Query("json", regex="^(json|markdown)$"),
):
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = conversation_manager.get_messages(conversation_id, limit=1000, offset=0)
    sessions = conversation_manager.get_search_sessions(conversation_id, limit=100, offset=0)

    conversation_payload = {
        "conversation": conversation_manager.conversation_to_dict(conversation),
        "messages": [
            conversation_manager.message_to_dict(message) for message in messages
        ],
        "search_sessions": [
            conversation_manager.search_session_to_dict(session) for session in sessions
        ],
    }

    if format == "json":
        return JSONResponse(conversation_payload)

    markdown_lines = [
        f"# Conversation: {conversation_payload['conversation'].get('title') or 'Untitled'}",
        "",
        "## Metadata",
        f"- ID: {conversation_payload['conversation']['id']}",
        f"- Created: {conversation_payload['conversation']['created_at']}",
        f"- Updated: {conversation_payload['conversation']['updated_at']}",
        "",
        "## Messages",
    ]
    for message in conversation_payload["messages"]:
        markdown_lines.append(
            f"- **{message['role'].title()}** ({message['created_at']}): {message['content']}"
        )
    markdown_lines.extend(
        [
            "",
            "## Search Sessions",
        ]
    )
    for session_record in conversation_payload["search_sessions"]:
        markdown_lines.append(
            f"- [{session_record['created_at']}] `{session_record['search_query']}`"
            f" ({session_record['result_count']} results) – {session_record['summary'] or 'No summary'}"
        )

    markdown = "\n".join(markdown_lines)
    return PlainTextResponse(content=markdown, media_type="text/markdown")
