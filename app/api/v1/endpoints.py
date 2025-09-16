from fastapi import APIRouter

api_routes = APIRouter()
from .routes import chat

api_routes.include_router(chat.chat_router, prefix="/chat", tags=["chat"])
