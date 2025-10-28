from fastapi import APIRouter
from .routes import chat

api_routes = APIRouter()

api_routes.include_router(chat.chat_router, prefix="/chat", tags=["chat"])
