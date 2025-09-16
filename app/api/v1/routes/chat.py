from fastapi import APIRouter
chat_router = APIRouter()

@chat_router.get("/messages")
def get_messages():
    return {"messages": []}