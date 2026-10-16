"""Public HTTP router assembly. Existing create_api_router imports remain valid."""
from fastapi import APIRouter
from app.api.chat import chat_router
from app.api.knowledge import knowledge_router


def create_api_router(settings, model, session_factory, tool_chat_service, knowledge_admin_service) -> APIRouter:
    router = APIRouter()
    router.include_router(chat_router(settings, model, session_factory, tool_chat_service, knowledge_admin_service))
    router.include_router(knowledge_router(knowledge_admin_service))
    return router
