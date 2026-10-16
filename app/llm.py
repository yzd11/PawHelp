"""LangChain model construction."""

from langchain_openai import ChatOpenAI

from app.config import Settings


def create_chat_model(settings: Settings, *, max_retries: int = 2) -> ChatOpenAI:
    """Create the application's single OpenAI-compatible chat model."""

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=str(settings.chat_base_url).rstrip("/"),
        timeout=settings.model_timeout_seconds,
        max_tokens=settings.model_max_output_tokens,
        streaming=True,
        stream_usage=True,
        max_retries=max_retries,
    )
