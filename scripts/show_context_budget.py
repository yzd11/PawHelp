"""Print calibrated context budgets. No database or inference requests."""
import json
from dataclasses import asdict

from app.config import get_settings
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.llm import create_chat_model
from app.workflow.graph import WorkflowChatService


def main() -> None:
    settings = get_settings()
    engine = create_async_engine_from_settings(settings)
    service = WorkflowChatService(settings=settings, model=create_chat_model(settings), session_factory=create_session_factory(engine))
    service._memory.startup_check()
    print(json.dumps(asdict(service._memory.budget), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
