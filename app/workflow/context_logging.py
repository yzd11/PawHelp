"""Local rotating context logs. These logs may contain customer personal data."""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable


def configure_context_logging(root: Path) -> Callable[[], None]:
    """Attach the rotating context log; the returned callable detaches it again."""
    path = root / "log" / "app.log"
    path.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("app.workflow")
    logger.setLevel(logging.INFO)
    if any(isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == path for h in logger.handlers):
        return lambda: None
    handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)

    def detach() -> None:
        logger.removeHandler(handler)
        handler.close()

    return detach
