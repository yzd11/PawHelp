"""Local rotating context logs. These logs may contain customer personal data."""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_context_logging(root: Path) -> None:
    path = root / "log" / "app.log"
    path.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("app.workflow")
    if not any(isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == path for h in logger.handlers):
        handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
