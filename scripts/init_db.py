"""Upgrade the configured database to the latest Alembic revision."""

from pathlib import Path

from alembic import command
from alembic.config import Config


def main() -> None:
    """Apply all pending schema migrations."""

    root = Path(__file__).resolve().parents[1]
    command.upgrade(Config(root / "alembic.ini"), "head")


if __name__ == "__main__":
    main()
