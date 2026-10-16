"""Import Markdown sources into authoritative pending knowledge chunks."""

import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.database.session import create_async_engine_from_settings, create_session_factory
from app.knowledge.ingestion import ingest_curated_knowledge


async def run(documents_root: Path) -> int:
    engine = create_async_engine_from_settings(get_settings())
    factory = create_session_factory(engine)
    try:
        async with factory() as session, session.begin():
            return await ingest_curated_knowledge(session, documents_root)
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=Path("knowledge"))
    args = parser.parse_args()
    count = asyncio.run(run(args.documents))
    print(f"ingested={count} status=pending")


if __name__ == "__main__":
    main()
