"""Run dense/BM25/hybrid/reranked retrieval and grounded-answer evaluation."""

import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.evaluation.runtime import (
    DEFAULT_EVAL_CASES_PATH,
    DEFAULT_GENERATION_STRATEGIES,
    run_retrieval_evaluation,
)
from app.knowledge.search import RetrievalStrategy


async def run(
    cases_path: Path,
    output_dir: Path,
    max_cases: int | None,
    sample_per_bucket: int | None,
    strategies: list[RetrievalStrategy] | None,
    retrieval_only: bool,
    full_generation: bool,
) -> tuple[Path, Path]:
    settings = get_settings()
    _, paths = await run_retrieval_evaluation(
        settings,
        cases_path,
        output_dir,
        max_cases=max_cases,
        sample_per_bucket=sample_per_bucket,
        strategies=strategies,
        generate_answers=not retrieval_only,
        generation_strategies=(
            None if full_generation else DEFAULT_GENERATION_STRATEGIES
        ),
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_EVAL_CASES_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--sample-per-bucket",
        type=int,
        help="Take N cases from every query bucket for a balanced smoke run.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[item.value for item in RetrievalStrategy],
        help="Run only selected retrieval strategies.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip answer generation and faithfulness judging.",
    )
    generation_group = parser.add_mutually_exclusive_group()
    generation_group.add_argument(
        "--full-generation",
        dest="full_generation",
        action="store_true",
        help="Generate answers for all selected strategies (the default).",
    )
    generation_group.add_argument(
        "--production-only",
        dest="full_generation",
        action="store_false",
        help="Generate answers for hybrid_rerank only to save model calls.",
    )
    parser.set_defaults(full_generation=True)
    args = parser.parse_args()
    json_path, markdown_path = asyncio.run(
        run(
            args.cases,
            args.output_dir,
            args.max_cases,
            args.sample_per_bucket,
            (
                [RetrievalStrategy(value) for value in args.strategies]
                if args.strategies
                else None
            ),
            args.retrieval_only,
            args.full_generation,
        )
    )
    print(f"json={json_path} markdown={markdown_path}")


if __name__ == "__main__":
    main()
