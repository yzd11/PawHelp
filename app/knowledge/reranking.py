"""Lazy local bge-reranker-v2-m3 adapter."""

from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any

from app.database.models import KnowledgeChunk
from app.knowledge.vectorization import vector_text


@dataclass(frozen=True)
class RerankedChunk:
    chunk: KnowledgeChunk
    score: float


class BgeReranker:
    """Score query/passage pairs locally and return descending Top-K results."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 8,
        query_max_length: int = 256,
        passage_max_length: int = 1024,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._query_max_length = query_max_length
        self._passage_max_length = passage_max_length
        self._model: Any | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    def _load(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    import torch
                    from FlagEmbedding import FlagReranker

                    use_cuda = torch.cuda.is_available()
                    self._model = FlagReranker(
                        self._model_name,
                        devices=["cuda:0" if use_cuda else "cpu"],
                        use_fp16=use_cuda,
                        query_max_length=self._query_max_length,
                        max_length=self._passage_max_length,
                        batch_size=self._batch_size,
                    )
        return self._model

    def rerank(
        self, query: str, chunks: Sequence[KnowledgeChunk], *, limit: int
    ) -> list[RerankedChunk]:
        if not chunks:
            return []
        # Loading once is not enough: compute_score also mutates model tensors.
        # Serialize the full call across parallel queries and chat requests.
        with self._inference_lock:
            scores = self._load().compute_score(
                [[query, vector_text(chunk)] for chunk in chunks],
                normalize=True,
            )
        if isinstance(scores, (int, float)):
            scores = [scores]
        ranked = sorted(
            (
                RerankedChunk(chunk=chunk, score=float(score))
                for chunk, score in zip(chunks, scores, strict=True)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[:limit]
