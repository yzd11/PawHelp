"""Lazy local BGE-M3 dense embedding adapter."""

from collections.abc import Sequence
from threading import Lock
from typing import Any


VECTOR_DIMENSION = 1024


class BgeM3Embedder:
    """Load BGE-M3 on first use and return only normalized dense vectors."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 16,
        max_length: int = 1024,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: Any | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    def _load(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    import torch
                    from FlagEmbedding import BGEM3FlagModel

                    use_cuda = torch.cuda.is_available()
                    self._model = BGEM3FlagModel(
                        self._model_name,
                        devices=["cuda:0" if use_cuda else "cpu"],
                        use_fp16=use_cuda,
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False,
                    )
        return self._model

    def warm(self) -> None:
        """Load weights and encode once so the first query skips CUDA start-up."""

        self.embed(["预热"])

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode text into BGE-M3's 1024-dimensional dense representation."""

        if not texts:
            return []
        # FlagEmbedding mutates the shared model during inference (half/to/eval).
        # Parallel queries may use different worker threads, not different models.
        with self._inference_lock:
            output = self._load().encode(
                list(texts),
                batch_size=self._batch_size,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vectors = output["dense_vecs"].tolist()
        if any(len(vector) != VECTOR_DIMENSION for vector in vectors):
            raise ValueError("BGE-M3 returned an unexpected dense vector dimension")
        return vectors
