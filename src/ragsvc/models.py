"""The two local neural models (bi-encoder and cross-encoder), loaded lazily.

PyTorch and sentence-transformers are only imported here, so the rest of the package (and
its tests) works without them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode_docs(self, texts: Sequence[str], progress: bool = False) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class STEmbedder:
    """sentence-transformers bi-encoder; returns L2-normalised float32 vectors."""

    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 64,
                 device: str = "auto"):
        from sentence_transformers import SentenceTransformer

        self.model_name, self.query_prefix, self.batch_size = model_name, query_prefix, batch_size
        self.device = pick_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)

    def _encode(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        emb = self.model.encode(list(texts), batch_size=self.batch_size,
                                normalize_embeddings=True, show_progress_bar=progress,
                                convert_to_numpy=True)
        return emb.astype(np.float32)

    def encode_docs(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        return self._encode(texts, progress)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        # BGE was trained with an instruction on the query side only
        return self._encode([self.query_prefix + t for t in texts])


class STReranker:
    """sentence-transformers cross-encoder: scores (query, passage) pairs jointly."""

    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=pick_device(device), max_length=max_length)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(query, t) for t in texts], batch_size=32,
                                    show_progress_bar=False)
        return [float(s) for s in scores]
