"""The evaluation set of robotics-docs-rag and the retrieval metrics computed on it.

A retrieved chunk is *relevant* when it comes from a gold document **and** contains one of
the question's answer keys (short phrases that answer it), the same definition as in
robotics-docs-rag, so numbers are directly comparable.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ragsvc.chunking import Chunk

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", text.replace("\\", "")).strip().lower()


@dataclass
class Question:
    id: str
    category: str
    question: str
    gold_docs: list[str] = field(default_factory=list)
    answer_keys: list[str] = field(default_factory=list)
    reference_answer: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.gold_docs)


def load_questions(path: Path | str) -> list[Question]:
    with Path(path).open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


def is_relevant(chunk: Chunk, q: Question) -> bool:
    if chunk.doc_id not in q.gold_docs:
        return False
    text = norm(chunk.text)
    return any(norm(k) in text for k in q.answer_keys)


def hit_at_k(rels: Sequence[bool], k: int) -> float:
    return float(any(rels[:k]))


def mrr_at_k(rels: Sequence[bool], k: int) -> float:
    for i, r in enumerate(rels[:k], start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: Sequence[bool], k: int, n_relevant: int) -> float:
    """Binary-gain nDCG; the ideal ranking puts all `n_relevant` existing items first."""
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1) if r)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, n_relevant) + 1))
    return dcg / idcg if idcg else 0.0
