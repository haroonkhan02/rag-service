"""Typed service settings: YAML file, then environment overrides, validated once at start-up.

    configs/service.yaml                  the defaults (checked into git)
    RAGSVC_CONFIG=path/to/other.yaml      use another file
    RAGSVC__RETRIEVAL__TOP_K=8            override one value (double underscore = nesting)

A typo in a key or a wrong type fails at start-up with a clear message, not on the first
request. Secrets (the Anthropic key) are never part of these settings; the SDK reads them
from the environment.
"""

from __future__ import annotations

import dataclasses
import os
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/service.yaml")
ENV_PREFIX = "RAGSVC__"
RETRIEVAL_MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CorpusSettings:
    ros2_commit: str = "0ea463a2b27865c93acc928eccd2276e47c9f71c"
    exclude_globs: tuple[str, ...] = ("Releases/*Complete-Changelog*", "_internal/*")
    pdfs: bool = True


@dataclass(frozen=True)
class EvalSettings:
    # the 66 hand-written questions of robotics-docs-rag, fetched at a pinned commit
    source_repo: str = "haroonkhan02/robotics-docs-rag"
    source_commit: str = "99393783595570318515d2da43fd911b6bfa9ac1"


@dataclass(frozen=True)
class ChunkingSettings:
    strategy: str = "section"
    max_words: int = 180
    overlap_words: int = 40
    section_header: bool = True


@dataclass(frozen=True)
class EmbeddingSettings:
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 64
    device: str = "auto"  # auto | mps | cpu | cuda


@dataclass(frozen=True)
class RetrievalSettings:
    mode: str = "hybrid_rerank"
    candidates: int = 30
    rrf_k: int = 60
    rerank_top: int = 30
    top_k: int = 5
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class GenerationSettings:
    model: str = "claude-opus-5-5"
    effort: str = "low"
    max_tokens: int = 1024


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    index_dir: Path = Path("artifacts/index")
    corpus: CorpusSettings = field(default_factory=CorpusSettings)
    eval: EvalSettings = field(default_factory=EvalSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)

    def validate(self) -> Settings:
        r = self.retrieval
        if r.mode not in RETRIEVAL_MODES:
            raise ConfigError(f"retrieval.mode must be one of {RETRIEVAL_MODES}, got {r.mode!r}")
        for name in ("candidates", "top_k", "rerank_top", "rrf_k"):
            if getattr(r, name) < 1:
                raise ConfigError(f"retrieval.{name} must be >= 1")
        if r.top_k > r.candidates:
            raise ConfigError("retrieval.top_k cannot exceed retrieval.candidates")
        if self.chunking.strategy not in ("fixed", "recursive", "section"):
            raise ConfigError(f"unknown chunking.strategy {self.chunking.strategy!r}")
        if self.embedding.device not in ("auto", "mps", "cpu", "cuda"):
            raise ConfigError(f"unknown embedding.device {self.embedding.device!r}")
        return self


def _coerce(value, tp, where: str):
    """Convert a YAML / env value to the annotated type, or raise ConfigError."""
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        return _build(tp, value, where)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return None if value is None else _coerce(value, args[0], where)
    if origin is tuple:
        if isinstance(value, str):
            value = [v for v in value.split(",") if v]
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{where}: expected a list")
        return tuple(str(v) for v in value)
    if tp is bool:
        if isinstance(value, str) and value.lower() in ("1", "true", "yes", "0", "false", "no"):
            return value.lower() in ("1", "true", "yes")
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        try:
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise ValueError
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: expected an integer, got {value!r}") from None
    if tp is Path:
        return Path(str(value))
    if tp is str:
        return str(value)
    raise ConfigError(f"{where}: unsupported type {tp}")  # pragma: no cover


def _build(cls, data: dict, where: str = ""):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"unknown setting(s) {[f'{where}{u}' for u in unknown]}")
    kwargs = {k: _coerce(v, hints[k], f"{where}{k}.") if dataclasses.is_dataclass(hints[k])
              else _coerce(v, hints[k], f"{where}{k}") for k, v in data.items()}
    return cls(**kwargs)


def _env_overrides(environ: dict[str, str]) -> dict:
    out: dict = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX):].lower().split("__")
        node = out
        for p in path[:-1]:
            node = node.setdefault(p, {})
        node[path[-1]] = yaml.safe_load(raw) if raw else raw
    return out


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_settings(path: Path | str | None = None,
                  environ: dict[str, str] | None = None) -> Settings:
    environ = dict(os.environ) if environ is None else environ
    path = Path(path or environ.get("RAGSVC_CONFIG") or DEFAULT_CONFIG)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    data = _merge(data or {}, _env_overrides(environ))
    return _build(Settings, data).validate()
