"""The index bundle: everything the service needs to answer searches, as three plain files.

    chunks.jsonl     one chunk per line (id, doc, section, text, source, page)
    embeddings.npy   float32 [n_chunks, dim], L2-normalised, same row order as chunks.jsonl
    manifest.json    what built it (corpus commit, embedding model, chunking) + SHA-256 of both

No vector database: at a few thousand chunks an exact matrix product is faster than an ANN
index and gives the same answer on every machine. The bundle is written to a temporary
folder and renamed into place, so a reader never sees a half-written index, and it is
checked against its manifest (and the service settings) when loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ragsvc.chunking import Chunk
from ragsvc.config import Settings

FORMAT_VERSION = 1
CHUNKS, EMBEDDINGS, MANIFEST = "chunks.jsonl", "embeddings.npy", "manifest.json"


class BundleError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_info(cfg: Settings) -> dict:
    """The settings that must match between the index and the service that loads it."""
    return {
        "corpus_commit": cfg.corpus.ros2_commit,
        "pdfs": cfg.corpus.pdfs,
        "chunking": {"strategy": cfg.chunking.strategy, "max_words": cfg.chunking.max_words,
                     "overlap_words": cfg.chunking.overlap_words,
                     "section_header": cfg.chunking.section_header},
        "embedding_model": cfg.embedding.model,
    }


@dataclass
class IndexBundle:
    chunks: list[Chunk]
    embeddings: np.ndarray
    manifest: dict

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.chunks):
            raise BundleError(f"{len(self.chunks)} chunks but embeddings of shape "
                              f"{self.embeddings.shape}")
        ids = [c.chunk_id for c in self.chunks]
        if len(set(ids)) != len(ids):
            raise BundleError("duplicate chunk ids")

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".index-", dir=out_dir.parent))
        try:
            with (tmp / CHUNKS).open("w", encoding="utf-8") as f:
                for c in self.chunks:
                    f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            np.save(tmp / EMBEDDINGS, self.embeddings.astype(np.float32), allow_pickle=False)
            manifest = {
                **self.manifest,
                "format_version": FORMAT_VERSION,
                "n_chunks": len(self.chunks),
                "n_documents": len({c.doc_id for c in self.chunks}),
                "dim": self.dim,
                "sha256": {CHUNKS: file_sha256(tmp / CHUNKS),
                           EMBEDDINGS: file_sha256(tmp / EMBEDDINGS)},
            }
            (tmp / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
            old = out_dir.with_name(out_dir.name + ".old")
            if out_dir.exists():
                shutil.rmtree(old, ignore_errors=True)
                os.replace(out_dir, old)
            os.replace(tmp, out_dir)
            shutil.rmtree(old, ignore_errors=True)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self.manifest = manifest
        return out_dir


def load_bundle(index_dir: Path, cfg: Settings | None = None, verify: bool = True) -> IndexBundle:
    index_dir = Path(index_dir)
    mpath = index_dir / MANIFEST
    if not mpath.exists():
        raise BundleError(f"no index in {index_dir}: run `ragsvc-build-index` first")
    manifest = json.loads(mpath.read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise BundleError(f"index format {manifest.get('format_version')} != {FORMAT_VERSION}")
    if verify:
        for name, expected in manifest["sha256"].items():
            if file_sha256(index_dir / name) != expected:
                raise BundleError(f"{name} does not match its checksum in {MANIFEST}")
    if cfg is not None:
        want, have = build_info(cfg), manifest.get("build", {})
        diff = sorted(k for k in want if want[k] != have.get(k))
        if diff:
            raise BundleError(f"index was built with different settings ({', '.join(diff)}); "
                              "rebuild it or change the service config")
    with (index_dir / CHUNKS).open(encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
    emb = np.load(index_dir / EMBEDDINGS, allow_pickle=False)
    return IndexBundle(chunks, emb, manifest)


def new_manifest(cfg: Settings, timings: dict | None = None) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    try:
        pypdf_version = version("pypdf")
    except PackageNotFoundError:  # pragma: no cover
        pypdf_version = None
    return {"build": build_info(cfg),
            "pypdf": pypdf_version,  # recorded, not checked: the server never parses PDFs
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "timings_s": timings or {}}
