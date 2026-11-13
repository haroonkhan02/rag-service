"""`ragsvc-build-index`: corpus -> section chunks -> embeddings -> index bundle.

Writes `artifacts/index/` (see `ragsvc.bundle`) and prints the manifest. Needs the
`models` extra (sentence-transformers) and the data from `ragsvc-download`.
"""

from __future__ import annotations

import argparse
import json
import time

from ragsvc.bundle import IndexBundle, new_manifest
from ragsvc.chunking import chunk_corpus
from ragsvc.config import Settings, load_settings
from ragsvc.download import PAPER_TITLES
from ragsvc.ingest import Document, load_corpus


def load_documents(cfg: Settings) -> list[Document]:
    rst_root = cfg.data_dir / "ros2_documentation" / "source"
    if not rst_root.exists():
        raise SystemExit(f"{rst_root} not found: run `ragsvc-download` first")
    pdf_dir = cfg.data_dir / "pdfs" if cfg.corpus.pdfs else None
    return load_corpus(rst_root, pdf_dir, cfg.corpus.exclude_globs, PAPER_TITLES)


def build_bundle(cfg: Settings, docs: list[Document], embedder) -> IndexBundle:
    ch = cfg.chunking
    t0 = time.perf_counter()
    chunks = chunk_corpus(docs, ch.strategy, ch.max_words, ch.overlap_words, ch.section_header)
    t_chunk = time.perf_counter() - t0
    t0 = time.perf_counter()
    vectors = embedder.encode_docs([c.text for c in chunks], progress=True)
    t_embed = time.perf_counter() - t0
    timings = {"chunk": round(t_chunk, 2), "embed": round(t_embed, 1)}
    return IndexBundle(chunks, vectors, new_manifest(cfg, timings))


def main(argv: list[str] | None = None) -> None:
    from ragsvc.models import STEmbedder

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_settings(args.config)
    docs = load_documents(cfg)
    print(f"{len(docs)} documents, {sum(len(d.text.split()) for d in docs):,} words")
    e = cfg.embedding
    embedder = STEmbedder(e.model, e.query_prefix, e.batch_size, e.device)
    bundle = build_bundle(cfg, docs, embedder)
    bundle.manifest["embedding_device"] = embedder.device
    out = bundle.save(cfg.index_dir)
    print(json.dumps({k: v for k, v in bundle.manifest.items() if k != "sha256"}, indent=1))
    print(f"index written to {out}")


if __name__ == "__main__":
    main()
