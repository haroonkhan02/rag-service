"""`ragsvc-serve`: load the index and models, warm them up, then serve.

Everything slow or fallible happens before the port opens: the bundle is checked against
its manifest and the settings, the models are loaded and run once. A process that starts
is therefore ready, and one that cannot serve fails at start-up with the reason.
"""

from __future__ import annotations

import argparse
import time

from ragsvc.bundle import load_bundle
from ragsvc.config import Settings, load_settings
from ragsvc.generate import make_generator
from ragsvc.search import Searcher
from ragsvc.service import RagService
from ragsvc.tracing import Tracer, setup_tracing


def build_service(cfg: Settings, generator=None, tracer: Tracer | None = None) -> RagService:
    from ragsvc.models import STEmbedder, STReranker

    bundle = load_bundle(cfg.index_dir, cfg)
    e = cfg.embedding
    embedder = STEmbedder(e.model, e.query_prefix, e.batch_size, e.device)
    reranker = STReranker(cfg.retrieval.reranker, e.device)
    searcher = Searcher(bundle, cfg.retrieval, embedder, reranker)
    searcher.search("warm up the models", "hybrid_rerank")  # first call compiles kernels
    return RagService(searcher, generator or make_generator(cfg.generation), cfg,
                      tracer or setup_tracing(cfg.tracing))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Serve the RAG API.")
    ap.add_argument("--config", default=None, help="settings file (default configs/service.yaml)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)

    import uvicorn

    from ragsvc.app import create_app

    cfg = load_settings(args.config)
    t0 = time.perf_counter()
    service = build_service(cfg)
    print(f"index: {len(service.searcher.chunks)} chunks, generator: {service.generator.name}, "
          f"ready in {time.perf_counter() - t0:.1f} s", flush=True)
    uvicorn.run(create_app(service), host=args.host or cfg.server.host,
                port=args.port or cfg.server.port, log_level="warning")


if __name__ == "__main__":
    main()
