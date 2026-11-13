"""`ragsvc-parity`: does the service's index + searcher reproduce robotics-docs-rag?

Runs every retrieval mode on the answerable questions of the shared evaluation set and
compares, per mode, with the published results of robotics-docs-rag (section chunking):
aggregate metrics, how many questions get the same relevance pattern in the top 10, and
how many get the same top-5 documents. Also times each search stage (warm, this machine).

Writes reports/parity.md and reports/parity.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

from ragsvc.bundle import file_sha256, load_bundle
from ragsvc.config import RETRIEVAL_MODES, load_settings
from ragsvc.evalset import hit_at_k, is_relevant, load_questions, mrr_at_k, ndcg_at_k
from ragsvc.search import Searcher

K = 10
METRICS = ("hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10")


def score_questions(searcher: Searcher, questions, mode: str) -> list[dict]:
    n_rel = {q.id: sum(is_relevant(c, q) for c in searcher.chunks) for q in questions}
    searcher.search(questions[0].question, mode, K)  # warm-up: model load, MPS kernels
    out = []
    for q in questions:
        res = searcher.search(q.question, mode, K)
        rels = [is_relevant(h.chunk, q) for h in res.hits]
        rels += [False] * (K - len(rels))
        out.append({"id": q.id, "rels": rels, "n_relevant": n_rel[q.id],
                    "top_docs": [h.chunk.doc_id for h in res.hits[:5]],
                    "timings_ms": res.timings_ms})
    return out


def aggregate(per_q: list[dict]) -> dict[str, float]:
    def mean(f):
        return statistics.mean(f(p) for p in per_q)

    return {
        "hit@1": mean(lambda p: hit_at_k(p["rels"], 1)),
        "hit@5": mean(lambda p: hit_at_k(p["rels"], 5)),
        "hit@10": mean(lambda p: hit_at_k(p["rels"], 10)),
        "mrr@10": mean(lambda p: mrr_at_k(p["rels"], K)),
        "ndcg@10": mean(lambda p: ndcg_at_k(p["rels"], K, p["n_relevant"])),
    }


def reference_rows(path: Path, strategy: str) -> dict[str, dict]:
    ref = json.loads(path.read_text())
    rows = {}
    for r in ref["results"]:
        if r["strategy"] != strategy:
            continue
        rows[r["mode"]] = {
            "metrics": {m: r[m]["mean"] for m in METRICS},
            "per_question": {p["id"]: p for p in r["per_question"]},
        }
    return rows


def compare(per_q: list[dict], ref: dict) -> dict:
    same_rels = sum(p["rels"] == ref["per_question"][p["id"]]["rels"] for p in per_q)
    same_top5 = sum(p["top_docs"] == ref["per_question"][p["id"]]["top_docs"] for p in per_q)
    return {"same_relevance_top10": same_rels, "same_top5_docs": same_top5}


def stage_latency(per_q: list[dict]) -> dict[str, dict[str, float]]:
    stages = sorted({s for p in per_q for s in p["timings_ms"]})
    out = {}
    for s in stages:
        v = sorted(p["timings_ms"][s] for p in per_q if s in p["timings_ms"])
        out[s] = {"median": round(statistics.median(v), 2),
                  "p95": round(v[round(0.95 * (len(v) - 1))], 2)}
    return out


def markdown(report: dict) -> str:
    n = report["n_questions"]
    lines = [
        "# Parity with robotics-docs-rag",
        "",
        f"{n} answerable questions, section chunking, {report['n_chunks']} chunks. "
        "\"ref\" = published results of robotics-docs-rag (Chroma HNSW); \"svc\" = this "
        "service (exact NumPy search). Same relevance = identical relevant/not pattern in the "
        "top 10; same top-5 = identical document list.",
        "",
        "| Mode | MRR@10 svc | MRR@10 ref | Hit@1 svc | Hit@1 ref | Hit@5 svc | Hit@5 ref "
        "| nDCG@10 svc | nDCG@10 ref | Same relevance | Same top-5 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, r in report["modes"].items():
        s, f = r["service"], r["reference"]
        lines.append(
            f"| {mode} | {s['mrr@10']:.3f} | {f['mrr@10']:.3f} | {s['hit@1']:.3f} | "
            f"{f['hit@1']:.3f} | {s['hit@5']:.3f} | {f['hit@5']:.3f} | {s['ndcg@10']:.3f} | "
            f"{f['ndcg@10']:.3f} | {r['same_relevance_top10']}/{n} | {r['same_top5_docs']}/{n} |")
    lines += ["", "Median / p95 time per stage in ms (warm, one query at a time, "
              f"{report['machine']}, embedding device `{report['device']}`):", "",
              "| Mode | Stage | Median | p95 |", "|---|---|---|---|"]
    for mode, r in report["modes"].items():
        for stage, v in r["latency_ms"].items():
            lines.append(f"| {mode} | {stage} | {v['median']:.2f} | {v['p95']:.2f} |")
    if report.get("chunks_identical_to_reference") is not None:
        lines += ["", "Chunk file byte-identical to the reference index: "
                  f"{'yes' if report['chunks_identical_to_reference'] else 'no'}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    from ragsvc.models import STEmbedder, STReranker

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--modes", nargs="+", default=list(RETRIEVAL_MODES), choices=RETRIEVAL_MODES)
    ap.add_argument("--reference-chunks", type=Path, default=None,
                    help="chunks_section.jsonl of a local robotics-docs-rag index to compare")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)
    cfg = load_settings(args.config)
    bundle = load_bundle(cfg.index_dir, cfg)
    e, r = cfg.embedding, cfg.retrieval
    embedder = STEmbedder(e.model, e.query_prefix, e.batch_size, e.device)
    reranker = STReranker(r.reranker, e.device) if "hybrid_rerank" in args.modes else None
    searcher = Searcher(bundle, r, embedder, reranker)
    questions = [q for q in load_questions(cfg.data_dir / "eval" / "questions.jsonl")
                 if q.answerable]
    refs = reference_rows(cfg.data_dir / "eval" / "reference_retrieval.json",
                          cfg.chunking.strategy)
    report = {"n_questions": len(questions), "n_chunks": len(bundle.chunks),
              "machine": platform.machine(), "device": embedder.device, "modes": {}}
    if args.reference_chunks:
        report["chunks_identical_to_reference"] = (
            file_sha256(args.reference_chunks) == bundle.manifest["sha256"]["chunks.jsonl"])
    for mode in args.modes:
        per_q = score_questions(searcher, questions, mode)
        row = {"service": aggregate(per_q), "reference": refs[mode]["metrics"],
               **compare(per_q, refs[mode]), "latency_ms": stage_latency(per_q)}
        report["modes"][mode] = row
        print(f"{mode:13s} mrr@10 svc={row['service']['mrr@10']:.3f} "
              f"ref={row['reference']['mrr@10']:.3f} same_rels={row['same_relevance_top10']}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "parity.json").write_text(json.dumps(report, indent=1) + "\n")
    (args.out / "parity.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
