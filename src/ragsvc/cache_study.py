"""How low can the semantic-cache threshold go before it serves the wrong result?

The cache is seeded with the evaluation questions; three kinds of new question then arrive:

    paraphrases   eval/paraphrases.jsonl: the same question in other words. Reusing the
                  cached result of its source question is a correct hit.
    near misses   eval/near_misses.jsonl: similar wording, different question ("record all
                  topics" vs "record only two topics"). Any reuse is wrong.
    distinct      each evaluation question against all the others. Any reuse is wrong.

For every threshold the study reports hit rate, wrong reuse, and the retrieval quality of
what a paraphrase is actually served (the cached passages on a hit, a fresh search on a
miss), scored against the paraphrase's own gold passages.

    ragsvc-cache-study            # writes reports/semantic_cache.md / .json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ragsvc.evalset import hit_at_k, is_relevant, mrr_at_k

K = 10
THRESHOLDS = tuple(round(0.80 + 0.01 * i, 2) for i in range(20))  # 0.80 .. 0.99


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def nearest(query_vecs: np.ndarray, cache_vecs: np.ndarray,
            exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Index and cosine similarity of the most similar cached vector for each query."""
    sims = query_vecs @ cache_vecs.T
    if exclude_self:
        np.fill_diagonal(sims, -1.0)
    idx = sims.argmax(axis=1)
    return idx, sims[np.arange(len(idx)), idx]


def quantiles(values: Sequence[float]) -> dict[str, float]:
    v = sorted(float(x) for x in values)
    return {"min": v[0], "p10": v[round(0.10 * (len(v) - 1))], "median": statistics.median(v),
            "p90": v[round(0.90 * (len(v) - 1))], "max": v[-1]}


def sweep(seed_ids: list[str], para_ids: list[str], para_idx: np.ndarray, para_sim: np.ndarray,
          near_sim: np.ndarray, distinct_sim: np.ndarray, fresh_rels: dict[str, list[bool]],
          seed_rels_for: dict[tuple[str, str], list[bool]],
          thresholds: Sequence[float] = THRESHOLDS) -> list[dict]:
    """`seed_rels_for[(paraphrase id, seed id)]`: relevance of the seed question's cached
    passages judged against the paraphrase's question."""
    rows = []
    n = len(para_ids)
    for t in thresholds:
        hit = para_sim >= t
        correct = hit & np.array([seed_ids[j] == pid
                                  for j, pid in zip(para_idx, para_ids, strict=True)])
        served = [seed_rels_for[(pid, seed_ids[j])] if h else fresh_rels[pid]
                  for pid, j, h in zip(para_ids, para_idx, hit, strict=True)]
        rows.append({
            "threshold": t,
            "paraphrase_hit_rate": float(hit.mean()),
            "paraphrase_correct_hits": int(correct.sum()),
            "paraphrase_wrong_hits": int((hit & ~correct).sum()),
            "near_miss_wrong_hits": int((near_sim >= t).sum()),
            "distinct_wrong_hits": int((distinct_sim >= t).sum()),
            "served_hit@5": sum(hit_at_k(r, 5) for r in served) / n,
            "served_mrr@10": sum(mrr_at_k(r, K) for r in served) / n,
        })
    return rows


MARGIN = 0.01


def pick_threshold(rows: list[dict], highest_wrong: float) -> float | None:
    """Lowest threshold with no wrong reuse that also clears the most similar wrong
    question by `MARGIN`: a near miss at 0.9497 should not decide between 0.94 and 0.95."""
    for r in rows:
        clean = not (r["paraphrase_wrong_hits"] or r["near_miss_wrong_hits"]
                     or r["distinct_wrong_hits"])
        if clean and r["threshold"] >= highest_wrong + MARGIN:
            return r["threshold"]
    return None


def markdown(rep: dict) -> str:
    n_p, n_n, n_s = rep["n_paraphrases"], rep["n_near_misses"], rep["n_seed"]
    s = rep["similarity"]
    lines = [
        "# Semantic cache: hit rate against wrong reuse",
        "",
        f"Cache seeded with the {n_s} evaluation questions. Query embeddings: "
        f"`{rep['embedding_model']}` (cosine similarity). {n_p} hand-written paraphrases "
        f"(reuse of the source question = correct), {n_n} hand-written near misses (any reuse "
        "= wrong), and every evaluation question against the others (any reuse = wrong).",
        "",
        "Similarity to the nearest cached question:",
        "",
        "| Set | min | p10 | median | p90 | max |",
        "|---|---|---|---|---|---|",
    ]
    for name, label in (("paraphrase_to_source", "Paraphrase to its source"),
                        ("near_miss", "Near miss to nearest question"),
                        ("distinct", "Evaluation question to nearest other")):
        q = s[name]
        lines.append(f"| {label} | {q['min']:.3f} | {q['p10']:.3f} | {q['median']:.3f} | "
                     f"{q['p90']:.3f} | {q['max']:.3f} |")
    lines += [
        "",
        f"Retrieval quality is what the {n_p} paraphrases are served (cached passages on a "
        f"hit, a fresh `{rep['mode']}` search on a miss), judged against each paraphrase's "
        "own gold passages.",
        "",
        f"| Threshold | Paraphrase hits | Wrong (paraphrase) | Wrong (near miss, of {n_n}) "
        f"| Wrong (distinct, of {n_s}) | Served Hit@5 | Served MRR@10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rep["sweep"]:
        mark = " **(default)**" if r["threshold"] == rep["chosen_threshold"] else ""
        lines.append(
            f"| {r['threshold']:.2f}{mark} | {r['paraphrase_hit_rate']:.1%} "
            f"({r['paraphrase_correct_hits']}/{n_p}) | {r['paraphrase_wrong_hits']} | "
            f"{r['near_miss_wrong_hits']} | {r['distinct_wrong_hits']} | "
            f"{r['served_hit@5']:.3f} | {r['served_mrr@10']:.3f} |")
    b = rep["baselines"]
    lines += [
        "",
        f"No cache, paraphrases searched fresh: Hit@5 {b['paraphrase_fresh']['hit@5']:.3f}, "
        f"MRR@10 {b['paraphrase_fresh']['mrr@10']:.3f}. Original wording: Hit@5 "
        f"{b['original']['hit@5']:.3f}, MRR@10 {b['original']['mrr@10']:.3f}.",
        "",
    ]
    if rep["chosen_threshold"] is None:
        lines.append("No threshold in the grid avoids wrong reuse on all three sets.")
    else:
        lines.append(f"Default threshold: {rep['chosen_threshold']:.2f}, the lowest value "
                     f"with no wrong reuse on any set and at least {MARGIN:.2f} above the "
                     f"most similar wrong question ({rep['highest_wrong_similarity']:.4f}). "
                     "It is chosen on these same questions, so wrong reuse on new traffic "
                     "can be higher than zero.")
    worst = rep["closest_near_misses"]
    lines += ["", "Closest near misses (the questions that set the threshold):", "",
              "| Similarity | Near miss | Cached question |", "|---|---|---|"]
    lines += [f"| {w['similarity']:.3f} | {w['question']} | {w['cached']} |" for w in worst]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    from ragsvc.bundle import load_bundle
    from ragsvc.config import load_settings
    from ragsvc.evalset import load_questions
    from ragsvc.models import STEmbedder, STReranker
    from ragsvc.search import Searcher

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--eval-dir", type=Path, default=Path("eval"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)
    cfg = load_settings(args.config)
    e, r = cfg.embedding, cfg.retrieval
    embedder = STEmbedder(e.model, e.query_prefix, e.batch_size, e.device)
    searcher = Searcher(load_bundle(cfg.index_dir, cfg), r, embedder,
                        STReranker(r.reranker, e.device))
    seed = load_questions(cfg.data_dir / "eval" / "questions.jsonl")
    by_id = {q.id: q for q in seed}
    paras = load_jsonl(args.eval_dir / "paraphrases.jsonl")
    nears = load_jsonl(args.eval_dir / "near_misses.jsonl")
    seed_ids = [q.id for q in seed]

    seed_vecs = embedder.encode_queries([q.question for q in seed])
    para_vecs = embedder.encode_queries([p["question"] for p in paras])
    near_vecs = embedder.encode_queries([p["question"] for p in nears])
    para_idx, para_sim = nearest(para_vecs, seed_vecs)
    near_idx, near_sim = nearest(near_vecs, seed_vecs)
    _, distinct_sim = nearest(seed_vecs, seed_vecs, exclude_self=True)
    own = np.array([float(para_vecs[i] @ seed_vecs[seed_ids.index(p["id"])])
                    for i, p in enumerate(paras)])

    def rels(hits, q) -> list[bool]:
        out = [is_relevant(h.chunk, q) for h in hits]
        return out + [False] * (K - len(out))

    searcher.search(seed[0].question, r.mode, K)  # warm-up
    cached_hits = {q.id: searcher.search(q.question, r.mode, K).hits for q in seed}
    fresh, seed_rels_for = {}, {}
    for p, j in zip(paras, para_idx, strict=True):
        q = by_id[p["id"]]
        fresh[p["id"]] = rels(searcher.search(p["question"], r.mode, K).hits, q)
        seed_rels_for[(p["id"], seed_ids[j])] = rels(cached_hits[seed_ids[j]], q)
    original = [rels(cached_hits[p["id"]], by_id[p["id"]]) for p in paras]

    def agg(rows) -> dict:
        return {"hit@5": sum(hit_at_k(x, 5) for x in rows) / len(rows),
                "mrr@10": sum(mrr_at_k(x, K) for x in rows) / len(rows)}

    rows = sweep(seed_ids, [p["id"] for p in paras], para_idx, para_sim, near_sim,
                 distinct_sim, fresh, seed_rels_for)
    order = np.argsort(-near_sim)[:5]
    wrong_para = para_sim[np.array([seed_ids[j] != p["id"]
                                    for j, p in zip(para_idx, paras, strict=True)])]
    highest_wrong = float(max(near_sim.max(), distinct_sim.max(),
                              wrong_para.max() if wrong_para.size else 0.0))
    report = {
        "embedding_model": e.model, "mode": r.mode,
        "n_seed": len(seed), "n_paraphrases": len(paras), "n_near_misses": len(nears),
        "similarity": {"paraphrase_to_source": quantiles(own), "near_miss": quantiles(near_sim),
                       "distinct": quantiles(distinct_sim)},
        "baselines": {"paraphrase_fresh": agg(list(fresh.values())), "original": agg(original)},
        "sweep": rows, "highest_wrong_similarity": highest_wrong,
        "chosen_threshold": pick_threshold(rows, highest_wrong),
        "closest_near_misses": [{"similarity": float(near_sim[i]),
                                 "question": nears[i]["question"],
                                 "cached": seed[near_idx[i]].question} for i in order],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "semantic_cache.json").write_text(json.dumps(report, indent=1) + "\n")
    (args.out / "semantic_cache.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
