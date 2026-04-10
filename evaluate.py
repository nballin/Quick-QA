"""Evaluation: EM, F1, recall@k on SQuAD."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from pipeline.encoder import build_encoder
from pipeline.index import load_index
from pipeline.reader import Reader
from pipeline.retrieval import Retriever
from pipeline.scorer import Scorer

logger = logging.getLogger(__name__)


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # drop leading articles (optional)
    for art in ("the ", "a ", "an "):
        if s.startswith(art):
            s = s[len(art) :]
    return s


def exact_match(prediction: str, ground_truths: list[str]) -> float:
    p = normalize_answer(prediction)
    for g in ground_truths:
        if p == normalize_answer(g):
            return 1.0
    return 0.0


def _f1_single(prediction: str, ground_truth: str) -> float:
    pred_t = normalize_answer(prediction).split()
    gold_t = normalize_answer(ground_truth).split()
    if not pred_t or not gold_t:
        return int(pred_t == gold_t)
    common = 0
    g_counts: dict[str, int] = {}
    for t in gold_t:
        g_counts[t] = g_counts.get(t, 0) + 1
    p_counts: dict[str, int] = {}
    for t in pred_t:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t, c in p_counts.items():
        common += min(c, g_counts.get(t, 0))
    if common == 0:
        return 0.0
    precision = common / len(pred_t)
    recall = common / len(gold_t)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, ground_truths: list[str]) -> float:
    return max(_f1_single(prediction, g) for g in ground_truths) if ground_truths else 0.0


def recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    top = retrieved_chunk_ids[:k]
    return 1.0 if relevant_chunk_ids & set(top) else 0.0


def _relevant_chunk_ids_for_example(
    answers: list[str],
    context: str,
    passage_store: Any,
) -> set[str]:
    """Chunks whose text contains any gold answer, or overlap gold context."""
    ans_l = [a.lower().strip() for a in answers if a.strip()]
    out: set[str] = set()
    ctx_snip = context[:120].lower().strip()
    for ch in passage_store.all():
        t = ch.text.lower()
        if any(a in t for a in ans_l):
            out.add(ch.chunk_id)
        elif ctx_snip and ctx_snip in t:
            out.add(ch.chunk_id)
    return out


def evaluate_squad(
    n_samples: int,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    split = config.get("evaluation", {}).get("squad_split", "validation")
    eval_k = config.get("evaluation", {}).get("eval_k", [1, 5, 10])
    ds = load_dataset("rajpurkar/squad", split=split)

    enc_cfg = config["encoder"]
    encoder = build_encoder(
        enc_cfg["backend"],
        model_name=enc_cfg["model_name"],
        device=enc_cfg["device"],
    )
    faiss_idx, bm25_idx, store = load_index(config)
    retriever = Retriever(
        faiss_idx,
        bm25_idx,
        store,
        encoder,
        config,
    )
    scorer = Scorer(encoder)
    r_cfg = config["reader"]
    reader = Reader(
        r_cfg["model_name"],
        r_cfg["device"],
        int(r_cfg.get("max_answer_length", 50)),
    )

    ret_cfg = config["retrieval"]
    top_f = int(ret_cfg.get("top_k_faiss", 100))
    top_b = int(ret_cfg.get("top_k_bm25", 100))
    hybrid = bool(ret_cfg.get("use_bm25_hybrid", True))
    top_rr = int(config.get("scorer", {}).get("top_k_rerank", 5))

    ems: list[float] = []
    f1s: list[float] = []
    recalls: dict[int, list[float]] = {k: [] for k in eval_k}

    per_example: list[dict[str, Any]] = []

    for i, row in enumerate(ds):
        if i >= n_samples:
            break
        question = row["question"]
        answers = row["answers"]["text"]
        context = row["context"]
        if not answers:
            continue

        rel_ids = _relevant_chunk_ids_for_example(list(answers), context, store)

        cands = retriever.retrieve(
            question,
            top_k_faiss=top_f,
            top_k_bm25=top_b,
            use_hybrid=hybrid,
        )
        reranked = scorer.rerank(question, cands, top_k=top_rr)
        chunk_ids_ret = [r.chunk.chunk_id for r in cands]
        for k in eval_k:
            recalls[k].append(
                recall_at_k(chunk_ids_ret, rel_ids, k) if rel_ids else 0.0,
            )

        ans = reader.read(question, reranked)
        ems.append(exact_match(ans.text, list(answers)))
        f1s.append(f1_score(ans.text, list(answers)))

        per_example.append(
            {
                "id": row.get("id", i),
                "question": question,
                "prediction": ans.text,
                "answers": answers,
                "em": ems[-1],
                "f1": f1s[-1],
            }
        )

    out = {
        "exact_match": sum(ems) / max(len(ems), 1),
        "f1": sum(f1s) / max(len(f1s), 1),
        "n": len(ems),
    }
    for k in eval_k:
        out[f"recall@{k}"] = sum(recalls[k]) / max(len(recalls[k]), 1)

    out["per_example"] = per_example
    return out


def print_results_table(metrics: dict[str, Any]) -> None:
    rows = [
        ("Exact Match", metrics.get("exact_match", 0)),
        ("F1", metrics.get("f1", 0)),
    ]
    recall_keys = [k for k in metrics if k.startswith("recall@")]
    for key in sorted(recall_keys, key=lambda x: int(x.split("@")[1])):
        rows.append((key.replace("recall@", "Recall@"), metrics[key]))

    print("┌─────────────────────┬────────┐")
    print("│ Metric              │ Score  │")
    print("├─────────────────────┼────────┤")
    for name, val in rows:
        print(f"│ {name:<19} │ {val:.4f} │")
    print("└─────────────────────┴────────┘")


def run_evaluation(
    config: dict[str, Any],
    project_root: Path,
    n_samples: int,
    output_path: Path | None = None,
) -> dict[str, Any]:
    metrics = evaluate_squad(n_samples, config, project_root)
    print_results_table(metrics)
    out_path = output_path or (project_root / "eval_results.json")
    # drop large per_example for json summary optional
    save = {k: v for k, v in metrics.items() if k != "per_example"}
    save["per_example_count"] = len(metrics.get("per_example", []))
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(save, f, indent=2)
    logger.info("Wrote %s", out_path)
    with (project_root / "eval_results_full.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
