#!/usr/bin/env python3
"""Quick-QA CLI: ingest → index → query → eval."""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Avoid OpenMP / PyTorch thread interaction segfaults on some platforms (e.g. Python 3.14 + faiss).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse

try:
    import torch

    torch.set_num_threads(1)
except Exception:
    pass
import logging
import sys
from pathlib import Path

import yaml

from pipeline.chunker import chunk_documents
from pipeline.encoder import build_encoder
from pipeline.index import build_index, load_index
from pipeline.ingestion import load_documents, load_squad_passages
from pipeline.reader import Reader
from pipeline.retrieval import Retriever
from pipeline.scorer import Scorer

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_paths(cfg: dict, root: Path) -> dict:
    paths = cfg.setdefault("paths", {})
    for key in ("corpus_dir", "index_dir", "faiss_index", "passage_store"):
        if key not in paths:
            continue
        p = Path(paths[key])
        if not p.is_absolute():
            paths[key] = str((root / p).resolve())
    return cfg


def corpus_has_files(corpus_dir: Path) -> bool:
    if not corpus_dir.is_dir():
        return False
    for p in corpus_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".pdf", ".docx"}:
            return True
    return False


def cmd_ingest(config_path: Path, verbose: bool) -> None:
    cfg = expand_paths(load_config(config_path), PROJECT_ROOT)
    ch = cfg.get("chunker", {})
    paths = cfg["paths"]

    corpus_dir = Path(paths["corpus_dir"])
    if corpus_has_files(corpus_dir):
        docs = load_documents(corpus_dir)
    else:
        logging.info("Corpus empty or no supported files — loading SQuAD passages.")
        n = int(cfg.get("evaluation", {}).get("num_eval_samples", 500))
        docs = load_squad_passages("validation", n=n)

    if not docs:
        logging.error("No documents to index.")
        sys.exit(1)

    chunks = chunk_documents(
        docs,
        chunk_size=int(ch.get("chunk_size", 300)),
        overlap=int(ch.get("chunk_overlap", 50)),
        split_by=str(ch.get("split_by", "sentence")),
    )
    if not chunks:
        logging.error("No chunks produced.")
        sys.exit(1)

    enc_cfg = cfg["encoder"]
    encoder = build_encoder(
        enc_cfg["backend"],
        **{k: v for k, v in enc_cfg.items() if k != "backend"},
    )
    build_index(chunks, encoder, cfg)
    dim = encoder.embedding_dim
    idx_path = Path(paths["faiss_index"])
    logging.info(
        "Ingest complete: %d documents → %d chunks, embedding_dim=%d, index at %s",
        len(docs),
        len(chunks),
        dim,
        idx_path,
    )


def cmd_query(config_path: Path, query: str, verbose: bool) -> None:
    cfg = expand_paths(load_config(config_path), PROJECT_ROOT)
    try:
        faiss_idx, bm25_idx, store = load_index(cfg)
    except FileNotFoundError as e:
        logging.error("%s Run 'python main.py ingest' first.", e)
        sys.exit(1)

    enc_cfg = cfg["encoder"]
    encoder = build_encoder(
        enc_cfg["backend"],
        **{k: v for k, v in enc_cfg.items() if k != "backend"},
    )
    retriever = Retriever(
        faiss_idx,
        bm25_idx,
        store,
        encoder,
        cfg,
    )
    scorer = Scorer(encoder)
    r_cfg = cfg["reader"]
    reader = Reader(
        r_cfg["model_name"],
        r_cfg["device"],
        int(r_cfg.get("max_answer_length", 50)),
    )

    ret_cfg = cfg["retrieval"]
    cands = retriever.retrieve(
        query,
        top_k_faiss=int(ret_cfg.get("top_k_faiss", 100)),
        top_k_bm25=int(ret_cfg.get("top_k_bm25", 100)),
        use_hybrid=bool(ret_cfg.get("use_bm25_hybrid", True)),
    )
    top_rr = int(cfg.get("scorer", {}).get("top_k_rerank", 5))
    reranked = scorer.rerank(query, cands, top_k=top_rr)
    ans = reader.read(query, reranked)

    print()
    print("Answer:", ans.text)
    print("Score:", f"{ans.score:.4f}")
    print("Source:", ans.source_doc)
    print("Chunk:", ans.chunk_id)
    print()
    print("Passage excerpt:", ans.passage[:500] + ("..." if len(ans.passage) > 500 else ""))
    print()


def cmd_eval(config_path: Path, n: int, verbose: bool) -> None:
    from evaluate import run_evaluation

    cfg = expand_paths(load_config(config_path), PROJECT_ROOT)
    run_evaluation(cfg, PROJECT_ROOT, n_samples=n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick-QA RAG pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Build corpus chunks and FAISS+BM25 index")

    p_q = sub.add_parser("query", help="Ask a question")
    p_q.add_argument("question", nargs="+", help="Query string")

    p_e = sub.add_parser("eval", help="Run SQuAD evaluation")
    p_e.add_argument("--dataset", default="squad", choices=["squad", "custom"])
    p_e.add_argument("--n", type=int, default=200, help="Number of examples")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config_path = args.config.resolve()
    if not config_path.is_file():
        logging.error("Config not found: %s", config_path)
        sys.exit(1)

    if args.command == "ingest":
        cmd_ingest(config_path, args.verbose)
    elif args.command == "query":
        q = " ".join(args.question)
        cmd_query(config_path, q, args.verbose)
    elif args.command == "eval":
        if args.dataset != "squad":
            logging.error("Only squad eval is implemented.")
            sys.exit(1)
        cmd_eval(config_path, args.n, args.verbose)


if __name__ == "__main__":
    main()
