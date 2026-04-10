"""Minimal Streamlit UI for Quick-QA (optional)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import streamlit as st
import yaml

from pipeline.encoder import build_encoder
from pipeline.index import load_index
from pipeline.reader import Reader
from pipeline.retrieval import Retriever
from pipeline.scorer import Scorer

PROJECT_ROOT = Path(__file__).resolve().parent


def load_cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_paths(cfg: dict, root: Path) -> None:
    paths = cfg.setdefault("paths", {})
    for key in ("corpus_dir", "index_dir", "faiss_index", "passage_store"):
        if key not in paths:
            continue
        p = Path(paths[key])
        if not p.is_absolute():
            paths[key] = str((root / p).resolve())


def main() -> None:
    st.set_page_config(page_title="Quick-QA", layout="wide")
    st.title("Quick-QA")

    cfg_path = PROJECT_ROOT / "config.yaml"
    cfg = load_cfg(cfg_path)
    expand_paths(cfg, PROJECT_ROOT)

    enc = cfg["encoder"]
    backend = st.sidebar.selectbox("Encoder backend", ["minilm", "trm", "cnn"], index=0)
    enc["backend"] = backend
    top_k = st.sidebar.slider("Retrieve top-k (before rerank)", 10, 200, 100)

    query = st.text_input("Question", placeholder="What is ...?")
    if st.button("Ask") and query.strip():
        try:
            faiss_idx, bm25_idx, store = load_index(cfg)
        except FileNotFoundError as e:
            st.error(f"{e} Run `python main.py ingest` first.")
            return

        encoder = build_encoder(
            enc["backend"],
            model_name=enc["model_name"],
            device=enc["device"],
        )
        retriever = Retriever(faiss_idx, bm25_idx, store, encoder, cfg)
        scorer = Scorer(encoder)
        r_cfg = cfg["reader"]
        reader = Reader(r_cfg["model_name"], r_cfg["device"], int(r_cfg.get("max_answer_length", 50)))

        ret_cfg = cfg["retrieval"]
        t0 = time.perf_counter()
        cands = retriever.retrieve(
            query,
            top_k_faiss=min(top_k, int(ret_cfg.get("top_k_faiss", 100))),
            top_k_bm25=int(ret_cfg.get("top_k_bm25", 100)),
            use_hybrid=bool(ret_cfg.get("use_bm25_hybrid", True)),
        )
        t1 = time.perf_counter()
        rr = int(cfg.get("scorer", {}).get("top_k_rerank", 5))
        reranked = scorer.rerank(query, cands, top_k=rr)
        t2 = time.perf_counter()
        ans = reader.read(query, reranked)
        t3 = time.perf_counter()

        st.subheader("Answer")
        st.write(ans.text)
        st.caption(f"Confidence: {ans.score:.4f} | Source: {ans.source_doc} | Chunk: {ans.chunk_id}")
        st.caption(
            f"Timing ms — retrieve: {(t1-t0)*1000:.0f}, rerank: {(t2-t1)*1000:.0f}, read: {(t3-t2)*1000:.0f}"
        )

        st.subheader("Top passages")
        for i, r in enumerate(reranked[:3]):
            text = r.chunk.text
            st.markdown(f"**[{i+1}]** (score {r.score:.4f})")
            st.text_area("passage", text[:2000], height=120, key=f"passage_{i}_{hash(text[:20])}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
