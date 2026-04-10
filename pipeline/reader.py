"""Extractive question answering over retrieved passages."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from pipeline.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    text: str
    score: float
    passage: str
    source_doc: str
    chunk_id: str
    start: int
    end: int


class Reader:
    def __init__(self, model_name: str, device: str, max_answer_length: int) -> None:
        self._model_name = model_name
        self._device = device
        self._max_len = max_answer_length
        self._tokenizer = None
        self._model = None
        self._fallback = False

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForQuestionAnswering.from_pretrained(self._model_name)
            if self._device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            elif self._device == "mps" and torch.backends.mps.is_available():
                self._model = self._model.to("mps")
            self._model.eval()
        except Exception as e:
            logger.error("Failed to load QA model %s: %s", self._model_name, e)
            self._fallback = True

    def _qa_one(self, question: str, context: str) -> tuple[str, float, int, int]:
        assert self._tokenizer is not None and self._model is not None
        device = next(self._model.parameters()).device
        enc = self._tokenizer(
            question,
            context,
            truncation=True,
            max_length=384,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self._model(**enc)
        start_logits = out.start_logits[0]
        end_logits = out.end_logits[0]
        seq_len = enc["input_ids"].shape[1]
        si = int(torch.argmax(start_logits))
        rest = end_logits[si : min(si + self._max_len + 1, seq_len)]
        rel = int(torch.argmax(rest))
        ej = si + rel
        score = float(start_logits[si] + end_logits[ej])
        input_ids = enc["input_ids"][0]
        span_ids = input_ids[si : ej + 1]
        text = self._tokenizer.decode(span_ids, skip_special_tokens=True).strip()
        offsets = offset_mapping[0]
        char_start = int(offsets[si][0]) if si < len(offsets) else 0
        char_end = int(offsets[ej][1]) if ej < len(offsets) else len(context)
        return text, score, char_start, char_end

    def read(self, query: str, passages: list[RetrievalResult]) -> Answer:
        self._ensure_model()
        if self._fallback or not passages:
            top = passages[0] if passages else None
            txt = (
                f"[FALLBACK] {top.chunk.text[:200]}"
                if top
                else "[FALLBACK] No passages."
            )
            return Answer(
                text=txt,
                score=0.0,
                passage=top.chunk.text if top else "",
                source_doc=top.chunk.source if top else "",
                chunk_id=top.chunk.chunk_id if top else "",
                start=0,
                end=0,
            )

        best: tuple[str, float, str, str, str, int, int] | None = None
        for r in passages:
            ctx = r.chunk.text
            try:
                ans, sc, cs, ce = self._qa_one(query, ctx)
                if not ans:
                    continue
                if best is None or sc > best[1]:
                    best = (ans, sc, ctx, r.chunk.source, r.chunk.chunk_id, cs, ce)
            except Exception as e:
                logger.warning("QA forward failed on chunk %s: %s", r.chunk.chunk_id, e)

        if best is None:
            top = passages[0]
            return Answer(
                text=f"[FALLBACK] {top.chunk.text[:200]}",
                score=0.0,
                passage=top.chunk.text,
                source_doc=top.chunk.source,
                chunk_id=top.chunk.chunk_id,
                start=0,
                end=0,
            )

        ans, sc, ctx, src, cid, cs, ce = best
        return Answer(
            text=ans,
            score=float(sc),
            passage=ctx,
            source_doc=src,
            chunk_id=cid,
            start=cs,
            end=ce,
        )

    def read_batch(
        self,
        queries: list[str],
        passages_per_query: list[list[RetrievalResult]],
    ) -> list[Answer]:
        return [self.read(q, p) for q, p in zip(queries, passages_per_query)]
