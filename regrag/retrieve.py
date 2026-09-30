"""Hybrid retrieval: BM25 + dense, state-filtered, fused with RRF, reranked by a cross-encoder."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from . import config
from .index import BM25Index, load_vectorstore
from .ingest import Chunk, load_chunks


@dataclass
class Hit:
    chunk: Chunk
    rrf: float = 0.0
    rerank: float = float("-inf")
    sources: set[str] = field(default_factory=set)      # {"bm25", "dense", "amendment"}
    flags: list[str] = field(default_factory=list)       # amendment / draft warnings


class HybridRetriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25Index(self.chunks)
        self.store = load_vectorstore()

    @cached_property
    def reranker(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(config.RERANK_MODEL, max_length=512)

    # --- candidate generation -------------------------------------------------
    def _dense(self, query: str, states: list[str] | None, k: int) -> list[Chunk]:
        flt = {"state": {"$in": states}} if states else None
        docs = self.store.similarity_search(query, k=k, filter=flt)
        return [self.by_id[d.metadata["chunk_id"]] for d in docs if d.metadata["chunk_id"] in self.by_id]

    def candidates(self, query: str, states: list[str] | None) -> list[Hit]:
        """Reciprocal-rank fusion of BM25 and dense rankings.

        The metadata filter is applied inside both retrievers, so a Gujarat
        question can never be answered from a Maharashtra clause.
        """
        hits: dict[str, Hit] = {}
        for name, ranked in (
            ("bm25", [c for c, _ in self.bm25.search(query, states, config.SPARSE_K)]),
            ("dense", self._dense(query, states, config.DENSE_K)),
        ):
            for rank, c in enumerate(ranked):
                h = hits.setdefault(c.chunk_id, Hit(c))
                h.rrf += 1.0 / (config.RRF_K + rank + 1)
                h.sources.add(name)
        return sorted(hits.values(), key=lambda h: h.rrf, reverse=True)

    # --- reranking ------------------------------------------------------------
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = self.reranker.predict([(query, h.chunk.embed_text) for h in hits], show_progress_bar=False)
        for h, s in zip(hits, scores):
            h.rerank = float(s)
        return sorted(hits, key=lambda h: h.rerank, reverse=True)

    def retrieve(self, query: str, states: list[str] | None, top_n: int = config.RERANK_TOP_N,
                 pool: int = 30) -> list[Hit]:
        return self.rerank(query, self.candidates(query, states)[:pool])[:top_n]
