"""Amendment handling: find later documents that change a retrieved clause.

For every retrieved clause from an earlier document we look for later documents
in the same state that amend it (per manifest `amends`) and refer to that clause
(or its parent regulation). Matches are:
  * flagged on the older hit ("may be superseded by ..."),
  * pulled into the context if retrieval missed them,
  * ranked ahead of the clause they amend (temporal ranking).
Draft documents are always flagged and never treated as superseding final law.
"""
from __future__ import annotations

from .ingest import Chunk
from .retrieve import Hit, HybridRetriever

MAX_PULLED_PER_HIT = 2


def _amends_chain(doc_id: str, docs_amending: dict[str, set[str]]) -> set[str]:
    """All documents that (transitively) amend doc_id."""
    out, stack = set(), [doc_id]
    while stack:
        for d in docs_amending.get(stack.pop(), ()):
            if d not in out:
                out.add(d)
                stack.append(d)
    return out


def _refers_to(amending: Chunk, clause: str) -> bool:
    if not clause:
        return False
    if amending.doc_type == "consolidated_regulation":
        return amending.clause == clause
    major = clause.split(".")[0]
    for ref in amending.refs:
        # "Regulation 7 substituted" covers 7.x; "Clause 5.1" covers 5.1 and 5.1.x
        if ref == clause or ref == major or clause.startswith(ref + "."):
            return True
    return False


class AmendmentChecker:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.docs_amending: dict[str, set[str]] = {}
        self.chunks_by_doc: dict[str, list[Chunk]] = {}
        for c in retriever.chunks:
            self.chunks_by_doc.setdefault(c.doc_id, []).append(c)
            if c.amends:
                self.docs_amending.setdefault(c.amends, set()).add(c.doc_id)

    def amending_chunks(self, chunk: Chunk) -> list[Chunk]:
        out = []
        for doc_id in _amends_chain(chunk.doc_id, self.docs_amending):
            for a in self.chunks_by_doc.get(doc_id, []):
                if a.date > chunk.date and a.status != "draft" and _refers_to(a, chunk.clause):
                    out.append(a)
        return out

    def check(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Annotate hits with amendment/draft flags, pull in missing amendments, apply temporal ranking."""
        present = {h.chunk.chunk_id: h for h in hits}
        pulled: list[Hit] = []
        for h in hits:
            c = h.chunk
            if c.status == "draft":
                h.flags.append(f"DRAFT - '{c.title}' is a draft and not yet in force.")
            if c.status == "summary":
                h.flags.append("SECONDARY SOURCE - discom summary of the order, not the order text itself.")
            later = self.amending_chunks(c)
            if not later:
                continue
            newest = max(later, key=lambda a: a.date)
            h.flags.append(
                f"POSSIBLY SUPERSEDED - clause {c.clause} ({c.date}) is amended/revisited by "
                f"'{newest.title}' dated {newest.date}; prefer the later provision."
            )
            missing = [a for a in later if a.chunk_id not in present]
            if missing:
                scored = self.retriever.rerank(query, [Hit(a, sources={"amendment"}) for a in missing])
                for ah in scored[:MAX_PULLED_PER_HIT]:
                    ah.flags.append(f"AMENDMENT - amends clause {c.clause} of '{c.title}'.")
                    present[ah.chunk.chunk_id] = ah
                    pulled.append(ah)
        return temporal_rank(hits + pulled)


def temporal_rank(hits: list[Hit]) -> list[Hit]:
    """Order by relevance, but never put a superseded clause ahead of the amendment that replaces it.

    Within each (state, clause) the newest final document wins; superseded
    clauses sink below their amendments while still being shown for traceability.
    """
    def key(h: Hit):
        superseded = any(f.startswith("POSSIBLY SUPERSEDED") for f in h.flags)
        draft = h.chunk.status == "draft"
        return (superseded or draft, -h.rerank, h.chunk.date)
    return sorted(hits, key=key)
