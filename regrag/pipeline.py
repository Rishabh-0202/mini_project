"""Linear RAG pipeline: understand -> retrieve (per state) -> amendment check -> generate.

    python -m regrag.pipeline "What is the net metering limit in Rajasthan?"
"""
from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from . import config
from .amendments import AmendmentChecker
from .generate import CITE_GROUP_RE, Answer, generate, llm
from .query import SubQuery, decompose, mentions_central, out_of_scope_message
from .retrieve import Hit, HybridRetriever


@dataclass
class Result:
    question: str
    answer: str
    abstained: bool
    sub_queries: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def components() -> tuple[HybridRetriever, AmendmentChecker]:
    r = HybridRetriever()
    return r, AmendmentChecker(r)


def state_scope(sq: SubQuery, question: str) -> list[str] | None:
    """Metadata filter for a sub-query: its state, plus central rules when asked about."""
    if sq.state is None:
        return None
    scope = [sq.state]
    if mentions_central(question):
        scope.append(config.CENTRAL)
    return scope


def retrieve_for(sq: SubQuery, question: str) -> list[Hit]:
    retriever, checker = components()
    hits = retriever.retrieve(sq.text, state_scope(sq, question))
    return checker.check(sq.text, hits)


def answer(question: str) -> Result:
    if msg := out_of_scope_message(question):
        return Result(question, msg, True)
    subs = decompose(question)
    model = llm()
    parts: list[tuple[SubQuery, Answer]] = []
    for sq in subs:
        hits = retrieve_for(sq, question)
        parts.append((sq, generate(sq.text if len(subs) > 1 else question, hits, model)))
    return combine(question, parts)


def combine(question: str, parts: list[tuple[SubQuery, Answer]]) -> Result:
    """Merge per-state answers. Citation ids are renumbered so they stay unique."""
    if len(parts) == 1:
        sq, a = parts[0]
        return Result(question, a.text, a.abstained, [asdict(sq)], a.citations, a.contexts)

    lines, citations, contexts = [], [], []
    for sq, a in parts:
        prefix = sq.state or "All states"
        tag = prefix[:2].upper()
        # S1 -> S1-GU inside every citation bracket, including grouped ones like [S1, S3]
        text = CITE_GROUP_RE.sub(lambda m: "[" + re.sub(r"(S\d+)", rf"\1-{tag}", m.group(1)) + "]", a.text)
        citations.extend({**c, "id": f"{c['id']}-{tag}"} for c in a.citations)
        lines.append(f"**{prefix}:** {text}")
        contexts.extend(a.contexts)
    return Result(question, "\n\n".join(lines), all(a.abstained for _, a in parts),
                  [asdict(sq) for sq, _ in parts], citations, contexts)


def main() -> None:
    q = " ".join(sys.argv[1:]) or "What is the maximum net metering capacity in Rajasthan?"
    r = answer(q)
    print(f"\nQ: {r.question}\nSub-queries: {[s['text'] for s in r.sub_queries]}\n\n{r.answer}\n")
    for c in r.citations:
        print(f"  [{c['id']}] {c['state']} | {c['title']} | clause {c['clause']} | {c['date']} | p.{c['page']}")
        for f in c["flags"]:
            print(f"       ! {f}")


if __name__ == "__main__":
    main()
