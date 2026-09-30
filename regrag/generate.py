"""Grounded answer generation with clause-level citations and abstention."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from . import config
from .retrieve import Hit

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"
ABSTAIN_MESSAGE = ("I can't answer this from the indexed regulations - the retrieved clauses don't "
                   "contain the information. Check the latest SERC orders directly.")

SYSTEM_PROMPT = f"""You are a regulatory analyst assistant for Indian renewable-energy regulations.
Answer ONLY from the numbered sources. Rules:
1. Start with a direct answer to the question (e.g. "Yes", "No", or the number/limit), then explain briefly.
2. Base the answer on CURRENT SOURCES. OLDER OR DRAFT SOURCES are for context only: if one of them
   says something different, say that the earlier/draft provision differs.
3. End every factual sentence with its citation, e.g. [S2]. Cite only sources you used.
4. Copy numbers (kW, MW, %, Rs, days, dates) exactly as written.
5. Name the clause number and the regulation/order with its date for the key fact.
6. If the only relevant source is a DRAFT, say it is a draft proposal and not law in force.
7. If no source answers the question, reply with only: {ABSTAIN_TOKEN}
8. Never use outside knowledge. At most 5 sentences."""


@dataclass
class Answer:
    text: str
    abstained: bool
    citations: list[dict] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)


def llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL,
                      temperature=temperature, num_ctx=6144)


def is_background(h: Hit) -> bool:
    """Superseded or draft material is shown to the model as context, not as the answer."""
    return h.chunk.status == "draft" or any(f.startswith("POSSIBLY SUPERSEDED") for f in h.flags)


def source_header(i: int, h: Hit) -> str:
    c = h.chunk
    clause = f"Clause {c.clause}" if c.clause else "Preamble/untitled section"
    sec = f" ({c.section_title})" if c.section_title else ""
    return f"[S{i}] State: {c.state} | {c.authority} | {c.title} | {clause}{sec} | dated {c.date} | page {c.page}"


def format_sources(hits: list[Hit], max_chars: int = 1200) -> str:
    """Numbered sources, split into CURRENT and OLDER/DRAFT sections (numbering follows `hits`)."""
    current, background = [], []
    for i, h in enumerate(hits, 1):
        flags = "".join(f"\nFLAG: {f}" for f in h.flags)
        block = f"{source_header(i, h)}{flags}\n{h.chunk.text[:max_chars]}"
        (background if is_background(h) else current).append(block)
    out = "CURRENT SOURCES:\n\n" + ("\n\n".join(current) or "(none)")
    if background:
        out += "\n\nOLDER OR DRAFT SOURCES (context only):\n\n" + "\n\n".join(background)
    return out


def citation(i: int, h: Hit) -> dict:
    c = h.chunk
    return {"id": f"S{i}", "state": c.state, "title": c.title, "clause": c.clause, "date": c.date,
            "page": c.page, "status": c.status, "doc_type": c.doc_type, "source_url": c.source_url,
            "flags": list(h.flags), "rerank": round(h.rerank, 2), "text": c.text}


CITE_GROUP_RE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")


def cited_ids(text: str) -> list[int]:
    """Source numbers cited as [S2], or grouped as [S2, S3] / [S2; S3]."""
    return [int(n) for group in CITE_GROUP_RE.findall(text) for n in re.findall(r"S(\d+)", group)]


def generate(question: str, hits: list[Hit], model: ChatOllama | None = None) -> Answer:
    contexts = [h.chunk.text for h in hits]
    best = max((h.rerank for h in hits), default=float("-inf"))
    if not hits or best < config.ABSTAIN_SCORE:
        return Answer(ABSTAIN_MESSAGE, True, [], contexts)

    prompt = f"{format_sources(hits)}\n\nQUESTION: {question}\n\nANSWER:"
    reply = (model or llm()).invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]).content.strip()

    cited = sorted({n for n in cited_ids(reply) if 0 < n <= len(hits)})
    # Small models sometimes append the abstain token to a real answer; keep the answer, drop the token.
    abstain_line = re.compile(rf"^.*{ABSTAIN_TOKEN.replace('_', '[_ ]')}.*$", re.I | re.M)
    if abstain_line.search(reply):
        if not cited:
            return Answer(ABSTAIN_MESSAGE, True, [], contexts)
        reply = abstain_line.sub("", reply).strip()
    return Answer(reply, False, [citation(i, hits[i - 1]) for i in cited], contexts)
