"""Grounded answer generation with clause-level citations and abstention."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from .retrieve import Hit

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"
ABSTAIN_MESSAGE = ("I can't answer this from the indexed regulations - the retrieved clauses don't "
                   "contain the information. Check the latest SERC orders directly.")

SYSTEM_PROMPT = f"""You are a regulatory analyst assistant for Indian renewable-energy regulations.
Answer ONLY from the numbered sources. Rules:
1. Open with the direct answer: Yes or No for a yes/no question, otherwise the figure or limit asked for. Then explain briefly.
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


def llm(temperature: float = 0.0, num_predict: int = 512) -> BaseChatModel:
    """The answer model, chosen by config.LLM_PROVIDER.

    Groq: hosted gpt-oss-120b (reliable, fast). Ollama: local llama3.2 (offline); num_predict caps its
    output because a small model in JSON mode can pad with whitespace for minutes."""
    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=config.GROQ_MODEL, temperature=temperature, max_retries=3)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL,
                      temperature=temperature, num_ctx=4096, num_predict=num_predict)


def tidy(text: str) -> str:
    """Normalise model output so citations parse: 【S1】, [ S1 ] and [S2†L3-L9] all become [S1]/[S2];
    non-breaking hyphens and spaces become plain ones."""
    text = text.replace("【", "[").replace("】", "]")
    for odd, plain in {"‑": "-", "‐": "-", " ": " ", " ": " "}.items():
        text = text.replace(odd, plain)
    text = re.sub(r"(S\d+)†[^\],;]*", r"\1", text)          # [S2†L3-L9] -> [S2]
    text = re.sub(r"\[\s+(?=S\d)", "[", text)               # [ S1 ] -> [S1 ]
    return re.sub(r"(S\d+)\s+\]", r"\1]", text)              # [S1 ]  -> [S1]


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


# Any [...] group that contains a source id: [S2], [S2, S3], [S2; S3], [S2-Clause 3, 2024-09-04]
CITE_GROUP_RE = re.compile(r"\[([^\[\]]*?\bS\d+[^\[\]]*)\]")


def cited_ids(text: str) -> list[int]:
    """Source numbers cited inside square brackets, e.g. [S2], [S2, S3], [S2-Clause 3]."""
    return [int(n) for group in CITE_GROUP_RE.findall(text) for n in re.findall(r"\bS(\d+)", group)]


def generate(question: str, hits: list[Hit], model: BaseChatModel | None = None) -> Answer:
    contexts = [h.chunk.text for h in hits]
    best = max((h.rerank for h in hits), default=float("-inf"))
    if not hits or best < config.ABSTAIN_SCORE:
        return Answer(ABSTAIN_MESSAGE, True, [], contexts)

    prompt = f"{format_sources(hits)}\n\nQUESTION: {question}\n\nANSWER:"
    reply = tidy((model or llm()).invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]).content.strip())

    cited = sorted({n for n in cited_ids(reply) if 0 < n <= len(hits)})
    # Small models sometimes append the abstain token to a real answer; keep the answer, drop the token.
    abstain_line = re.compile(rf"^.*{ABSTAIN_TOKEN.replace('_', '[_ ]')}.*$", re.I | re.M)
    if abstain_line.search(reply):
        if not cited:
            return Answer(ABSTAIN_MESSAGE, True, [], contexts)
        reply = abstain_line.sub("", reply).strip()
    return Answer(reply, False, [citation(i, hits[i - 1]) for i in cited], contexts)
