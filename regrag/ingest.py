"""PDF -> clean text -> clause-level chunks with metadata.

Regulations are numbered ("6", "6.2", "7.3", "12. Definitions.-"), so we split on
clause headings instead of fixed windows. Every chunk carries the document's
metadata (state, authority, doc type, date, status) plus its clause number, the
title of its parent section, the page it starts on, and - for amending
documents - the principal-regulation clauses it refers to.

Run:  python -m regrag.ingest
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from pypdf import PdfReader

from . import config

# "6", "6.2", "6.2.1" optionally followed by "." then the heading/body text.
HEADING_RE = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+(?P<rest>\S.*)$")
# Words that follow a bare number in tables/quantities rather than in clause headings.
UNIT_WORDS = {"kw", "mw", "gw", "kv", "kva", "kwh", "kwp", "v", "rs", "rs.", "inr", "%", "days",
              "day", "months", "month", "years", "year", "hours", "per", "of", "to", "and", "or"}
TOC_RE = re.compile(r"(\.{5,}|…{2,}|_{5,})")
PAGE_NO_RE = re.compile(r"^\s*(page\s*\d+\s*(of\s*\d+)?|\d{1,3})\s*$", re.I)
# References an amending document makes to clauses of the principal regulation.
AMENDING_TYPES = {"amendment", "order"}
# A clause reference only counts as "amended" if an amending verb precedes it closely
# (e.g. "Regulation 7 ... shall be substituted", "increase the limit ... in regulation 7").
AMEND_VERB_RE = re.compile(r"substitut|amend|insert|omit|delet|replac|increas|revis|modif", re.I)
AMEND_WINDOW = 120
REF_RE = re.compile(r"\b(?:[Rr]egulations?|[Cc]lauses?|[Rr]ule)\s+(?:No\.\s*)?(\d{1,2}(?:\.\d{1,2}){0,2})(?![\d])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    clause: str
    section_title: str
    page: int
    state: str
    authority: str
    doc_type: str
    title: str
    date: str
    status: str
    amends: str
    source_url: str
    refs: list[str] = field(default_factory=list)

    @property
    def embed_text(self) -> str:
        """Contextual header so both BM25 and embeddings 'see' the state and document."""
        clause = f"Clause {self.clause}" if self.clause else "Preamble"
        sec = f" ({self.section_title})" if self.section_title else ""
        return f"{self.state} | {self.authority} | {self.title} | {clause}{sec}\n{self.text}"


def load_manifest() -> list[dict]:
    return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))["documents"]


def extract_lines(pdf_path) -> list[tuple[int, str]]:
    """Return (page_number, line) pairs with page furniture removed."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\t", " ").replace(" ", " ")
        lines = [re.sub(r"\s{2,}", " ", ln).strip() for ln in text.splitlines()]
        pages.append((i, [ln for ln in lines if ln]))

    # Lines repeated on many pages are running headers/footers.
    counts = Counter(ln for _, lines in pages for ln in set(lines))
    n_pages = len(pages)
    running = {ln for ln, c in counts.items() if n_pages >= 4 and c >= max(3, 0.4 * n_pages)}

    out = []
    for pno, lines in pages:
        for ln in lines:
            if ln in running or PAGE_NO_RE.match(ln) or TOC_RE.search(ln):
                continue
            if re.fullmatch(r"[_\-=\s]+", ln):
                continue
            out.append((pno, ln))
    return out


def _uses_dotted_top_level(lines: list[tuple[int, str]]) -> bool:
    """True if the document writes top-level clauses as "4. Title" rather than "4 Title".

    In such documents an undotted "7 Signing of ..." is a table row, not a clause.
    """
    dotted = undotted = 0
    for _, ln in lines:
        m = re.match(r"^(\d{1,2})(\.?)\s+[A-Z]", ln)
        if m:
            if m[2]:
                dotted += 1
            else:
                undotted += 1
    return dotted >= 3 and dotted >= undotted


def _is_heading(num: str, rest: str, last_major: int | None, line: str = "",
                dotted_style: bool = False, strict_subclauses: bool = False) -> bool:
    """Decide whether a numbered line starts a new clause.

    Clause numbers only move forward, in small steps. Numbered table rows and list
    items ("1. Upto 6 kW", "7 Signing of") restart at 1 and so fail that test.
    """
    is_sub = "." in num
    if dotted_style and not is_sub and not line.startswith(num + "."):
        return False
    first = rest.split()[0].lower().rstrip(",:;")
    if first in UNIT_WORDS:
        return False
    if not (rest[0].isupper() or rest[0] in "“\"'([‘"):
        return False
    major = int(num.split(".")[0])
    if major == 0 or major > 80:
        return False
    if last_major is None:
        return major <= 3
    if is_sub:
        if strict_subclauses:
            # Amending documents quote the principal's clauses ("5.1 shall be substituted...");
            # those quotes belong to the amending paragraph, not new clauses.
            return major == last_major
        return last_major <= major <= last_major + 3
    return last_major < major <= last_major + 2


def split_clauses(lines: list[tuple[int, str]], heading_style: str = "auto",
                  strict_subclauses: bool = False) -> list[dict]:
    """Group lines into clause blocks: {clause, section_title, page, lines}."""
    blocks: list[dict] = [{"clause": "", "section_title": "", "page": lines[0][0] if lines else 1, "lines": []}]
    last_major: int | None = None
    section_titles: dict[str, str] = {}
    dotted_style = heading_style == "dotted" or (heading_style == "auto" and _uses_dotted_top_level(lines))
    for pno, ln in lines:
        m = HEADING_RE.match(ln)
        if m and _is_heading(m["num"], m["rest"], last_major, ln, dotted_style, strict_subclauses):
            num, rest = m["num"], m["rest"]
            major = num.split(".")[0]
            last_major = int(major)
            if "." not in num:
                # Short top-level line is a section title ("6 Eligible Consumer and individual project capacity")
                section_titles[major] = re.split(r"[.:–-]\s", rest)[0][:90]
            blocks.append({"clause": num, "section_title": section_titles.get(major, ""), "page": pno, "lines": [ln]})
        else:
            blocks[-1]["lines"].append(ln)
    return [b for b in blocks if b["lines"]]


def _pack(blocks: list[dict]) -> list[dict]:
    """Merge tiny blocks into their predecessor and split oversized ones on line boundaries."""
    merged: list[dict] = []
    for b in blocks:
        text = "\n".join(b["lines"])
        if merged and len(text) < config.MIN_CHUNK_CHARS and len(merged[-1]["text"]) + len(text) <= config.MAX_CHUNK_CHARS:
            merged[-1]["text"] += "\n" + text
            continue
        merged.append({**b, "text": text})

    out: list[dict] = []
    for b in merged:
        if len(b["text"]) <= config.MAX_CHUNK_CHARS:
            out.append(b)
            continue
        buf: list[str] = []
        size = 0
        for ln in b["text"].split("\n"):
            if size + len(ln) > config.MAX_CHUNK_CHARS and buf:
                out.append({**b, "text": "\n".join(buf)})
                # one line of overlap keeps sentences that straddle the cut readable
                buf, size = buf[-1:], len(buf[-1])
            buf.append(ln)
            size += len(ln) + 1
        if buf:
            out.append({**b, "text": "\n".join(buf)})
    return out


def amended_refs(text: str) -> list[str]:
    """Clause numbers this text amends: references with an amending verb within AMEND_WINDOW chars."""
    refs = set()
    for m in REF_RE.finditer(text):
        window = text[max(0, m.start() - AMEND_WINDOW): m.end() + AMEND_WINDOW]
        if AMEND_VERB_RE.search(window):
            refs.add(m.group(1))
    return sorted(refs)


def chunk_document(meta: dict) -> list[Chunk]:
    lines = extract_lines(config.RAW_DIR / meta["file"])
    if not lines:
        return []
    blocks = split_clauses(
        lines,
        heading_style=meta.get("heading_style", "auto"),
        strict_subclauses=meta["doc_type"] in AMENDING_TYPES,
    )
    chunks = []
    for i, b in enumerate(_pack(blocks)):
        refs = amended_refs(b["text"]) if meta.get("amends") else []
        chunks.append(Chunk(
            chunk_id=f"{meta['doc_id']}::{b['clause'] or 'pre'}::{i}",
            doc_id=meta["doc_id"],
            text=b["text"],
            clause=b["clause"],
            section_title=b["section_title"],
            page=b["page"],
            state=meta["state"],
            authority=meta["authority"],
            doc_type=meta["doc_type"],
            title=meta["title"],
            date=meta["date"],
            status=meta["status"],
            amends=meta.get("amends") or "",
            source_url=meta["source_url"],
            refs=refs,
        ))
    return chunks


def run() -> list[Chunk]:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[Chunk] = []
    for meta in load_manifest():
        if meta.get("skip"):
            print(f"SKIP {meta['file']}: {meta.get('skip_reason', '')}")
            continue
        chunks = chunk_document(meta)
        if not chunks:
            print(f"WARN {meta['file']}: no extractable text")
            continue
        n_clauses = len({c.clause for c in chunks if c.clause})
        print(f"{meta['file']:<48} {len(chunks):>4} chunks, {n_clauses:>3} distinct clauses")
        all_chunks.extend(chunks)
    with config.CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_chunks)} chunks -> {config.CHUNKS_PATH}")
    return all_chunks


def load_chunks() -> list[Chunk]:
    with config.CHUNKS_PATH.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


if __name__ == "__main__":
    run()
