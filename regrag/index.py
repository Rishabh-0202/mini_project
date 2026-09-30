"""Build / load the dense (Chroma + Ollama embeddings) and sparse (BM25) indexes.

Run:  python -m regrag.index          (re-ingests if chunks.jsonl is missing)
"""
from __future__ import annotations

import re
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

from . import config
from .ingest import Chunk, load_chunks
from .ingest import run as run_ingest

STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or shall
that the their these this to under was which with within any such other""".split())
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)


def chunk_metadata(c: Chunk) -> dict:
    """Chroma metadata must be scalar; lists are joined."""
    return {
        "chunk_id": c.chunk_id, "doc_id": c.doc_id, "clause": c.clause, "section_title": c.section_title,
        "page": c.page, "state": c.state, "authority": c.authority, "doc_type": c.doc_type,
        "title": c.title, "date": c.date, "status": c.status, "amends": c.amends,
        "refs": ",".join(c.refs),
    }


def build_vectorstore(chunks: list[Chunk]) -> Chroma:
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    store = Chroma(collection_name=config.COLLECTION, embedding_function=embeddings(),
                   persist_directory=str(config.CHROMA_DIR),
                   collection_metadata={"hnsw:space": "cosine"})
    docs = [Document(page_content=c.embed_text, metadata=chunk_metadata(c)) for c in chunks]
    batch = 64
    for i in range(0, len(docs), batch):
        store.add_documents(docs[i:i + batch], ids=[d.metadata["chunk_id"] for d in docs[i:i + batch]])
        print(f"  embedded {min(i + batch, len(docs))}/{len(docs)}")
    return store


def load_vectorstore() -> Chroma:
    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError("Vector store missing - run `python -m regrag.index` first.")
    return Chroma(collection_name=config.COLLECTION, embedding_function=embeddings(),
                  persist_directory=str(config.CHROMA_DIR))


class BM25Index:
    """BM25 over the chunk's contextual text; filtering by state happens on the scores."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])

    def search(self, query: str, states: list[str] | None, k: int) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            c = self.chunks[i]
            if states and c.state not in states:
                continue
            if scores[i] <= 0:
                break
            out.append((c, float(scores[i])))
            if len(out) == k:
                break
        return out


def run() -> None:
    if not config.CHUNKS_PATH.exists():
        run_ingest()
    chunks = load_chunks()
    print(f"Embedding {len(chunks)} chunks with {config.EMBED_MODEL} ...")
    build_vectorstore(chunks)
    print(f"Chroma index written to {config.CHROMA_DIR}")


if __name__ == "__main__":
    run()
