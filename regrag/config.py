"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "re_regulations"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Chunking
MIN_CHUNK_CHARS = 250
MAX_CHUNK_CHARS = 1800

# Retrieval
DENSE_K = 25          # candidates from the vector store
SPARSE_K = 25         # candidates from BM25
RRF_K = 60            # reciprocal-rank-fusion constant
RERANK_TOP_N = 5      # chunks passed to the LLM per state (small local models degrade with more)
ABSTAIN_SCORE = float(os.getenv("ABSTAIN_SCORE", "-6.0"))  # best cross-encoder logit below this => abstain

STATES = ["Gujarat", "Maharashtra", "Karnataka", "Tamil Nadu", "Rajasthan"]
CENTRAL = "Central"
