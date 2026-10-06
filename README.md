# RegRAG: Q&A over Indian state renewable-energy regulations

Assignment 3 mini-project: Rishabh Tripathi, Jagath Ponnanna PM, Yashraj Singh Srinet.

Electricity regulation in India is a state subject. Net metering limits, banking, wheeling and open
access rules are set by each SERC and revised through frequent amendments. RegRAG answers questions
such as *"What is the net metering limit in Rajasthan?"* using only the regulation text. Every answer
cites the **clause, order and date** it relies on, flags clauses that a later amendment may have
**superseded**, and **abstains** when the indexed documents don't cover the question.

**Models.** Answers are written by **Groq `openai/gpt-oss-120b`** by default, which needs `GROQ_API_KEY`
in `.env`. Embeddings (`nomic-embed-text` via Ollama) and reranking (a MiniLM cross-encoder) always run
locally. To run fully offline, set `LLM_PROVIDER=ollama` in `.env`; answers then come from local
`llama3.2`, which is less accurate (see *Results*).

## Architecture

```
PDFs ─► ingest.py ──► clause-level chunks + metadata (state, authority, doc type, date, status, amended refs)
            │
            ├─► index.py ─► ChromaDB (nomic-embed-text)       BM25 (rank-bm25)
            │                        │                             │
question ─► query.py: detect state(s), split comparisons ─► retrieve.py: state-filtered dense + BM25
                                                               ─► RRF fusion ─► cross-encoder rerank
                                                                        │
                                          amendments.py: find later docs amending each retrieved clause,
                                          flag "POSSIBLY SUPERSEDED", pull amendments in, temporal ranking
                                                                        │
                                          generate.py: answer only from sources, cite [S#] + clause/date,
                                          current vs older/draft sources separated, abstain if unsupported
```

`graph.py` wraps the same components in a **LangGraph** agent:

```
classify ─► retrieve (state-routed) ─► check_amendments ─► generate ─► verify (LLM groundedness judge)
               ▲                                                          │
               └────────────── reformulate ◄── ungrounded & retries left ─┤
                                                                          ▼
                                                                       format
```

| Proposal item | Where it lives |
|---|---|
| Clause/section chunking + metadata | `regrag/ingest.py`, `data/manifest.json` |
| Vector store | `regrag/index.py` (ChromaDB, cosine) |
| State detection, multi-state decomposition, out-of-scope abstain | `regrag/query.py` |
| Hybrid BM25 + dense, metadata filter, cross-encoder rerank | `regrag/retrieve.py` |
| Amendment check + temporal ranking | `regrag/amendments.py` |
| Cited, abstaining generation | `regrag/generate.py` |
| LangGraph agentic workflow | `regrag/graph.py` |
| RAGAS + retrieval/answer metrics on hand-written Q&A | `eval/run_eval.py`, `eval/testset.json` |
| Streamlit demo | `app.py` |

## Corpus (13 PDFs, 12 indexed)

| State | Documents |
|---|---|
| Gujarat | GERC Net Metering Regulations 2016 · consolidated with 1st Amendment 2017 · 4th Amendment 2024 |
| Maharashtra | MERC Grid Interactive Rooftop RE Regulations 2019 · Statements of Reasons for 1st (2023) and 2nd (2024) Amendments |
| Karnataka | KERC solar / SRTPV generic tariff order for FY24 (01.06.2023) |
| Tamil Nadu | TANGEDCO summary of TNERC GISS Order 8/2021 · **draft** TNERC GISS Regulations 2024 |
| Rajasthan | RERC DREGS Regulations 2021 · RERC suo-motu order 07.02.2024 (net metering limit raised to 1 MW) |
| Central | MoP Green Energy Open Access Rules 2022 |

Sources and dates are listed in `data/manifest.json`. **Known gaps:**

- The newest 2025 amendments (GERC 5th, RERC 3rd, KERC DSPV) had no downloadable PDF.
- One KERC order is a scanned image and is skipped; including it would need OCR.
- For Maharashtra, the amendments are represented by their Statements of Reasons rather than the gazette text.
- Several PDFs are copies hosted by third parties. Their URLs are recorded in the manifest.

## Setup

```bash
uv venv .venv --python 3.12            # Python 3.12 recommended
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
ollama pull nomic-embed-text           # embeddings (always local)
ollama pull llama3.2                   # only needed for LLM_PROVIDER=ollama
copy .env.example .env                 # then add your GROQ_API_KEY
```

```bash
python scripts/download_docs.py        # fetch PDFs from manifest URLs into data/raw/
python -m regrag.ingest                # -> data/processed/chunks.jsonl
python -m regrag.index                 # -> data/chroma/  (~15 min on CPU)
```

## Usage

```bash
python -m regrag.pipeline "What is the current net metering capacity limit in Rajasthan?"
python -m regrag.graph "Compare the transformer capacity limits in Gujarat and Maharashtra"
streamlit run app.py
pytest tests
python scripts/make_figures.py         # presentation figures -> docs/figures/ (PNG + SVG)
```

## Project notebook: `RegRag_updated.ipynb`

**This is the notebook submitted as the project.** It rebuilds the whole system as about 30 short,
commented functions in one place, and does **not** import the `regrag` package, so every step can be
read top to bottom. All outputs are saved, so the results can be read without running anything.

### Sections

| Section | What it does | Main functions |
|---|---|---|
| 1. Settings | Folders, models, chunk and search sizes | (variables) |
| 2. Read & split PDFs | PDF → clean lines → clause-sized chunks with metadata. A numbered line only starts a clause if the numbering moves forward in small steps, which filters out table rows and quantities like "1 MW" | `read_pdf_lines`, `is_new_clause`, `split_into_clauses`, `amended_clauses`, `make_chunks` |
| 3. Build indexes | ChromaDB vectors (`nomic-embed-text`, saved in `data/chroma_notebook/`) + BM25 keyword index | `search_text`, `words` |
| 4. Understand the question | Detects the state from names, regulators, power companies and cities; declines uncovered states; splits comparisons into one question per state | `find_states`, `out_of_scope`, `question_for_state` |
| 5. Search | BM25 + vector search inside the state filter, merged with reciprocal-rank fusion, reranked by a cross-encoder → top 5 | `vector_search`, `keyword_search`, `hybrid_search`, `rerank`, `retrieve` |
| 6. Check amendments | Flags clauses a later final order changed, pulls in the newer version, ranks current law first | `changes_clause`, `newer_versions`, `check_amendments` |
| 7. Write the answer | Groq `gpt-oss-120b` answers only from numbered sources, cites `[S#]`, declines if unsupported | `format_sources`, `write_answer` |
| 8. Full pipeline | Steps 4–7 in one call; comparisons answered per state and joined | `make_plan`, `search_scope`, `join_answers`, `answer` |
| 9. Agent (LangGraph) | Same steps as a graph, plus a self-check: verify → rewrite → retry once, keeping the first answer if the retry is worse | `classify`, `retrieve_node`, `amendments_node`, `write_node`, `verify`, `rewrite` |
| 10. Evaluation | Runs all 34 test questions live and scores retrieval and answers; shows the saved RAGAS scores | `run_question`, `key_numbers`, `score` |

### Running it

1. Start Ollama (`ollama serve`) with `nomic-embed-text` pulled. It's used for the embeddings.
2. Put `GROQ_API_KEY` in `.env`. If it's missing, the notebook asks for it.
3. Open the notebook in VS Code, choose the kernel **Python (RegRAG .venv)** and click **Run All**.

The first run builds the vector index (about 10 minutes on a CPU), which is then saved and reused.
Section 10 asks 34 questions, taking about 9 minutes because of a pause between Groq calls for the
free-tier rate limit. The other LLM cells take a few seconds each.

### Evaluation results (section 10, live with Groq `gpt-oss-120b`)

| Measure | Score | Meaning |
|---|---|---|
| Retrieval hit | **100%** | A clause from the expected document reached the model for every answerable question |
| Cross-state contamination | **0%** | No clause from a state not in the question ever reached the model |
| Decline accuracy | **97.1%** (33/34) | Answered the answerable questions and declined the others (Kerala; a Karnataka FY2026 charge) |
| Citation rate | **100%** | Every answer cites its sources |
| Key-number recall (automatic) | 64% | Share of the reference answer's figures (1 MW, 65%, Rs 4.50…) that appear in the answer |
| **Correct answers, judged by hand** | **31 / 34 (91%)** | Including both two-state comparisons, all amendment questions and both decline questions |

**Why key-number recall understates quality.** The automatic check counts a miss whenever any number
from the reference answer is absent, even when the reference lists extra details the question didn't
ask for. Of the 11 answers it flags, **8 are correct**:
- GJ06 gives the right fee (Rs 10,000) without the other fee slabs.
- RJ01 gives 1 MW without the old 500 kW limit.
- MH01 omits the 1 kW minimum.
- TN01 omits the HT capacity range.
- KA02, GJ08, GJ09 and MS02 omit side details.

**3 are real errors:**
- **MH05** declined, although the Maharashtra registration fee is in Regulation 9.1: retrieval ranked the fee table too low.
- **MH03** cites the right clause (11.4(c)) but doesn't say what it says.
- **MH04** reaches the right conclusion (small consumers are exempt) for the wrong reason: it quotes a 5,000 MW rule about wheeling charges instead of the 10 kW exemption.

**RAGAS** (shown in section 10 from the saved run): faithfulness **0.64**, context precision **0.79**,
answer relevancy **0.80**. These were measured when answers came from the local `llama3.2` (3B), before
answers switched to Groq. Re-running RAGAS for the Groq answers needs more tokens than Groq's free daily
quota allows in one go, so it would have to be spread over two days with `eval/run_eval.py ragas --ids …`.

*LLM wording can vary between runs, so the exact list of flagged answers may differ slightly on a re-run.*

## Other notebooks (`notebooks/`)

| Notebook | Where | What it does |
|---|---|---|
| `notebooks/RegRAG_run_all.ipynb` | Local | The shortest tour: runs each `regrag` file once (config → ingest → index → query → retrieve → amendments → generate → pipeline → graph → tests) |
| `notebooks/RegRAG_walkthrough.ipynb` | Local (VS Code, kernel "Python (RegRAG .venv)") | Runs the `regrag` package one step at a time, showing chunks, BM25 vs dense vs fused rankings, reranker scores, amendment flags, the prompt, answers, the agent trace and the saved evaluation results |
| `notebooks/RegRAG - RAG With LangGraph.ipynb` | Local | The project in the class-lab format (`init_chat_model` + Groq, FAISS, LangGraph `StateGraph`) |
| `notebooks/RegRAG_colab.ipynb` | Google Colab (T4 GPU recommended) | Installs everything inside Colab (packages, Ollama, embedding model, PDFs, index), asks for the Groq key, then runs the walkthrough |

For Colab, build the upload bundle first. It excludes `.env`, `.venv`, PDFs and indexes:

```bash
python scripts/make_colab_bundle.py    # -> dist/regrag_colab.zip
```

Then open `RegRAG_colab.ipynb` in Colab (**File → Upload notebook**), switch to a T4 GPU runtime,
**Run all**, and upload the zip when prompted.

## Evaluation

The test set in `eval/testset.json` has **34 hand-written questions**. Each ground-truth answer was
checked against the clause text in the PDFs. The set covers:

- single-state facts
- amendment and supersession cases (Gujarat 6.2 in 2016 vs 2017, Rajasthan 500 kW → 1 MW, Gujarat 2024 amendment)
- draft-regulation questions (Tamil Nadu)
- central rules
- multi-state comparisons
- questions that should be declined (a Karnataka FY2026 charge; Kerala)

```bash
python eval/run_eval.py retrieval      # no LLM, ~2 min
python eval/run_eval.py answers        # ~1 h on CPU, resumable
python eval/run_eval.py judge-check    # verify the RAGAS judge (GROQ_API_KEY in .env) and list models
python eval/run_eval.py ragas          # RAGAS faithfulness / context precision / answer relevancy
python eval/run_eval.py report
```

The project notebook's section 10 runs this same test set live with Groq; its results are in
[*Project notebook*](#project-notebook-regrag_updatedipynb) above. The tables below are from the
`regrag` package's full evaluation, run earlier with the local model.

### Results (linear pipeline, `llama3.2` 3B on a CPU-only laptop)

| Retrieval (no LLM) | Value |
|---|---|
| hit@5 (expected source among the top 5 reranked chunks) | **0.969** |
| Expected source anywhere in the LLM context (top 5 + pulled-in amendments) | **1.000** |
| MRR | **0.914** |
| Cross-state contamination (chunks from a state not asked about) | **0.000** |

| Answers (34 questions) | Value |
|---|---|
| Abstain accuracy (declines the 2 out-of-scope questions, answers the 32 in-scope ones) | **1.000** |
| Citation rate (answered questions with ≥1 valid `[S#]`) | **0.938** |
| Key-fact recall (numbers/limits from the reference found in the answer) | **0.639** |
| Median latency per answer | **61 s** |

Full per-question results are in `eval/results/report_pipeline.md`.

**Reading the numbers.** Retrieval is close to perfect: the right clause, including later
amendments, reaches the model for every question, and no answer drew on another state's rules. The
errors come from the 3B generator. It picked the wrong row from the Gujarat fee table (Rs 50,000
instead of Rs 10,000), left out the second of two tariffs (Karnataka "with capital subsidy"), and
confused the transformer limit with the 6 kW threshold in one comparison answer. Key-fact recall is
the metric this affects most.

**RAGAS.** With the local `llama3.2` as the judge, every RAGAS metric came back `NaN`. The 3B model
copies RAGAS's instructions back as a JSON *string* instead of producing the required JSON object,
and Ollama's JSON mode accepts a bare string. RAGAS therefore uses a hosted judge (Groq, set with
`RAGAS_JUDGE=groq` in `.env`). Answers are still generated locally, and `ResponseRelevancy` still
uses the local `nomic-embed-text` embeddings. `eval/ragas_compat.py` separately works around a
removed Vertex AI import in newer langchain-community.

| RAGAS (judge: Groq `openai/gpt-oss-120b`) | Score | Questions scored |
|---|---|---|
| Faithfulness | **0.640** | 32 / 32 |
| Context precision (with reference) | **0.789** | 31 / 32 |
| Answer relevancy | **0.803** | 32 / 32 |

Per-question scores are in `eval/results/ragas_pipeline.csv`. For MS02's context precision, the
judge's output failed to parse on both attempts. MS02 is a two-state comparison, so its context is
about twice as long as other questions'. The Groq free tier allows 200,000 tokens per day for this
model, so the 32 questions were scored over two days. `--ids` and `--metrics` re-score chosen
questions and metrics while keeping the other saved scores:

```bash
python eval/run_eval.py ragas --metrics context_precision --ids MS02
python eval/run_eval.py report
```

**What the RAGAS scores say.**
- **Context precision (0.79)** is consistent with the retrieval metrics: relevant clauses are ranked
  high.
- **Faithfulness (0.64)** is the weak spot, as the manual review suggested. The 3B generator
  sometimes states facts its sources don't support. GJ06 (wrong fee slab), MH04 and MS01 (a
  comparison answer that confused the transformer limit with the 6 kW threshold) score 0.
- **Answer relevancy (0.80)** is lowered by very short answers, e.g. "1. One Mega-watt (01 MW) [S4]"
  (RJ01: 0.48).
- **By state**, faithfulness is higher for Rajasthan, Karnataka, Tamil Nadu and central rules
  (0.76 over 14 questions) than for Gujarat and Maharashtra (0.58 over 16), whose documents have
  more amendment layers and tables.
- **Comparison questions** (MS01, MS02) are the weakest overall.

**Agentic mode.** Tested end to end on *"maximum net metering capacity in Gujarat versus
Rajasthan"*. The trace showed routing to two states, amendment flags on Gujarat, a verifier
rejection for Rajasthan, and a query rewrite. Retrieval on the rewritten query scored lower (5.24 vs
6.60), so the agent rolled back to its first, correct answer ("1 MW", RERC order 07.02.2024) and
added a warning. It takes about 3 minutes per question on CPU.

## Limitations

- **The evaluation numbers above were measured with local `llama3.2` (3B).** The weak points (wrong
  table rows, missed sub-answers, faithfulness 0.64) came from that model, which is why the default
  answer model is now Groq `gpt-oss-120b`. In spot checks it answered questions correctly where
  `llama3.2` failed (the Gujarat 50% exemption, the Maharashtra 70% transformer limit). Re-run
  `eval/run_eval.py answers` and `ragas` to measure it formally.
- **Using Groq sends the question and the retrieved regulation text** (public documents) to Groq's
  API, and uses free-tier quota. `LLM_PROVIDER=ollama` keeps everything on the laptop.
- **Amendment detection is heuristic.** It links clause references that appear near amending verbs.
  A Statement of Reasons that says "no modification" to a clause can still raise a *possibly
  superseded* flag, which is why the flag says "possibly".
- **Chunking relies on text extraction.** Tables flatten to text and scanned PDFs are skipped.
- **The agent's verifier and rewriter use the same answer model.** With `llama3.2` they sometimes
  rejected correct answers. A retry is kept only if it retrieves better sources, so a bad rewrite
  cannot replace a good answer.
- **Chunk labels can mislead.** In the package's ingestion a short clause can be merged into the
  previous clause's chunk. For example, Gujarat's 65% transformer limit (clause 5) sits in a chunk
  labelled clause 3.3, so a comparison question can pick a different Gujarat rule. The from-scratch
  notebook's simpler splitter happens to keep that clause as its own chunk and answers 65% correctly;
  merging short clauses *forward* into the next clause would fix it in the package.
- **Not legal advice.** Always verify against the cited order.
