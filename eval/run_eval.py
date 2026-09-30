"""Evaluation in resumable stages (local CPU inference is slow, so each stage saves its output).

    python eval/run_eval.py retrieval                 # no LLM: hit@k, MRR, cross-state contamination
    python eval/run_eval.py answers [--agent]         # generate answers -> eval/results/answers_<mode>.jsonl
    python eval/run_eval.py ragas [--agent] [--limit N]   # RAGAS faithfulness / context precision / answer relevancy
    python eval/run_eval.py report [--agent]          # write eval/results/report_<mode>.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from regrag import config  # noqa: E402

TESTSET = ROOT / "eval" / "testset.json"
RESULTS = ROOT / "eval" / "results"
KEY_FACT_RE = re.compile(
    r"(?:Rs\.?|INR)\s?\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?\s?(?:%|kW|MW|kVA|paise|days?|years?)(?![a-z])", re.I)


def load_testset() -> list[dict]:
    return json.loads(TESTSET.read_text(encoding="utf-8"))


def norm_fact(s: str) -> str:
    s = re.sub(r"\s|,", "", s.lower()).replace("inr", "rs").replace("rs.", "rs")
    s = re.sub(r"^0+(?=\d)", "", s)          # "01mw" -> "1mw"
    return re.sub(r"days?$", "day", re.sub(r"years?$", "year", s))


NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "five": 5, "six": 6, "ten": 10, "fifteen": 15, "twenty": 20,
                "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "hundred": 100,
                "five hundred": 500}
UNIT_WORDS_MAP = [(r"per\s?cent", "%"), (r"mega-?\s?watts?", "MW"), (r"kilo-?\s?watts?", "kW")]


def spell_out_numbers(text: str) -> str:
    """'thirty percent' -> '30 %', 'one megawatt' -> '1 MW' so worded answers match numeric references."""
    for pat, unit in UNIT_WORDS_MAP:
        text = re.sub(pat, unit, text, flags=re.I)
    words = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    return re.sub(rf"\b({words})\s+(%|MW|kW|days?)",
                  lambda m: f"{NUMBER_WORDS[m.group(1).lower()]} {m.group(2)}", text, flags=re.I)


def key_facts(text: str) -> set[str]:
    return {norm_fact(m) for m in KEY_FACT_RE.findall(spell_out_numbers(text))}


# --- stage 1: retrieval (no LLM) ------------------------------------------------
def stage_retrieval(k: int) -> dict:
    from regrag.pipeline import components, retrieve_for
    from regrag.query import decompose, out_of_scope_message

    components()
    rows = []
    for item in load_testset():
        if out_of_scope_message(item["question"]):
            rows.append({"id": item["id"], "type": item["type"], "out_of_scope": True})
            continue
        hits, top_k = [], []
        for sq in decompose(item["question"]):
            sub_hits = retrieve_for(sq, item["question"])   # top-n reranked + pulled-in amendments
            hits.extend(sub_hits)
            top_k.extend(sub_hits[:k])
        docs = [h.chunk.doc_id for h in top_k]
        allowed = set(item["states"]) | {config.CENTRAL}
        contaminated = sum(h.chunk.state not in allowed for h in hits) if item["states"] else 0
        first = next((i + 1 for i, d in enumerate(docs) if d in item["expected_docs"]), None)
        scored_q = bool(item["expected_docs"])
        rows.append({
            "id": item["id"], "type": item["type"],
            "hit": None if not scored_q else first is not None,
            "context_hit": None if not scored_q else any(h.chunk.doc_id in item["expected_docs"] for h in hits),
            "rr": None if not scored_q else (1 / first if first else 0.0),
            "contaminated": contaminated, "n_hits": len(hits),
            "superseded_flags": sum(any(f.startswith("POSSIBLY SUPERSEDED") for f in h.flags) for h in hits),
        })
    scored = [r for r in rows if r.get("hit") is not None]
    summary = {
        f"hit@{k}": round(sum(r["hit"] for r in scored) / len(scored), 3),
        "hit_in_llm_context": round(sum(r["context_hit"] for r in scored) / len(scored), 3),
        "MRR": round(sum(r["rr"] for r in scored) / len(scored), 3),
        "contamination_rate": round(sum(r.get("contaminated", 0) for r in rows) /
                                    max(1, sum(r.get("n_hits", 0) for r in rows)), 3),
        "questions_scored": len(scored),
    }
    save(f"retrieval.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=2))
    for r in rows:
        if r.get("hit") is False:
            print("  MISS", r["id"])
    return summary


# --- stage 2: answers -------------------------------------------------------------
def stage_answers(agent: bool, limit: int | None) -> None:
    path = RESULTS / f"answers_{mode(agent)}.jsonl"
    done = {json.loads(l)["id"] for l in path.open(encoding="utf-8")} if path.exists() else set()
    items = [i for i in load_testset() if i["id"] not in done][:limit]
    print(f"{len(done)} done, {len(items)} to go -> {path}")
    if agent:
        from regrag.graph import run as run_agent
    else:
        from regrag.pipeline import answer
    RESULTS.mkdir(parents=True, exist_ok=True)
    for n, item in enumerate(items, 1):
        t = time.time()
        if agent:
            res, trace = run_agent(item["question"])
        else:
            res, trace = answer(item["question"]), []
        row = {"id": item["id"], **res.to_dict(), "trace": trace, "seconds": round(time.time() - t, 1)}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{n}/{len(items)}] {item['id']} {row['seconds']}s abstained={res.abstained}")


def load_answers(agent: bool) -> dict[str, dict]:
    path = RESULTS / f"answers_{mode(agent)}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.open(encoding="utf-8"))}


def answer_metrics(agent: bool) -> dict:
    from regrag.generate import cited_ids
    answers = load_answers(agent)
    tests = {t["id"]: t for t in load_testset()}
    abst_ok, fact_recall, cited, rows = [], [], [], []
    for tid, a in answers.items():
        t = tests[tid]
        should_abstain = t["type"] == "abstain"
        abst_ok.append(a["abstained"] == should_abstain)
        if not should_abstain:
            facts = key_facts(t["ground_truth"])
            got = key_facts(a["answer"])
            if facts:
                fact_recall.append(len(facts & got) / len(facts))
            if not a["abstained"]:
                # re-parse the saved text so grouped citations like [S2, S3] count
                cited.append(bool(a["citations"]) or bool(cited_ids(a["answer"])))
        rows.append({"id": tid, "abstain_correct": a["abstained"] == should_abstain,
                     "facts_expected": sorted(key_facts(t["ground_truth"])),
                     "facts_found": sorted(key_facts(t["ground_truth"]) & key_facts(a["answer"])),
                     "seconds": a["seconds"]})
    return {
        "answered": len(answers),
        "abstain_accuracy": round(sum(abst_ok) / len(abst_ok), 3),
        "key_fact_recall": round(sum(fact_recall) / max(1, len(fact_recall)), 3),
        "citation_rate": round(sum(cited) / max(1, len(cited)), 3),
        # median: a laptop sleeping mid-run inflates the mean
        "median_seconds": round(statistics.median(a["seconds"] for a in answers.values()), 1),
        "rows": rows,
    }


# --- stage 3: RAGAS ---------------------------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def judge_chat():
    """LangChain chat model used as the RAGAS judge, chosen by RAGAS_JUDGE in .env.

    ollama (default): the local generator model - too weak to follow RAGAS's JSON prompts reliably.
    groq:             hosted model via Groq's OpenAI-compatible API (needs GROQ_API_KEY).
    """
    import os
    provider = os.getenv("RAGAS_JUDGE", "ollama").lower()
    if provider == "groq":
        from langchain_openai import ChatOpenAI
        key = os.getenv("GROQ_API_KEY")
        if not key:
            sys.exit("RAGAS_JUDGE=groq but GROQ_API_KEY is not set - add it to .env (see .env.example).")
        model = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
        return ChatOpenAI(model=model, base_url=GROQ_BASE_URL, api_key=key, temperature=0,
                          max_retries=6, timeout=120), f"groq:{model}"
    from langchain_ollama import ChatOllama
    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL,
                      temperature=0, num_ctx=8192, format="json"), f"ollama:{config.LLM_MODEL}"


def stage_judge_check() -> None:
    """List the judge provider's models and make one tiny call, before spending time on a full run."""
    import os
    if os.getenv("RAGAS_JUDGE", "ollama").lower() == "groq" and os.getenv("GROQ_API_KEY"):
        import urllib.request
        req = urllib.request.Request(f"{GROQ_BASE_URL}/models",
                                     headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                                              "User-Agent": "regrag-eval"})
        with urllib.request.urlopen(req, timeout=30) as r:
            ids = sorted(m["id"] for m in json.load(r)["data"])
        print("Groq models available to this key:\n  " + "\n  ".join(ids))
    chat, name = judge_chat()
    print(f"\nTest call to {name}:", chat.invoke('Reply with the JSON {"ok": true} and nothing else.').content)


RAGAS_METRICS = {  # CLI name -> result column
    "faithfulness": "faithfulness",
    "context_precision": "llm_context_precision_with_reference",
    "answer_relevancy": "answer_relevancy",
}


def stage_ragas(agent: bool, limit: int | None, ids: list[str] | None = None,
                only: list[str] | None = None) -> dict:
    sys.path.insert(0, str(ROOT / "eval"))
    import ragas_compat  # noqa: F401  (must precede ragas import)
    from langchain_ollama import OllamaEmbeddings
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, ResponseRelevancy

    tests = {t["id"]: t for t in load_testset()}
    answers = [a for a in load_answers(agent).values()
               if not a["abstained"] and tests[a["id"]]["type"] != "abstain" and a["contexts"]
               and (not ids or a["id"] in ids)][:limit]
    samples = [SingleTurnSample(user_input=a["question"], response=a["answer"],
                                retrieved_contexts=a["contexts"], reference=tests[a["id"]]["ground_truth"])
               for a in answers]
    chat, judge_name = judge_chat()
    judge = LangchainLLMWrapper(chat)
    only = only or list(RAGAS_METRICS)
    factories = {"faithfulness": Faithfulness, "context_precision": LLMContextPrecisionWithReference,
                 "answer_relevancy": lambda: ResponseRelevancy(strictness=1)}
    metrics = [factories[m]() for m in only]
    emb = (LangchainEmbeddingsWrapper(OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL))
           if "answer_relevancy" in only else None)   # only relevancy needs embeddings (local Ollama)
    print(f"RAGAS {only} on {len(samples)} answered questions, judge={judge_name}")
    # one worker keeps free-tier rate limits happy; retries back off on 429s
    result = evaluate(EvaluationDataset(samples), metrics=metrics, llm=judge, embeddings=emb,
                      run_config=RunConfig(timeout=900, max_workers=1, max_retries=10, max_wait=120))
    df = result.to_pandas()
    df.insert(0, "id", [a["id"] for a in answers])
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv = RESULTS / f"ragas_{mode(agent)}.csv"
    if (ids or len(only) < len(RAGAS_METRICS)) and csv.exists():
        # partial re-run: overwrite only the re-scored (id, metric) cells, keep every other score
        import pandas as pd
        old = pd.read_csv(csv).set_index("id")
        new = df.set_index("id")
        for col in (RAGAS_METRICS[m] for m in only):
            if col in new:
                old.loc[new.index, col] = new[col]
        df = old.reset_index()
        order = {t: i for i, t in enumerate(tests)}
        df = df.sort_values("id", key=lambda s: s.map(order)).reset_index(drop=True)
    df.to_csv(csv, index=False)
    cols = [c for c in ("faithfulness", "llm_context_precision_with_reference", "answer_relevancy") if c in df]
    summary = {c: round(float(df[c].mean(skipna=True)), 3) for c in cols}
    summary["n"] = len(df)
    summary["n_scored"] = {c: int(df[c].notna().sum()) for c in cols}   # NaN = the judge's output failed to parse
    summary["judge"] = judge_name
    save(f"ragas_{mode(agent)}.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


# --- report ------------------------------------------------------------------------
def stage_report(agent: bool) -> None:
    m = mode(agent)
    retr = read(f"retrieval.json", {}).get("summary", {})
    ans = answer_metrics(agent) if (RESULTS / f"answers_{m}.jsonl").exists() else {}
    rag = read(f"ragas_{m}.json", {})
    lines = [f"# Evaluation report ({m})", "",
             f"Model: `{config.LLM_MODEL}` · embeddings: `{config.EMBED_MODEL}` · reranker: `{config.RERANK_MODEL}`", "",
             "## Retrieval (no LLM)", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in retr.items()]
    lines += ["", "## Answers", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in ans.items() if k != "rows"]
    lines += ["", "## RAGAS", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rag.items()] or ["| (not run) | |"]
    if ans:
        lines += ["", "## Per question", "", "| id | abstain ok | key facts found / expected | s |", "|---|---|---|---|"]
        for r in ans["rows"]:
            lines.append(f"| {r['id']} | {'✅' if r['abstain_correct'] else '❌'} | "
                         f"{len(r['facts_found'])}/{len(r['facts_expected'])} {r['facts_found']} | {r['seconds']} |")
    out = RESULTS / f"report_{m}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def mode(agent: bool) -> str:
    return "agent" if agent else "pipeline"


def save(name: str, obj) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read(name: str, default):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["retrieval", "answers", "ragas", "report", "judge-check"])
    ap.add_argument("--agent", action="store_true", help="use the LangGraph agent instead of the linear pipeline")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="+", default=None, help="restrict RAGAS to these test ids")
    ap.add_argument("--metrics", nargs="+", choices=list(RAGAS_METRICS), default=None,
                    help="RAGAS metrics to (re)compute; others keep their saved scores")
    ap.add_argument("-k", type=int, default=config.RERANK_TOP_N)
    a = ap.parse_args()
    {"retrieval": lambda: stage_retrieval(a.k), "answers": lambda: stage_answers(a.agent, a.limit),
     "ragas": lambda: stage_ragas(a.agent, a.limit, a.ids, a.metrics), "report": lambda: stage_report(a.agent),
     "judge-check": stage_judge_check}[a.stage]()
