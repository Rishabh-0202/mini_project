"""Agentic workflow (LangGraph).

    classify -> retrieve (state-routed) -> check_amendments -> generate -> verify
                  ^                                                     |
                  +---------- reformulate  <---- ungrounded & retries left
                                                                         |
                                                                  format (done)

    python -m regrag.graph "Compare net metering limits in Gujarat and Rajasthan"
"""
from __future__ import annotations

import json
import sys
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from . import config
from .generate import Answer, format_sources, generate, llm
from .pipeline import Result, combine, components, state_scope
from .query import SubQuery, decompose, is_comparison, out_of_scope_message
from .retrieve import Hit

MAX_RETRIES = 1


class Slot(TypedDict, total=False):
    sub: SubQuery
    query: str               # current (possibly reformulated) retrieval query
    hits: list[Hit]
    answer: Answer | None
    grounded: bool | None
    note: str
    attempts: int
    previous: tuple        # (answer, hits, note) of the attempt before a reformulation


class GraphState(TypedDict):
    question: str
    comparison: bool
    slots: list[Slot]
    trace: list[str]
    result: Result | None


# --- nodes ------------------------------------------------------------------
def classify(state: GraphState) -> dict:
    if msg := out_of_scope_message(state["question"]):
        return {"slots": [], "result": Result(state["question"], msg, True),
                "trace": state["trace"] + ["classify: out of scope -> abstain"]}
    subs = decompose(state["question"])
    slots = [Slot(sub=s, query=s.text, hits=[], answer=None, grounded=None, note="", attempts=0) for s in subs]
    route = ", ".join(s.state or "ALL" for s in subs)
    return {"slots": slots, "comparison": is_comparison(state["question"]),
            "trace": state["trace"] + [f"classify: routed to [{route}]"]}


def retrieve(state: GraphState) -> dict:
    retriever, _ = components()
    trace = []
    for slot in state["slots"]:
        if slot["answer"] is None:
            slot["hits"] = retriever.retrieve(slot["query"], state_scope(slot["sub"], state["question"]))
            best = slot["hits"][0].rerank if slot["hits"] else float("nan")
            trace.append(f"retrieve[{slot['sub'].state or 'ALL'}]: {len(slot['hits'])} hits, best rerank {best:.2f}")
    return {"slots": state["slots"], "trace": state["trace"] + trace}


def check_amendments(state: GraphState) -> dict:
    _, checker = components()
    trace = []
    for slot in state["slots"]:
        if slot["answer"] is None:
            before = len(slot["hits"])
            slot["hits"] = checker.check(slot["query"], slot["hits"])
            n_flag = sum(any(f.startswith("POSSIBLY SUPERSEDED") for f in h.flags) for h in slot["hits"])
            trace.append(f"amendments[{slot['sub'].state or 'ALL'}]: {n_flag} superseded flags, "
                         f"{len(slot['hits']) - before} amendment chunks pulled in")
    return {"slots": state["slots"], "trace": state["trace"] + trace}


def generate_node(state: GraphState) -> dict:
    model = llm()
    multi = len(state["slots"]) > 1
    for slot in state["slots"]:
        if slot["answer"] is None:
            q = slot["sub"].text if multi else state["question"]
            slot["answer"] = generate(q, slot["hits"], model)
            slot["attempts"] += 1
    return {"slots": state["slots"], "trace": state["trace"] + ["generate: answers drafted"]}


VERIFY_PROMPT = """You check whether an ANSWER is fully supported by SOURCES.
Return JSON: {"grounded": true|false, "unsupported": "<the first unsupported claim, or empty>"}.
A claim is supported only if its facts and numbers appear in SOURCES."""


def _json_object(text: str) -> dict:
    """The first {...} object in a reply (hosted models may wrap JSON in prose or code fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def verify(state: GraphState) -> dict:
    judge = llm(num_predict=160)                       # a verdict is ~30 tokens
    if config.LLM_PROVIDER == "ollama":
        judge = judge.bind(format="json")              # Ollama's JSON mode keeps the small model on format
    trace = []
    for slot in state["slots"]:
        a = slot["answer"]
        if slot["grounded"] is not None:     # already verified in an earlier pass
            continue
        if slot.get("previous"):
            prev_answer, prev_hits, prev_note = slot.pop("previous")
            best = lambda hs: max((h.rerank for h in hs), default=float("-inf"))  # noqa: E731
            if a.abstained or best(slot["hits"]) < best(prev_hits):
                slot["answer"], slot["hits"], slot["note"] = prev_answer, prev_hits, prev_note
                slot["grounded"] = False
                trace.append(f"verify[{slot['sub'].state or 'ALL'}]: retry was worse, rolled back to first answer")
                continue
        if a is None or a.abstained:
            slot["grounded"] = True          # abstaining is always grounded
            continue
        msg = f"SOURCES:\n{format_sources(slot['hits'], 1200)}\n\nANSWER:\n{a.text}"
        try:
            verdict = _json_object(judge.invoke([SystemMessage(VERIFY_PROMPT), HumanMessage(msg)]).content)
            judged = bool(verdict.get("grounded", False))
            ok = judged and bool(a.citations)
            if ok:
                slot["note"] = ""
            elif not a.citations:
                slot["note"] = "answer has no citations"
            else:
                slot["note"] = verdict.get("unsupported") or "the verifier judged some claims unsupported"
        except (json.JSONDecodeError, AttributeError):
            ok, slot["note"] = bool(a.citations), "verifier returned malformed output"
        slot["grounded"] = ok
        trace.append(f"verify[{slot['sub'].state or 'ALL'}]: grounded={ok}" + ("" if ok else f" ({slot['note'][:80]})"))
    return {"slots": state["slots"], "trace": state["trace"] + trace}


REWRITE_PROMPT = """Rewrite the user's question in plain English so it matches the wording of Indian
electricity regulations (e.g. "net metering", "eligible consumer", "sanctioned load", "contract demand",
"banking", "wheeling charges", "cross subsidy surcharge", "rooftop solar PV system").
Keep the same state name and intent. Output only the rewritten English question, nothing else."""


def _valid_rewrite(new_q: str, slot: Slot) -> bool:
    """Reject rewrites a small model garbles: non-English, too long, or missing the state."""
    state_ok = slot["sub"].state is None or slot["sub"].state.lower() in new_q.lower()
    return bool(new_q) and new_q.isascii() and len(new_q) < 300 and state_ok


def reformulate(state: GraphState) -> dict:
    model = llm(temperature=0.0, num_predict=80)        # one rewritten question
    trace = []
    for slot in state["slots"]:
        if slot["grounded"] is False and slot["attempts"] <= MAX_RETRIES:
            new_q = model.invoke([SystemMessage(REWRITE_PROMPT), HumanMessage(slot["query"])]).content.strip()
            new_q = new_q.strip('"').splitlines()[0] if new_q else ""
            if not _valid_rewrite(new_q, slot):
                trace.append(f"reformulate[{slot['sub'].state or 'ALL'}]: rejected rewrite '{new_q[:80]}', keeping answer")
                slot["attempts"] = MAX_RETRIES + 1      # stop retrying this slot
                continue
            trace.append(f"reformulate[{slot['sub'].state or 'ALL'}]: '{slot['query']}' -> '{new_q}'")
            # keep the first attempt so a worse retry can be rolled back
            slot["previous"] = (slot["answer"], slot["hits"], slot["note"])
            slot["query"], slot["answer"], slot["grounded"] = new_q, None, None
    return {"slots": state["slots"], "trace": state["trace"] + trace}


def format_node(state: GraphState) -> dict:
    parts = []
    for slot in state["slots"]:
        a = slot["answer"]
        if slot["grounded"] is False and not a.abstained:
            a = Answer(a.text + f"\n\n> ⚠ Verification could not confirm every claim ({slot['note'][:120]}). "
                                "Check the cited clauses before relying on this.", a.abstained, a.citations, a.contexts)
        parts.append((slot["sub"], a))
    return {"result": combine(state["question"], parts), "trace": state["trace"] + ["format: done"]}


# --- edges ------------------------------------------------------------------
def after_verify(state: GraphState) -> str:
    retry = any(s["grounded"] is False and s["attempts"] <= MAX_RETRIES for s in state["slots"])
    return "reformulate" if retry else "format"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("check_amendments", check_amendments)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify)
    g.add_node("reformulate", reformulate)
    g.add_node("format", format_node)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", lambda s: "done" if s.get("result") else "retrieve",
                            {"done": END, "retrieve": "retrieve"})
    g.add_edge("retrieve", "check_amendments")
    g.add_edge("check_amendments", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, {"reformulate": "reformulate", "format": "format"})
    g.add_edge("reformulate", "retrieve")
    g.add_edge("format", END)
    return g.compile()


def run(question: str) -> tuple[Result, list[str]]:
    out = build_graph().invoke({"question": question, "comparison": False, "slots": [], "trace": [], "result": None})
    return out["result"], out["trace"]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Compare the net metering capacity limits in Gujarat and Rajasthan."
    res, trace = run(q)
    print("\n".join(f"  · {t}" for t in trace))
    print(f"\n{res.answer}\n")
    for c in res.citations:
        print(f"  [{c['id']}] {c['state']} | {c['title']} | clause {c['clause']} | {c['date']}")
