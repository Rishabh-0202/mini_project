"""Fast unit tests for deterministic components (no Ollama, no index needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from regrag.amendments import _refers_to, temporal_rank  # noqa: E402
from regrag.generate import ABSTAIN_MESSAGE, generate  # noqa: E402
from regrag.ingest import Chunk, amended_refs, split_clauses  # noqa: E402
from regrag.query import decompose, detect_states, out_of_scope_message  # noqa: E402
from regrag.retrieve import Hit  # noqa: E402


def chunk(**kw) -> Chunk:
    base = dict(chunk_id="x", doc_id="D", text="t", clause="", section_title="", page=1, state="Gujarat",
                authority="GERC", doc_type="principal_regulation", title="T", date="2016-01-01",
                status="final", amends="", source_url="", refs=[])
    return Chunk(**{**base, **kw})


# --- query understanding ------------------------------------------------------
def test_detects_state_names_regulators_and_discoms():
    assert detect_states("Net metering limit in Gujarat?") == ["Gujarat"]
    assert detect_states("What does MERC say about banking?") == ["Maharashtra"]
    assert detect_states("BESCOM rooftop tariff") == ["Karnataka"]
    assert detect_states("How does net metering work?") == []


def test_comparison_is_split_into_one_subquery_per_state():
    subs = decompose("Compare transformer limits in Gujarat and Maharashtra")
    assert [s.state for s in subs] == ["Gujarat", "Maharashtra"]
    assert all(s.text.endswith(f"in {s.state}?") for s in subs)
    assert "Maharashtra" not in subs[0].text
    subs = decompose("What is the maximum net metering capacity in Gujarat versus Rajasthan?")
    assert subs[1].text == "What is the maximum net metering capacity in Rajasthan?"


def test_uncovered_state_abstains_up_front():
    assert "Kerala" in out_of_scope_message("Net metering limit in Kerala?")
    assert out_of_scope_message("Compare Kerala and Gujarat") is None     # Gujarat is still answerable
    assert out_of_scope_message("Net metering in Rajasthan") is None


# --- clause chunking ----------------------------------------------------------
def test_clause_splitter_ignores_table_rows_and_quantities():
    lines = [(1, l) for l in [
        "1 Short title and commencement",
        "2 Definitions",
        "3 Eligible Consumer and individual project capacity",
        "3.1 The Eligible Consumer shall be a consumer of the licensee.",
        "3.2 The maximum capacity shall be 50% of sanctioned load;",
        "1 MW capacity is the upper bound.",          # quantity, not a clause
        "2 Approval from Chief Electrical Inspector",  # table row restarting at 2
        "4 Procedure for Application",
    ]]
    clauses = [b["clause"] for b in split_clauses(lines)]
    assert clauses == ["1", "2", "3", "3.1", "3.2", "4"]


def test_amending_documents_keep_quoted_clauses_inside_the_amending_paragraph():
    lines = [(1, l) for l in [
        "1. Short Title",
        "2. Commencement",
        "3. Amendment in Clause 5.1 of the Principal Regulations.",
        "5.1 The distribution licensee shall update capacity...",
        "4. Regulation 7 of the Principal Regulations shall be substituted as under:",
    ]]
    clauses = [b["clause"] for b in split_clauses(lines, heading_style="dotted", strict_subclauses=True)]
    assert clauses == ["1", "2", "3", "4"]


def test_amended_refs_needs_an_amending_verb_and_ignores_years():
    assert amended_refs("Regulation 7 of the Principal Regulations shall be substituted") == ["7"]
    assert amended_refs("Capacity (Subject to Regulation 6.2) of the rooftop") == []
    assert amended_refs("the RERC DREGS Regulations 2021 were amended") == []


# --- amendment handling -------------------------------------------------------
def test_refers_to_matches_clause_parent_and_consolidated():
    amd = chunk(doc_type="amendment", refs=["7"])
    assert _refers_to(amd, "7.3") and _refers_to(amd, "7") and not _refers_to(amd, "17.1")
    consol = chunk(doc_type="consolidated_regulation", clause="6.2")
    assert _refers_to(consol, "6.2") and not _refers_to(consol, "6.1")


def test_temporal_rank_puts_superseded_and_draft_after_current():
    old = Hit(chunk(chunk_id="old", date="2016-01-01"), rerank=9.0, flags=["POSSIBLY SUPERSEDED - ..."])
    new = Hit(chunk(chunk_id="new", date="2024-01-01"), rerank=5.0)
    draft = Hit(chunk(chunk_id="draft", status="draft"), rerank=8.0)
    assert [h.chunk.chunk_id for h in temporal_rank([old, draft, new])][0] == "new"


# --- generation guardrail -----------------------------------------------------
def test_low_rerank_score_abstains_without_calling_the_llm():
    weak = [Hit(chunk(), rerank=-11.0)]
    ans = generate("anything", weak, model=object())   # model is never invoked
    assert ans.abstained and ans.text == ABSTAIN_MESSAGE


def test_eval_key_fact_normalisation():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    from run_eval import key_facts
    assert key_facts("up to one megawatt (01 MW) and 65% of peak; INR 10,000") >= {"1mw", "65%", "rs10000"}
    assert key_facts("at least thirty percent of monthly consumption") == {"30%"}
    assert key_facts("Rs, Rs. and INR without numbers") == set()


def test_grouped_citations_are_parsed():
    from regrag.generate import cited_ids
    assert cited_ids("limit is 1 MW [S2, S3] and 70% [S1]; see [S4; S5]") == [2, 3, 1, 4, 5]
