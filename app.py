"""Streamlit demo:  streamlit run app.py"""
import json
import time
import warnings

import streamlit as st

warnings.filterwarnings("ignore")

from regrag import config  # noqa: E402
from regrag.query import decompose, out_of_scope_message  # noqa: E402

st.set_page_config(page_title="RE Regulations RAG", page_icon="☀️", layout="wide")

EXAMPLES = [
    "What is the current net metering capacity limit in Rajasthan?",
    "Is a residential consumer in Gujarat limited to 50% of sanctioned load for rooftop solar?",
    "Compare the distribution transformer capacity limits for rooftop solar in Gujarat and Maharashtra.",
    "What is the KERC approved tariff for 1 kW to 10 kW domestic rooftop solar for FY24?",
    "Under the proposed 2024 Tamil Nadu regulations, what capacity range is allowed for gross metering?",
    "What banking provisions apply to green energy open access consumers under the Ministry of Power rules?",
    "What is the net metering limit for rooftop solar in Kerala?",
]


@st.cache_resource(show_spinner="Loading indexes and reranker...")
def load():
    from regrag.pipeline import components
    return components()


@st.cache_data
def manifest():
    return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))["documents"]


with st.sidebar:
    st.header("Settings")
    use_agent = st.toggle("Agentic mode (LangGraph)", value=False,
                          help="Adds a groundedness check and re-retrieves with a reformulated query if the answer "
                               "is not supported. Roughly 2x slower.")
    st.caption(f"LLM `{config.LLM_MODEL}` · embeddings `{config.EMBED_MODEL}` · reranker `{config.RERANK_MODEL.split('/')[-1]}`")
    st.divider()
    st.subheader("Example questions")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.q = ex
    st.divider()
    with st.expander("Indexed documents"):
        for d in manifest():
            mark = " ⛔ skipped" if d.get("skip") else (" ✏️ draft" if d["status"] == "draft" else "")
            st.markdown(f"**{d['state']}** · {d['date']} · [{d['title']}]({d['source_url']}){mark}")

st.title("☀️ State Renewable-Energy Regulation Assistant")
st.caption("Answers from SERC / MoP orders for Gujarat, Maharashtra, Karnataka, Tamil Nadu and Rajasthan, "
           "with clause-level citations and amendment checks. Not legal advice - verify against the cited order.")

question = st.text_input("Ask a question", key="q",
                         placeholder="e.g. What is the net metering limit for a 2 MW commercial plant in Gujarat?")

if question:
    oos = out_of_scope_message(question)
    subs = decompose(question)
    route = "out of scope" if oos else ", ".join(s.state or "all states" for s in subs)
    st.markdown(f"**Routing:** {route}" + (f" · {len(subs)} sub-queries" if len(subs) > 1 else ""))

    load()
    t = time.time()
    with st.spinner("Retrieving, checking amendments and generating (local CPU inference can take 1-2 minutes)..."):
        if use_agent:
            from regrag.graph import run as run_agent
            result, trace = run_agent(question)
        else:
            from regrag.pipeline import answer
            result, trace = answer(question), []
    elapsed = time.time() - t

    (st.warning if result.abstained else st.success)(
        "Abstained - insufficient context" if result.abstained else f"Answered in {elapsed:.0f}s")
    st.markdown(result.answer)

    if trace:
        with st.expander("Agent trace"):
            for step in trace:
                st.markdown(f"- `{step}`")

    if result.citations:
        st.subheader("Sources")
        for c in result.citations:
            badges = []
            if c["status"] == "draft":
                badges.append("✏️ DRAFT")
            if any(f.startswith("POSSIBLY SUPERSEDED") for f in c["flags"]):
                badges.append("⚠️ possibly superseded")
            if any(f.startswith("AMENDMENT") for f in c["flags"]):
                badges.append("🔁 amendment")
            label = (f"[{c['id']}] {c['state']} · {c['title']} · clause {c['clause'] or '-'} · "
                     f"{c['date']} · p.{c['page']} {' '.join(badges)}")
            with st.expander(label):
                for f in c["flags"]:
                    st.info(f)
                st.text(c["text"])
                st.markdown(f"[Source PDF]({c['source_url']}) · rerank score {c['rerank']}")
