"""Pitch-deck figures from the saved evaluation results -> docs/figures/*.png (+ .svg).

    python scripts/make_figures.py

Every number is read from eval/results/, data/manifest.json and data/processed/chunks.jsonl,
so the figures stay in sync when the evaluation is re-run. Palette: the dataviz reference
palette (validated: categorical blue/orange, ordinal blue ramp).
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
OUT = ROOT / "docs" / "figures"

# --- tokens (dataviz reference palette, light) ---------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"                          # categorical slots 1-2
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue 250..650
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential 100..700
W, H, DPI = 16, 9, 120                                        # 1920 x 1080

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 16, "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "axes.linewidth": 1,
    "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelsize": 15, "ytick.labelsize": 17,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "svg.fonttype": "none",
})


def load_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def header(fig, title, subtitle):
    fig.text(0.06, 0.92, title, fontsize=34, fontweight="bold", color=INK, va="bottom")
    fig.text(0.06, 0.875, subtitle, fontsize=19, color=INK2, va="bottom")


def footer(fig, text):
    fig.text(0.06, 0.035, text, fontsize=14, color=MUTED)


def clean_axes(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left" if grid_axis == "x" else "bottom"].set_color(BASELINE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=DPI, facecolor=SURFACE)
    fig.savefig(OUT / f"{name}.svg", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# --- 1. KPI tiles --------------------------------------------------------------------
def kpi_tiles(retr, ans):
    tiles = [
        (f"{retr['hit_in_llm_context']:.0%}", "of questions: the\nright clause reached\nthe model"),
        (f"{retr['contamination_rate']:.0%}", "of answers drew on\nanother state's rules"),
        (f"{ans['abstain_accuracy']:.0%}", "correct answer-or-\ndecline decisions"),
        (f"{ans['citation_rate']:.1%}", "of answers cite the\nclause, order and date"),
    ]
    fig = plt.figure(figsize=(W, H))
    header(fig, "RegRAG at a glance",
           f"Evaluated on {ans['answered']} hand-written questions across 5 states + central rules")
    left, gap, top, h = 0.06, 0.025, 0.7, 0.44
    w = (0.88 - gap * 3) / 4
    for i, (value, label) in enumerate(tiles):
        x = left + i * (w + gap)
        fig.patches.append(FancyBboxPatch((x, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.015",
                                          transform=fig.transFigure, facecolor="#f3f2ee",
                                          edgecolor="#e6e4dd", linewidth=1))
        fig.text(x + 0.022, top - 0.2, value, fontsize=60, fontweight="bold", color=INK, va="bottom")
        fig.text(x + 0.022, top - 0.235, label, fontsize=18, color=INK2, va="top", linespacing=1.4)
    footer(fig, "Source: eval/results (retrieval.json, answers_pipeline.jsonl). Local model: llama3.2 (3B) on a CPU-only laptop.")
    save(fig, "01_kpi_tiles")


# --- 2. Retrieval vs generation ------------------------------------------------------
def retrieval_vs_generation(retr, ans, rag):
    retrieval = [("Right clause reached the model", retr["hit_in_llm_context"]),
                 ("Source in top 5 (hit@5)", retr["hit@5"]),
                 ("Mean reciprocal rank", retr["MRR"]),
                 ("RAGAS context precision", rag["llm_context_precision_with_reference"])]
    generation = [("Answers with citations", ans["citation_rate"]),
                  ("RAGAS answer relevancy", rag["answer_relevancy"]),
                  ("RAGAS faithfulness", rag["faithfulness"]),
                  ("Key numbers present in answer", ans["key_fact_recall"])]
    rows = [(n, v, BLUE) for n, v in retrieval] + [None] + [(n, v, ORANGE) for n, v in generation]

    fig = plt.figure(figsize=(W, H))
    header(fig, "Retrieval is near-perfect; generation is the gap",
           "Every metric on a 0–1 scale. Blue: finding the right clause. Orange: answering from it.")
    ax = fig.add_axes([0.33, 0.12, 0.6, 0.7])
    y = list(range(len(rows)))[::-1]
    for yi, row in zip(y, rows):
        if row is None:
            continue
        name, v, color = row
        ax.barh(yi, v, height=0.62, color=color)
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=17, color=INK, fontweight="bold")
    ax.set_yticks([yi for yi, r in zip(y, rows) if r], [r[0] for r in rows if r])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1.0"])
    clean_axes(ax, "x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["Retrieval", "Generation (llama3.2, 3B)"], loc="upper right",
               bbox_to_anchor=(0.94, 0.875), frameon=False, ncol=2, fontsize=17)
    footer(fig, "RAGAS judge: Groq openai/gpt-oss-120b. Context precision scored on 31 of 32 answered questions.")
    save(fig, "02_retrieval_vs_generation")


# --- 3. Faithfulness by state ------------------------------------------------------------
GROUPS = {"GJ": "Gujarat", "MH": "Maharashtra", "RJ": "Rajasthan", "KA": "Karnataka",
          "TN": "Tamil Nadu", "CT": "Central rules", "MS": "Comparisons"}


def faithfulness_by_state(rag_df, overall):
    rag_df = rag_df.assign(group=rag_df["id"].str[:2].map(GROUPS))
    g = rag_df.groupby("group")["faithfulness"].agg(["mean", "count"]).sort_values("mean", ascending=False)

    fig = plt.figure(figsize=(W, H))
    header(fig, "Faithfulness is lowest for Gujarat and multi-state comparisons",
           "Mean RAGAS faithfulness per group. Gujarat mixes fee tables with amendments; "
           "comparisons merge two states.")
    ax = fig.add_axes([0.08, 0.16, 0.86, 0.64])
    x = range(len(g))
    ax.bar(x, g["mean"], width=0.58, color=BLUE)
    for xi, (m, n) in zip(x, zip(g["mean"], g["count"])):
        ax.text(xi, m + 0.02, f"{m:.2f}", ha="center", fontsize=19, fontweight="bold", color=INK)
        ax.text(xi, -0.09, f"n = {int(n)}", ha="center", fontsize=14, color=MUTED, transform=ax.get_xaxis_transform())
    ax.axhline(overall, color=INK2, linewidth=1.5)
    ax.text(len(g) - 0.5, overall + 0.015, f"All questions: {overall:.2f}", ha="right", fontsize=15, color=INK2)
    ax.set_xticks(list(x), g.index)
    ax.tick_params(axis="x", labelsize=16, labelcolor=INK2, pad=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1.0"])
    clean_axes(ax, "y")
    footer(fig, "Faithfulness = share of the answer's claims supported by the retrieved clauses (RAGAS, judge: gpt-oss-120b). Small groups: read as indicative.")
    save(fig, "03_faithfulness_by_state")


# --- 4. Regulation timeline ---------------------------------------------------------------
KIND = {"principal_regulation": "base", "tariff_order": "base", "central_rules": "base", "discom_summary": "base",
        "consolidated_regulation": "amend", "amendment": "amend", "statement_of_reasons": "amend", "order": "amend",
        "draft_regulation": "draft"}
NOTES = {  # doc_id -> annotation (facts verified against the PDFs during the build)
    "RJ_SUOMOTU_2024": "Net metering cap 500 kW → 1 MW",
    "GJ_NM_CONSOL": "Residential exempted\nfrom 50% cap",
    "GJ_NM_AMD4_2024": "Up to 6 kW: grid upgrade\ncost moved to the discom",
    "TN_GISS_2024_DRAFT": "Draft: gross metering\n150–999 kW",
}


def regulation_timeline():
    docs = [d for d in json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))["documents"]
            if not d.get("skip")]
    lanes = ["Gujarat", "Maharashtra", "Karnataka", "Tamil Nadu", "Rajasthan", "Central"]
    kinds = Counter(KIND[d["doc_type"]] for d in docs)

    fig = plt.figure(figsize=(W, H))
    header(fig, "The rules keep moving",
           f"{len(docs)} indexed documents: {kinds['base']} base regulations or orders, "
           f"{kinds['amend']} amendments or revisions, {kinds['draft']} draft")
    ax = fig.add_axes([0.14, 0.14, 0.8, 0.66])
    yof = {s: i for i, s in enumerate(lanes[::-1])}
    for lane in lanes:
        dates = [date.fromisoformat(d["date"]) for d in docs if d["state"] == lane]
        ax.plot([min(dates), max(dates)], [yof[lane]] * 2, color=BASELINE, linewidth=2, zorder=1)
    for d in docs:
        dt, y, k = date.fromisoformat(d["date"]), yof[d["state"]], KIND[d["doc_type"]]
        if k == "base":
            ax.scatter(dt, y, s=260, marker="o", color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        elif k == "amend":
            ax.scatter(dt, y, s=300, marker="D", color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3)
        else:
            ax.scatter(dt, y, s=260, marker="o", facecolor=SURFACE, edgecolor=MUTED, linewidth=2.5, zorder=3)
        if d["doc_id"] in NOTES:
            ax.annotate(NOTES[d["doc_id"]], (dt, y), xytext=(0, 22), textcoords="offset points",
                        ha="center", va="bottom", fontsize=14, color=INK2, linespacing=1.3)
    ax.set_yticks(list(yof.values()), list(yof.keys()))
    ax.set_ylim(-0.6, len(lanes) - 0.1)
    ax.set_xlim(date(2015, 9, 1), date(2025, 6, 1))
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    clean_axes(ax, "x")
    ax.spines["left"].set_visible(False)
    legend = [plt.Line2D([], [], marker="o", ls="", color=BLUE, markersize=13, label="Base regulation / order"),
              plt.Line2D([], [], marker="D", ls="", color=ORANGE, markersize=12, label="Amendment or revision"),
              plt.Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED,
                         markeredgewidth=2.5, markersize=13, label="Draft (not in force)")]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.94, 0.875), frameon=False, ncol=3, fontsize=16)
    footer(fig, "Source: data/manifest.json. One scanned KERC order (no text layer) is excluded; 2025 amendments had no downloadable PDF.")
    save(fig, "04_regulation_timeline")


# --- 5. Retrieval funnel --------------------------------------------------------------------
def retrieval_funnel(chunks):
    from regrag import config
    total = len(chunks)
    gujarat = sum(c["state"] == "Gujarat" for c in chunks)
    stages = [
        (f"{total}", "All clause chunks in the index"),
        (f"{gujarat}", "After the state filter (Gujarat)"),
        (f"≤ {config.SPARSE_K + config.DENSE_K}", f"BM25 top {config.SPARSE_K} + dense top {config.DENSE_K}, fused"),
        ("30", "Pool sent to the cross-encoder reranker"),
        (f"{config.RERANK_TOP_N}", "Clauses the model answers from (+ any later amendments)"),
    ]
    values = [total, gujarat, config.SPARSE_K + config.DENSE_K, 30, config.RERANK_TOP_N]

    fig = plt.figure(figsize=(W, H))
    header(fig, f"From {total} clauses to {config.RERANK_TOP_N} cited sources",
           "What happens to one Gujarat question. Each step narrows the search; nothing from other states survives.")
    ax = fig.add_axes([0.06, 0.1, 0.88, 0.72])
    ax.set_xlim(0, total * 1.02)
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.axis("off")
    for i, ((num, label), v, color) in enumerate(zip(stages, values, RAMP)):
        y = len(stages) - 1 - i
        width = max(v, total * 0.012)
        ax.barh(y, width, left=(total - width) / 2, height=0.42, color=color)
        ax.text(total / 2, y + 0.28, f"{num}  ·  {label}", ha="center", va="bottom", fontsize=18, color=INK)
    footer(fig, "Counts from data/processed/chunks.jsonl and regrag/config.py (DENSE_K, SPARSE_K, rerank pool, RERANK_TOP_N).")
    save(fig, "05_retrieval_funnel")


# --- 6. RAGAS heatmap (appendix) -------------------------------------------------------------
def ragas_heatmap(rag_df):
    from matplotlib.colors import ListedColormap

    cols = [("faithfulness", "Faithfulness"), ("llm_context_precision_with_reference", "Context precision"),
            ("answer_relevancy", "Answer relevancy")]
    data = rag_df[[c for c, _ in cols]].T.to_numpy(dtype=float)
    cmap = ListedColormap(SEQ)
    cmap.set_bad("#eeede8")

    fig = plt.figure(figsize=(W, H))
    header(fig, "Per-question RAGAS scores",
           "Darker = better. Most answers score well; the failures concentrate in a few questions (GJ06, MH04, MS01).")
    ax = fig.add_axes([0.14, 0.3, 0.8, 0.48])
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (r, c), v in pd.DataFrame(data).stack(future_stack=True).items():
        text = "–" if pd.isna(v) else f"{v:.1f}"
        ax.text(c, r, text, ha="center", va="center", fontsize=12,
                color=SURFACE if (not pd.isna(v) and v >= 0.6) else INK)
    ax.set_yticks(range(len(cols)), [n for _, n in cols])
    ax.set_xticks(range(len(rag_df)), rag_df["id"], rotation=90, fontsize=13, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks([x - 0.5 for x in range(1, len(rag_df))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(cols))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    cax = fig.add_axes([0.14, 0.11, 0.3, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1.0])
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=13, length=0, colors=MUTED)
    fig.text(0.46, 0.115, "–  = not scored (judge output failed to parse)", fontsize=13, color=MUTED)
    footer(fig, "IDs: GJ Gujarat · MH Maharashtra · RJ Rajasthan · KA Karnataka · TN Tamil Nadu · CT central rules · MS multi-state.")
    save(fig, "06_ragas_heatmap")


# --- 7. Technical workflow -------------------------------------------------------------------
def technical_workflow(n_chunks):
    import textwrap

    from regrag import config
    x0, x1, gap, cols = 0.06, 0.95, 0.02, 7
    w = (x1 - x0 - gap * (cols - 1)) / cols
    colx = [x0 + i * (w + gap) for i in range(cols)]

    fig = plt.figure(figsize=(W, H))
    header(fig, "Technical workflow",
           "Built once offline, then every question runs the online pipeline. Everything runs locally except the RAGAS judge.")

    def box(x, y, width, h, title, detail, edge="#e1e0d9", lw=1.0, wrap=17):
        fig.patches.append(FancyBboxPatch((x, y), width, h, boxstyle="round,pad=0,rounding_size=0.01",
                                          transform=fig.transFigure, facecolor="#f3f2ee", edgecolor=edge, linewidth=lw))
        fig.text(x + 0.008, y + h - 0.018, title, fontsize=15, fontweight="bold", color=INK, va="top")
        fig.text(x + 0.008, y + h - 0.062, "\n".join(textwrap.wrap(detail, wrap)), fontsize=12, color=INK2,
                 va="top", linespacing=1.35)

    def arrow(xa, ya, xb, yb, color=MUTED):
        fig.patches.append(matplotlib.patches.FancyArrowPatch((xa, ya), (xb, yb), transform=fig.transFigure,
                                                              arrowstyle="-|>", mutation_scale=18, color=color, lw=1.6))

    def lane(y, text):
        fig.text(x0, y, text, fontsize=13, fontweight="bold", color=INK2, va="bottom")

    # Offline lane
    oy, oh = 0.61, 0.19
    lane(oy + oh + 0.012, f"OFFLINE · BUILD ONCE  ({n_chunks} clause chunks)")
    offline = [("PDF corpus", "13 SERC / MoP documents: 5 states + central rules"),
               ("Extract text", "pypdf; strip page numbers, running headers, tables of contents"),
               ("Clause chunks", f"Split on clause numbers; {config.MIN_CHUNK_CHARS}–{config.MAX_CHUNK_CHARS} characters each"),
               ("Tag metadata", "State, authority, doc type, date, draft/final, amended clauses")]
    for i, (t, d) in enumerate(offline):
        box(colx[i], oy, w, oh, t, d)
        if i:
            arrow(colx[i - 1] + w, oy + oh / 2, colx[i], oy + oh / 2)
    ix = colx[5]
    box(ix, oy, 2 * w + gap, oh, "Indexes", f"ChromaDB, {config.EMBED_MODEL} embeddings (768-d, cosine) + BM25 keyword index",
        edge=BLUE, lw=2, wrap=38)
    arrow(colx[3] + w, oy + oh / 2, ix, oy + oh / 2)
    fig.text((colx[3] + w + ix) / 2, oy + oh / 2 + 0.015, "embed + index", ha="center", fontsize=12, color=MUTED)

    # Online lane
    ny, nh = 0.30, 0.22
    lane(ny + nh + 0.012, "ONLINE · EVERY QUESTION")
    online = [("Question", "e.g. “Net metering cap in Rajasthan?”"),
              ("Understand", "Detect state(s); split comparisons; decline if state not covered"),
              ("Hybrid search", f"BM25 top {config.SPARSE_K} + dense top {config.DENSE_K} inside the state filter; RRF fusion"),
              ("Rerank", f"MiniLM cross-encoder scores 30; keep top {config.RERANK_TOP_N}"),
              ("Amendments", "Find later orders; flag superseded clauses; newest first"),
              ("Generate", f"{config.LLM_MODEL} via Ollama; cite [S#] on every sentence"),
              ("Answer", "Clause, order and date, or “can't answer”")]
    for i, (t, d) in enumerate(online):
        box(colx[i], ny, w, nh, t, d, edge=BLUE if t == "Amendments" else "#e1e0d9", lw=2 if t == "Amendments" else 1)
        if i:
            arrow(colx[i - 1] + w, ny + nh / 2, colx[i], ny + nh / 2)

    # Indexes -> Hybrid search (elbow)
    ey = 0.565
    hx = colx[2] + w / 2
    sx = ix + w + gap / 2
    fig.lines.append(plt.Line2D([sx, sx], [oy, ey], transform=fig.transFigure, color=MUTED, lw=1.6))
    fig.lines.append(plt.Line2D([sx, hx], [ey, ey], transform=fig.transFigure, color=MUTED, lw=1.6))
    arrow(hx, ey, hx, ny + nh)
    fig.text((sx + hx) / 2, ey + 0.008, "retrieve from both indexes (state-filtered)", ha="center", va="bottom",
             fontsize=12, color=MUTED)

    # Agent loop: Generate -> verify -> back to Hybrid search
    ly = 0.255
    gx = colx[5] + w / 2
    fig.lines.append(plt.Line2D([gx, gx], [ny, ly], transform=fig.transFigure, color=ORANGE, lw=2.2))
    fig.lines.append(plt.Line2D([gx, hx], [ly, ly], transform=fig.transFigure, color=ORANGE, lw=2.2))
    arrow(hx, ly, hx, ny, color=ORANGE)
    fig.text((gx + hx) / 2, ly - 0.012, "Agent mode (LangGraph): verify the answer is grounded → rewrite the query "
             "→ retry once; roll back if the retry retrieves worse sources", ha="center", va="top", fontsize=12.5,
             color=INK)

    # Evaluation lane
    vy, vh = 0.05, 0.1
    lane(vy + vh + 0.012, "EVALUATE")
    fig.patches.append(FancyBboxPatch((x0, vy), x1 - x0, vh, boxstyle="round,pad=0,rounding_size=0.01",
                                      transform=fig.transFigure, facecolor="#f3f2ee", edgecolor="#e1e0d9", linewidth=1))
    fig.text(x0 + 0.012, vy + vh / 2,
             "34 hand-written questions  →  retrieval metrics (hit@5, MRR, contamination; no LLM)  ·  "
             "answer metrics (decline accuracy, citations, key numbers)\n"
             "·  RAGAS faithfulness, context precision, answer relevancy (judge: Groq gpt-oss-120b)  ·  "
             "11 unit tests (pytest)",
             fontsize=13, color=INK2, va="center", linespacing=1.5)
    save(fig, "07_technical_workflow")


# --- 8. Models & workflow (simplified) --------------------------------------------------------
def models_and_workflow(n_chunks):
    from regrag import config
    CHIP = "#e6effb"
    mono = "Consolas"
    steps = [  # (title, chips, spec)
        ("Ingest", ["pypdf"], f"split by clause\n{n_chunks} chunks"),
        ("Embed & index", [config.EMBED_MODEL], "ChromaDB + BM25\n768-d, cosine"),
        ("Retrieve", ["BM25 + dense"], f"top {config.SPARSE_K} + {config.DENSE_K}\nstate filter, RRF"),
        ("Rerank", ["MiniLM-L6-v2"], f"cross-encoder\nkeeps top {config.RERANK_TOP_N}"),
        ("Amendments", ["dates + refs"], "flags superseded\nnewest first"),
        ("Generate", [f"{config.LLM_MODEL} · 3B"], "Ollama, temp 0\ncites or abstains"),
    ]
    x0, x1 = 0.06, 0.94
    cw = (x1 - x0) / len(steps)
    cx = [x0 + cw * (i + 0.5) for i in range(len(steps))]
    ly = 0.64

    fig = plt.figure(figsize=(W, H))
    header(fig, "How RegRAG works: models and tools",
           "Six steps from PDF to cited answer. Every model runs locally on a CPU laptop.")

    # offline / per-question brackets
    def bracket(a, b, label):
        y = 0.755
        xa, xb = cx[a] - cw * 0.4, cx[b] + cw * 0.4
        fig.lines.append(plt.Line2D([xa, xa, xb, xb], [y - 0.015, y, y, y - 0.015], transform=fig.transFigure,
                                    color=BASELINE, lw=1.5))
        fig.text((xa + xb) / 2, y + 0.012, label, ha="center", va="bottom", fontsize=15, fontweight="bold", color=INK2)
    bracket(0, 1, "OFFLINE · ONCE")
    bracket(2, 5, "EVERY QUESTION")

    fig.lines.append(plt.Line2D([cx[0], cx[-1]], [ly, ly], transform=fig.transFigure, color=BASELINE, lw=2, zorder=1))
    for i, (title, chips, spec) in enumerate(steps):
        r = 0.05  # diameter in figure height; width scaled by H/W so it renders as a circle
        fig.patches.append(matplotlib.patches.Ellipse((cx[i], ly), r * H / W, r, transform=fig.transFigure,
                                                      facecolor=BLUE, edgecolor=SURFACE, lw=3, zorder=2))
        fig.text(cx[i], ly, str(i + 1), ha="center", va="center", fontsize=22, fontweight="bold", color=SURFACE, zorder=3)
        fig.text(cx[i], 0.56, title, ha="center", va="top", fontsize=22, fontweight="bold", color=INK)
        for j, chip in enumerate(chips):
            fig.text(cx[i], 0.48 - 0.058 * j, chip, ha="center", va="top", fontsize=16, family=mono, color=INK,
                     bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.8", facecolor=CHIP, edgecolor="none"))
        fig.text(cx[i], 0.39, spec, ha="center", va="top", fontsize=16, color=INK2, linespacing=1.45)

    # bottom strip
    strip = [("ORCHESTRATION", "LangChain + LangGraph"),
             ("AGENT LOOP", "verify → rewrite → retry"),
             ("EVALUATION", "RAGAS · gpt-oss-120b judge"),
             ("DEMO UI", "Streamlit")]
    sy, sh, sgap = 0.06, 0.1, 0.015
    sw = (x1 - x0 - sgap * (len(strip) - 1)) / len(strip)
    for i, (k, v) in enumerate(strip):
        sx = x0 + i * (sw + sgap)
        fig.patches.append(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0,rounding_size=0.012",
                                          transform=fig.transFigure, facecolor="#f3f2ee", edgecolor="#e6e4dd", lw=1))
        fig.text(sx + 0.015, sy + sh - 0.022, k, fontsize=12.5, fontweight="bold", color=MUTED, va="top")
        fig.text(sx + 0.015, sy + 0.028, v, fontsize=16, color=INK, va="bottom")
    save(fig, "08_models_and_workflow")


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "eval"))
    from run_eval import answer_metrics

    retr = load_json("retrieval.json")["summary"]
    ans = answer_metrics(False)
    rag = load_json("ragas_pipeline.json")
    rag_df = pd.read_csv(RESULTS / "ragas_pipeline.csv")
    chunks = [json.loads(l) for l in (ROOT / "data" / "processed" / "chunks.jsonl").open(encoding="utf-8")]

    kpi_tiles(retr, ans)
    retrieval_vs_generation(retr, ans, rag)
    faithfulness_by_state(rag_df, rag["faithfulness"])
    regulation_timeline()
    retrieval_funnel(chunks)
    ragas_heatmap(rag_df)
    technical_workflow(len(chunks))
    models_and_workflow(len(chunks))


if __name__ == "__main__":
    main()
