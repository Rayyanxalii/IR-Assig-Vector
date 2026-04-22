# 23k-0532
# Rayyan Ali

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vsm_ir import (
    Preprocessor, VSMIndex, QueryProcessor,
    load_corpus, parse_query_file, evaluate,
    SPEECHES_DIR, STOPWORD_FILE, QUERY_FILE, ALPHA, _lemmatize,
)

# Page config
st.set_page_config(
    page_title="VSM IR – Trump Speeches",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Animated result card */
.result-card {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 14px 20px;
    margin-bottom: 10px;
    animation: fadeUp 0.35s ease both;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.25);
    border-color: rgba(99,102,241,0.7);
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Score progress bar */
.score-track {
    background: rgba(255,255,255,0.08);
    border-radius: 6px; height: 7px; margin-top: 10px;
}
.score-fill {
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    height: 100%; border-radius: 6px;
    transition: width 0.9s cubic-bezier(.4,0,.2,1);
}

/* Rank badge */
.rank {
    display: inline-flex; align-items: center; justify-content: center;
    width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; font-weight: 700; font-size: 13px;
    margin-right: 10px; flex-shrink: 0;
}

/* Metric card */
.kpi-card {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
}
.kpi-value {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-label { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; }

/* Section subtitle */
.subtitle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.4rem; }

/* Search bar override */
div[data-testid="stTextInput"] input {
    font-size: 1.1rem !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# Loads & cache the VSM index
@st.cache_resource(show_spinner=False)
def load_everything():
    pp = Preprocessor(STOPWORD_FILE)
    corpus = load_corpus(SPEECHES_DIR)
    idx = VSMIndex()
    idx.build(corpus, pp)
    qs = parse_query_file(QUERY_FILE)
    return pp, idx, qs


with st.spinner("Building VSM index — please wait..."):
    preprocessor, index, queries = load_everything()

qp = QueryProcessor(index, preprocessor)


# Sidebar
with st.sidebar:
    st.markdown("## VSM IR System")
    st.divider()

    page = st.radio(
        "Go to",
        ["Free-Text Search", "Benchmark Queries", "Term Inspector", "Index Statistics"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"**Documents:** {index.N}")
    st.markdown(f"**Vocabulary:** {len(index.vocabulary):,} terms")
    st.markdown(f"**Alpha (a):** {ALPHA}")


# PAGE 1 – FREE-TEXT SEARCH
if page == "Free-Text Search":
    st.markdown("# Free-Text Search")
    st.markdown('<p class="subtitle">Search across all Trump speeches using TF-IDF cosine similarity.</p>', unsafe_allow_html=True)

    col_q, col_a = st.columns([5, 1])
    with col_q:
        query = st.text_input("", placeholder="e.g.  immigration reform   |   american energy   |   hillary clinton", label_visibility="collapsed")
    with col_a:
        alpha_val = st.number_input("α threshold", min_value=0.0, max_value=1.0, value=ALPHA, step=0.001, format="%.3f")

    if query.strip():
        results = qp.search(query.strip(), alpha=alpha_val)

        if not results:
            st.warning(f"No documents found above α = {alpha_val:.3f}. Try lowering the threshold or rephrasing.")
        else:
            st.markdown(f"### {len(results)} document(s) matched")
            max_score = results[0][1]
            for rank, (doc_id, score) in enumerate(results, 1):
                pct = int((score / max_score) * 100)
                delay = f"{rank * 0.04:.2f}s"
                st.markdown(f"""
                <div class="result-card" style="animation-delay:{delay}">
                  <div style="display:flex; align-items:center; justify-content:space-between;">
                    <div><span class="rank">{rank}</span>
                      <strong style="font-size:1.05rem;">speech_{doc_id}</strong>
                    </div>
                    <span style="color:#a78bfa; font-weight:600; font-size:1rem;">Cosine Score: {score:.5f}</span>
                  </div>
                  <div class="score-track"><div class="score-fill" style="width:{pct}%"></div></div>
                </div>""", unsafe_allow_html=True)


# PAGE 2 – BENCHMARK QUERIES
elif page == "Benchmark Queries":
    st.markdown("# Benchmark Queries")
    st.markdown(f'<p class="subtitle">Evaluate all {len(queries)} predefined queries against gold-standard document sets.</p>', unsafe_allow_html=True)

    if st.button("Run All Benchmark Queries", type="primary", use_container_width=True):
        total_p = total_r = total_f = 0.0
        all_rows = []

        bar = st.progress(0, text="Running queries…")
        for i, q_info in enumerate(queries):
            res = qp.search(q_info["query"], alpha=ALPHA)
            ret_ids = {d for d, _ in res}
            exp = q_info["expected_docs"]
            exp_len = q_info["expected_len"] or len(exp)
            m = evaluate(ret_ids, exp) if exp else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            total_p += m["precision"]; total_r += m["recall"]; total_f += m["f1"]
            all_rows.append({"idx": i + 1, "q_info": q_info, "results": res,
                              "ret_ids": ret_ids, "metrics": m, "exp_len": exp_len})
            bar.progress((i + 1) / len(queries), text=f"Query {i + 1}/{len(queries)}")
        bar.empty()

        n = len(queries)
        # Macro-average KPIs
        c1, c2, c3 = st.columns(3)
        for col, label, val, color in zip(
            [c1, c2, c3],
            ["Macro Precision", "Macro Recall", "Macro F1"],
            [total_p / n, total_r / n, total_f / n],
            ["#10b981", "#6366f1", "#f59e0b"],
        ):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-value" style="background:linear-gradient(135deg,{color},{color}cc);
                       -webkit-background-clip:text;">{val:.3f}</div>
                  <div class="kpi-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Per-query expandable detail
        st.markdown("### Per-Query Details")
        for r in all_rows:
            p, rec, f1 = r["metrics"]["precision"], r["metrics"]["recall"], r["metrics"]["f1"]
            with st.expander(
                f"Q{r['idx']}: {r['q_info']['query'][:65]}  —  "
                f"Retr: {len(r['ret_ids'])}  Gold: {r['exp_len']}  F1: {f1:.3f}"
            ):
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{p:.3f}")
                m2.metric("Recall",    f"{rec:.3f}")
                m3.metric("F1",        f"{f1:.3f}")

                if r["results"]:
                    st.markdown("**Retrieved Documents:**")
                    for rank, (doc_id, score) in enumerate(r["results"], 1):
                        st.markdown(f"- **{rank}.** `speech_{doc_id}` — Cosine Score: **{score:.5f}**")
                else:
                    st.info("No documents retrieved for this query.")


# PAGE 3 – TERM INSPECTOR
elif page == "Term Inspector":
    st.markdown("# Term Inspector")
    st.markdown('<p class="subtitle">Look up any term\'s IDF, document frequency, and TF-IDF weights across the corpus.</p>', unsafe_allow_html=True)

    term_raw = st.text_input("", placeholder="Enter a term to inspect…  e.g. immigration", label_visibility="collapsed")

    if term_raw.strip():
        lemma = _lemmatize(term_raw.strip().lower())

        c1, c2 = st.columns(2)
        c1.metric("Input Term", term_raw.strip())
        c2.metric("Lemma (WordNet morphy)", lemma)

        if lemma not in index.idf:
            st.error(f"**'{lemma}'** is not in the index vocabulary. Try a different form of the word.")
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric("IDF Score",          f"{index.idf[lemma]:.4f}")
            k2.metric("Document Frequency", f"{index.df[lemma]:,}")
            k3.metric("% of Corpus",        f"{index.df[lemma] / index.N * 100:.1f}%")

            docs = [(d, v) for d, vec in index.tfidf.items() if (v := vec.get(lemma, 0)) > 0]
            docs.sort(key=lambda x: x[1], reverse=True)

            st.markdown(f"### Found in **{len(docs)}** document(s)")
            for rank, (doc_id, tfidf_score) in enumerate(docs, 1):
                tf_val = index.tf[doc_id].get(lemma, 0)
                pct = int((tfidf_score / docs[0][1]) * 100)
                delay = f"{rank * 0.03:.2f}s"
                st.markdown(f"""
                <div class="result-card" style="animation-delay:{delay}">
                  <div style="display:flex; align-items:center; justify-content:space-between;">
                    <div><span class="rank">{rank}</span>
                      <strong style="font-size:1.05rem;">speech_{doc_id}</strong>
                    </div>
                    <span style="color:#a78bfa; font-weight:600; font-size:1rem;">TF-IDF: {tfidf_score:.5f} &nbsp;|&nbsp; TF: {tf_val}</span>
                  </div>
                  <div class="score-track"><div class="score-fill" style="width:{pct}%"></div></div>
                </div>""", unsafe_allow_html=True)


# PAGE 4 – INDEX STATISTICS
elif page == "Index Statistics":
    st.markdown("# Index Statistics")
    st.markdown('<p class="subtitle">Overview of the built TF-IDF index.</p>', unsafe_allow_html=True)

    avg_terms = sum(len(v) for v in index.tfidf.values()) / index.N

    s1, s2, s3 = st.columns(3)
    for col, label, val in zip(
        [s1, s2, s3],
        ["Total Documents", "Vocabulary Size", "Avg Terms / Doc"],
        [str(index.N), f"{len(index.vocabulary):,}", f"{avg_terms:.1f}"],
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-value">{val}</div>
              <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    col_idf, col_df = st.columns(2)

    with col_idf:
        st.markdown("### Top 10 by IDF *(most specific)*")
        top_idf = sorted(index.idf.items(), key=lambda x: x[1], reverse=True)[:10]
        for rank, (term, idf_val) in enumerate(top_idf, 1):
            st.markdown(f"**{rank}.** `{term}` — IDF: **{idf_val:.4f}**")

    with col_df:
        st.markdown("### Top 10 by DF *(most common)*")
        top_df = sorted(index.df.items(), key=lambda x: x[1], reverse=True)[:10]
        for rank, (term, df_val) in enumerate(top_df, 1):
            st.markdown(f"**{rank}.** `{term}` — DF: **{df_val}**")

    st.markdown("")
    st.markdown("### Vocabulary Coverage")
    very_rare = sum(1 for v in index.df.values() if v == 1)
    rare      = sum(1 for v in index.df.values() if 2 <= v <= 5)
    common    = sum(1 for v in index.df.values() if v > 5)
    st.markdown(f"- **Hapax (DF=1):** {very_rare:,} terms")
    st.markdown(f"- **Rare (DF 2–5):** {rare:,} terms")
    st.markdown(f"- **Common (DF>5):** {common:,} terms")
