import streamlit as st
from ingest import load_last_bm25_stats, infer_code_language

st.set_page_config(page_title="BM25 Database", page_icon="🔎")

st.markdown("# BM25 Database")

bm25_stats = load_last_bm25_stats()

if not bm25_stats:
    st.info("Ingest codebase first to generate ingest_bm25.json.")
    st.stop()

entries = bm25_stats.get("entries", [])
tokenizer = bm25_stats.get("tokenizer", "N/A")
chunks_indexed = bm25_stats.get("chunks_indexed", 0)

with st.container(border=True):
    st.caption("Tokenizer")
    st.code(tokenizer, language="python")

with st.container(border=True):
    st.caption("BM25 Corpus Summary")
    st.write(f"Chunks indexed: {chunks_indexed}")
    st.write(f"Entries stored: {len(entries)}")

if not entries:
    st.info("No BM25 entries found. Ingest codebase first.")
    st.stop()

source_filter = st.text_input("Filter by source", placeholder="e.g. app.py")
max_rows = st.slider("Rows to display", min_value=1, max_value=50, value=10)

filtered_entries = entries
if source_filter.strip():
    source_filter_lower = source_filter.strip().lower()
    filtered_entries = [
        e for e in entries if source_filter_lower in str(e.get("source", "")).lower()
    ]

for entry in filtered_entries[:max_rows]:
    with st.container(border=True):
        st.subheader(f"Chunk {entry.get('id')}")
        st.caption(f"Source: {entry.get('source', 'Unknown source')}")
        st.write(f"Token count: {entry.get('token_count', 0)}")

        with st.expander("Tokens", expanded=False):
            st.write(entry.get("tokens_preview", []))

        with st.expander("Text Preview", expanded=False):
            source = entry.get("source", "")
            st.code(entry.get("text_preview", ""), language=infer_code_language(source))

if len(filtered_entries) > max_rows:
    st.caption(f"Showing {max_rows} of {len(filtered_entries)} matching entries.")
