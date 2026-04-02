import streamlit as st
from ingest import qdrant, load_last_bm25_stats, infer_code_language
from config import COLLECTION_NAME

st.set_page_config(page_title="BM25 Database", page_icon="🔎")

st.markdown("# BM25 Database")
st.markdown(
    """
    <style>
    @media (max-width: 900px) {
        [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def has_vector_data():
    try:
        if hasattr(qdrant, "collection_exists") and not qdrant.collection_exists(COLLECTION_NAME):
            return False

        count_response = qdrant.count(collection_name=COLLECTION_NAME, exact=False)
        return count_response.count > 0
    except Exception:
        return False


if not has_vector_data():
    st.info("Ingest codebase first.")
    st.stop()

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
max_rows = 18

filtered_entries = entries
if source_filter.strip():
    source_filter_lower = source_filter.strip().lower()
    filtered_entries = [
        e for e in entries if source_filter_lower in str(e.get("source", "")).lower()
    ]

visible_entries = filtered_entries[:max_rows]
columns = st.columns(3, gap="medium")

for idx, entry in enumerate(visible_entries):
    with columns[idx % 3]:
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
