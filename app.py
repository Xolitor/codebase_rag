import streamlit as st
from rag import search, generate_answer_stream
from ingest import init_collection, ingest_codebase

st.title("💬 Codebase RAG Chatbot")

st.sidebar.title("Instructions")
st.sidebar.write("""
1. Click "Ingest codebase" to index your code files (make sure to have a `repo` folder with code files in the same directory as this app).
2. Once ingested, ask questions about your codebase in the input box.
3. View the generated answer, sources, and retrieved chunks.
""")

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

if "ingested" not in st.session_state:
    st.session_state.ingested = False

if st.sidebar.button("Ingest codebase"):
    init_collection()
    ingest_stats = ingest_codebase("./repo")

    if isinstance(ingest_stats, dict):
        indexed_count = ingest_stats.get("chunks_indexed", 0)
        total_tokens = ingest_stats.get("total_tokens_used", 0)
        avg_tokens = ingest_stats.get("avg_tokens_per_chunk", 0.0)
        total_cost = ingest_stats.get("total_embedding_cost_usd", 0.0)
    else:
        indexed_count = ingest_stats
        total_tokens = 0
        avg_tokens = 0.0
        total_cost = 0.0

    st.session_state.ingested = indexed_count > 0
    if st.session_state.ingested:
        st.success(
            f"Codebase indexed! ({indexed_count} chunks)\n"
            f"Total embedding tokens: {total_tokens} | "
            f"Avg tokens/chunk: {avg_tokens:.2f} | "
            f"Estimated embedding cost: ${total_cost:.8f}"
        )
    else:
        st.warning("No code chunks were indexed. Check your repo folder and file types.")

query = st.text_input("Ask a question:")

if query:
    results = search(query)
    answer_stream = generate_answer_stream(query, results)

    st.write('### Answer')
    st.write_stream(answer_stream)

    with st.expander("Sources"):
        for r in results:
            st.write(r["source"])

    with st.expander("Retrieved chunks"):
        for r in results:
            st.write(r["text"][:300])

