import streamlit as st
from rag import search, generate_answer_stream
from ingest import init_collection, ingest_codebase, qdrant
from config import COLLECTION_NAME

st.title("💬 Codebase RAG Chatbot")

st.write("## Instructions")
st.write("""
1. Click "Ingest codebase" to index your code files (make sure to have a `repo` folder with code files in the same directory as this app).
2. Once ingested, ask questions about your codebase in the input box.
3. View the generated answer, sources, and retrieved chunks.
""")

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.sidebar.info("Select a page above.")


def has_vector_data():
    try:
        if hasattr(qdrant, "collection_exists") and not qdrant.collection_exists(COLLECTION_NAME):
            return False

        count_response = qdrant.count(collection_name=COLLECTION_NAME, exact=False)
        return count_response.count > 0
    except Exception:
        return False

if st.button("Ingest codebase"):
    init_collection()
    ingest_stats = ingest_codebase("./repo")

    if isinstance(ingest_stats, dict):
        indexed_count = ingest_stats.get("chunks_indexed", 0)
    else:
        indexed_count = ingest_stats

    if indexed_count > 0:
        st.success(f"Codebase indexed! ({indexed_count} chunks)")
    else:
        st.warning("No code chunks were indexed. Check your repo folder and file types.")

query = st.text_input("Ask a question:")

if query:
    if not has_vector_data():
        st.info("Ingest codebase first.")
    else:
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

