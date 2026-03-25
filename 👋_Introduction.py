import streamlit as st
from rag import search, generate_answer_stream
from ingest import init_collection, ingest_codebase, ingest_codebase_from_github, ingest_codebase_from_uploads, qdrant
from config import COLLECTION_NAME, MODE

st.title("💬 Codebase RAG Chatbot")
st.caption(f"Current mode: {MODE}")

st.write("## Instructions")
st.write("""
1. Upload your codebase for ingestion by clicking the 'Ingest codebase' button, or provide a GitHub URL in the sidebar for direct ingestion from GitHub.
2. Once ingested, ask questions about the codebase in the input box.
3. View the generated answer, sources, retrieved chunks. You can also check the 'Vector Demo' page to see details about the vector database.
""")

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.sidebar.info("Select a page above.")

if MODE == "demo":
    url = st.sidebar.text_input(
        "GitHub URL",
        key="demo_github_url",
        placeholder="https://github.com/owner/repo",
    )

    if st.sidebar.button("Ingest from GitHub", use_container_width=True):
        if not url.strip():
            st.sidebar.warning("Enter a GitHub URL first.")
        else:
            with st.sidebar:
                with st.spinner("Ingesting from GitHub..."):
                    init_collection()
                    ingest_stats = ingest_codebase_from_github(url.strip())

            indexed_count = ingest_stats.get("chunks_indexed", 0) if isinstance(ingest_stats, dict) else ingest_stats
            if indexed_count > 0:
                st.sidebar.success(f"Codebase indexed from GitHub! ({indexed_count} chunks)")
            else:
                st.sidebar.warning("No code chunks were indexed from GitHub. Check the URL and file types.")

    st.sidebar.markdown("<h1 style='text-align: center; margin: 8px 0;'>OR</h1>", unsafe_allow_html=True)

    uploaded_files = st.sidebar.file_uploader(
        "Drag and drop code files",
        accept_multiple_files="directory",
        type=["py", "js", "ts", "java", "cpp", "c", "go", "rb", "php", "html", "css"],
        key="demo_upload_files",
    )

    if st.sidebar.button("Ingest uploaded files", use_container_width=True):
        if not uploaded_files:
            st.sidebar.warning("Upload a folder or files first.")
        else:
            with st.sidebar:
                with st.spinner("Ingesting uploaded files..."):
                    init_collection()
                    ingest_stats = ingest_codebase_from_uploads(uploaded_files)

            indexed_count = ingest_stats.get("chunks_indexed", 0) if isinstance(ingest_stats, dict) else ingest_stats
            if indexed_count > 0:
                st.sidebar.success(f"Uploaded files indexed! ({indexed_count} chunks)")
            else:
                st.sidebar.warning("No code chunks were indexed from uploaded files.")

def has_vector_data():
    try:
        if hasattr(qdrant, "collection_exists") and not qdrant.collection_exists(COLLECTION_NAME):
            return False

        count_response = qdrant.count(collection_name=COLLECTION_NAME, exact=False)
        return count_response.count > 0
    except Exception:
        return False

# if st.button("Ingest codebase"):
#     init_collection()
#     ingest_stats = ingest_codebase("./repo")

#     if isinstance(ingest_stats, dict):
#         indexed_count = ingest_stats.get("chunks_indexed", 0)
#     else:
#         indexed_count = ingest_stats

#     if indexed_count > 0:
#         st.success(f"Codebase indexed! ({indexed_count} chunks)")
#     else:
#         st.warning("No code chunks were indexed. Check your repo folder and file types.")

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

