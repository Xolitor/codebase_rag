import streamlit as st
import time
from rag import search, generate_answer_stream
from ingest import init_collection, ingest_codebase, ingest_codebase_from_github, ingest_codebase_from_uploads, qdrant, infer_code_language
from config import COLLECTION_NAME, MODE, MAX_INPUT_CHARS, RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS


def is_rate_limited(now_s):
    history_key = "rate_limit_history"
    history = st.session_state.get(history_key, [])
    history_before = len(history)

    # Keep only recent requests inside the current window.
    cutoff = now_s - RATE_LIMIT_WINDOW_SECONDS
    history = [ts for ts in history if ts >= cutoff]
    st.session_state[history_key] = history

    print(
        f"[rate_limit] check before={history_before} after={len(history)} "
        f"limit={RATE_LIMIT_MAX_REQUESTS}/{RATE_LIMIT_WINDOW_SECONDS}s"
    )

    if len(history) >= RATE_LIMIT_MAX_REQUESTS:
        retry_after_s = int(max(1, RATE_LIMIT_WINDOW_SECONDS - (now_s - history[0])))
        print(
            f"[rate_limit] blocked retry_after_s={retry_after_s} "
            f"recent_requests={len(history)}"
        )
        return True, retry_after_s

    print(f"[rate_limit] allowed recent_requests={len(history)}")
    return False, 0


def record_request(now_s):
    history_key = "rate_limit_history"
    history = st.session_state.get(history_key, [])
    history.append(now_s)
    st.session_state[history_key] = history
    print(f"[rate_limit] recorded total_recent_requests={len(history)}")

st.title("💬 Codebase RAG Assistant")
st.caption(f"Current mode: {MODE}")

st.write("## Instructions")
st.write("""
1. Ingest a codebase from the sidebar using a GitHub URL or uploaded files.
2. Ask a question about the ingested codebase in the input box.
3. Review the answer, sources, and retrieved chunks (with vector/BM25 scores and previews).
""")

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.sidebar.info("Select a page above.")

# if MODE == "demo":
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

if query and len(query) > MAX_INPUT_CHARS:
    print(f"[input_guard] blocked length={len(query)} max={MAX_INPUT_CHARS}")
    st.error("Input too long")
    st.stop()

if query:
    print(f"[query] received length={len(query)}")
    now_s = time.time()
    limited, retry_after_s = is_rate_limited(now_s)
    if limited:
        st.error(
            f"Rate limit exceeded. Try again in {retry_after_s}s "
            f"({RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s)."
        )
        st.stop()
    record_request(now_s)

if query:
    if not has_vector_data():
        st.info("Ingest codebase first.")
    else:
        request_started_at = time.perf_counter()
        retrieval_started_at = time.perf_counter()

        results = search(query)
        retrieval_latency_s = time.perf_counter() - retrieval_started_at

        answer_metrics = {}
        answer_stream = generate_answer_stream(query, results, answer_metrics)

        st.write('### Answer')
        st.write_stream(answer_stream)

        total_latency_s = time.perf_counter() - request_started_at

        with st.expander("Answer metrics", expanded=True):
            prompt_tokens = answer_metrics.get("prompt_tokens")
            completion_tokens = answer_metrics.get("completion_tokens")
            total_tokens = answer_metrics.get("total_tokens")
            estimated_chat_cost_usd = answer_metrics.get("estimated_chat_cost_usd")
            generation_latency_s = answer_metrics.get("generation_latency_s")

            token_col1, token_col2, token_col3 = st.columns(3, gap="small")
            with token_col1:
                with st.container(border=True):
                    st.caption("Prompt tokens")
                    st.subheader(str(prompt_tokens) if prompt_tokens is not None else "N/A")

            with token_col2:
                with st.container(border=True):
                    st.caption("Completion tokens")
                    st.subheader(str(completion_tokens) if completion_tokens is not None else "N/A")

            with token_col3:
                with st.container(border=True):
                    st.caption("Total tokens")
                    st.subheader(str(total_tokens) if total_tokens is not None else "N/A")

            perf_col1, perf_col2, perf_col3 = st.columns(3, gap="small")
            with perf_col1:
                with st.container(border=True):
                    st.caption("Estimated chat cost (USD)")
                    if estimated_chat_cost_usd is None:
                        st.subheader("N/A")
                    else:
                        st.subheader(f"${estimated_chat_cost_usd:.8f}")

            with perf_col2:
                with st.container(border=True):
                    st.caption("Retrieval latency")
                    st.subheader(f"{retrieval_latency_s:.3f}s")

            with perf_col3:
                with st.container(border=True):
                    st.caption("Generation latency")
                    st.subheader(f"{generation_latency_s:.3f}s" if generation_latency_s is not None else "N/A")

            st.caption(f"Total latency: {total_latency_s:.3f}s")

        with st.expander("Sources"):
            for r in results:
                st.write(r["source"])

        with st.expander("Retrieved chunks"):
            for idx, r in enumerate(results, start=1):
                hybrid_score = r.get("score")
                vector_score = r.get("vector_score")
                bm25_score = r.get("bm25_score")
                bm25_metrics = r.get("bm25_metrics") or {}
                with st.container(border=True):
                    st.subheader(f"Chunk #{idx}")
                    st.caption(
                        f"Hybrid score: {hybrid_score:.6f}"
                        if isinstance(hybrid_score, (int, float))
                        else "Hybrid score: N/A"
                    )
                    st.caption(
                        f"Vector similarity: {vector_score:.6f}"
                        if isinstance(vector_score, (int, float))
                        else "Vector similarity: N/A"
                    )
                    st.caption(
                        f"BM25 score: {bm25_score:.6f}"
                        if isinstance(bm25_score, (int, float))
                        else "BM25 score: N/A"
                    )

                    if bm25_metrics:
                        tf_sum = bm25_metrics.get("tf_sum")
                        idf_sum = bm25_metrics.get("idf_sum")
                        matched_terms_count = bm25_metrics.get("matched_terms_count")
                        bm25_term_score_sum = bm25_metrics.get("bm25_term_score_sum")
                        st.caption(
                            "BM25 metrics: "
                            f"matched_terms={matched_terms_count}, "
                            f"tf_sum={tf_sum}, "
                            f"idf_sum={idf_sum}, "
                            f"term_score_sum={bm25_term_score_sum}"
                        )

                        with st.expander("BM25 term details", expanded=False):
                            st.write(bm25_metrics.get("per_term", []))

                    source = r.get("source", "")
                    chunk_preview = r.get("text", "")[:300]
                    st.code(chunk_preview, language=infer_code_language(source))

