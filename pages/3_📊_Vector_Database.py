import streamlit as st
from ingest import qdrant, load_last_ingest_stats
from config import COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_SIZE

st.set_page_config(page_title="Vector Database", page_icon="📊")

st.markdown("# Vector Database")
# st.sidebar.header("Vector Database ")


def get_similarity_method(collection_info):
    try:
        vectors_cfg = collection_info.config.params.vectors

        if hasattr(vectors_cfg, "distance"):
            return str(vectors_cfg.distance)

        if isinstance(vectors_cfg, dict):
            if "distance" in vectors_cfg:
                return str(vectors_cfg["distance"])

            if vectors_cfg:
                first_cfg = next(iter(vectors_cfg.values()))
                if hasattr(first_cfg, "distance"):
                    return str(first_cfg.distance)
                if isinstance(first_cfg, dict) and "distance" in first_cfg:
                    return str(first_cfg["distance"])
    except Exception:
        pass

    return "Cosine"


def get_vector_metrics():
    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        count_response = qdrant.count(collection_name=COLLECTION_NAME, exact=True)

        sample_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            with_vectors=True,
        )

        if sample_points and sample_points[0].vector is not None:
            vector_size = len(sample_points[0].vector)
        else:
            vector_size = VECTOR_SIZE

        return {
            "vector_size": vector_size,
            "similarity": get_similarity_method(collection_info),
            "embedding_model": EMBEDDING_MODEL,
            "chunk_count": count_response.count,
        }
    except Exception:
        return None


def render_metric_cards(metrics):
    row1_col1, row1_col2 = st.columns(2, gap="small")
    row2_col1, row2_col2 = st.columns(2, gap="small")

    with row1_col1:
        with st.container(border=True):
            st.caption("🟦 Vector Size")
            st.subheader(str(metrics["vector_size"]))

    with row1_col2:
        with st.container(border=True):
            st.caption("🟩 Similarity")
            st.subheader(str(metrics["similarity"]))

    with row2_col1:
        with st.container(border=True):
            st.caption("🟨 Embedding Model")
            st.subheader(str(metrics["embedding_model"]))

    with row2_col2:
        with st.container(border=True):
            st.caption("🟥 Chunks Generated")
            st.subheader(str(metrics["chunk_count"]))


def render_ingest_stats():
    ingest_stats = load_last_ingest_stats()
    if not ingest_stats:
        return

    total_tokens_used = ingest_stats.get("total_tokens_used", 0)
    avg_tokens_per_chunk = ingest_stats.get("avg_tokens_per_chunk", 0.0)
    total_embedding_cost_usd = ingest_stats.get("total_embedding_cost_usd", 0.0)

    st.caption("Ingestion Usage Summary")
    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        with st.container(border=True):
            st.caption("🟪 Total Tokens")
            st.subheader(f"{total_tokens_used}")

    with col2:
        with st.container(border=True):
            st.caption("🟧 Avg Tokens / Chunk")
            st.subheader(f"{avg_tokens_per_chunk:.2f}")

    with col3:
        with st.container(border=True):
            st.caption("🟫 Total Embedding Cost (USD)")
            st.subheader(f"${total_embedding_cost_usd:.8f}")

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
else:
    metrics = get_vector_metrics()
    if metrics:
        render_metric_cards(metrics)
        render_ingest_stats()

    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=5,
        with_vectors=True,
    )

    if not points:
        st.info("Ingest codebase first.")
    else:
        for p in points:
            with st.container(border=True):
                st.subheader(f"Chunk {p.id}")

                with st.expander("Text Preview", expanded=False):
                    chunk_text = p.payload.get("text", "")
                    st.write(chunk_text[:800])
                    st.caption(f"Characters in chunk: {len(chunk_text)}")

                with st.expander("Source", expanded=False):
                    st.write(p.payload.get("source", "Unknown source"))

                with st.expander("Vector Preview (first 5 dimensions)", expanded=False):
                    vector_preview = p.vector[:5] if p.vector is not None else []
                    st.write(vector_preview)

                with st.expander("Embedding Usage", expanded=False):
                    chunk_tokens = p.payload.get("embedding_tokens")
                    chunk_cost = p.payload.get("embedding_cost_usd")
                    st.write(f"Tokens used: {chunk_tokens if chunk_tokens is not None else 'N/A'}")
                    if chunk_cost is not None:
                        st.write(f"Estimated cost (USD): ${chunk_cost:.8f}")
                    else:
                        st.write("Estimated cost (USD): N/A")

                # with st.expander("Payload", expanded=False):
                #     st.json(p.payload)
