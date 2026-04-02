import json
import os
import streamlit as st
from ingest import load_last_chunking_strategies

st.set_page_config(page_title="Benchmark Results", page_icon="🏁")
st.markdown("# Benchmark Results")
st.write(
    "This benchmark was run using the `repo` folder from the example codebase and 5 test queries "
    "designed to evaluate different metrics across key RAG architecture features."
)
st.caption("Latest benchmark and ingestion metrics captured by this app.")

CHUNKING_STRATEGIES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "chunking_strategies.json"
)
HYBRID_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "hybrid_search_results.json"
)
VECTOR_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "vector_search_results.json"
)


def load_json_file(path):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def avg(values):
    valid = [v for v in values if isinstance(v, (int, float))]
    if not valid:
        return None
    return sum(valid) / len(valid)


def summarize_query_metrics(results):
    queries = (results or {}).get("queries", [])
    return {
        "avg_prompt_tokens": avg([q.get("prompt_tokens") for q in queries]),
        "avg_completion_tokens": avg([q.get("completion_tokens") for q in queries]),
        "avg_total_tokens": avg([q.get("total_tokens") for q in queries]),
        "avg_estimated_chat_cost_usd": avg([q.get("estimated_chat_cost_usd") for q in queries]),
        "avg_retrieval_latency_s": avg([q.get("retrieval_latency_s") for q in queries]),
        "avg_generation_latency_s": avg([q.get("generation_latency_s") for q in queries]),
    }


def summarize_chunk_metrics(results):
    queries = (results or {}).get("queries", [])
    all_chunks = []
    top1_chunks = []

    for q in queries:
        chunks = q.get("chunks_used", []) or []
        all_chunks.extend(chunks)
        if chunks:
            top1_chunks.append(chunks[0])

    return {
        "avg_vector_similarity_all": avg([c.get("vector_similarity") for c in all_chunks]),
        "avg_hybrid_score_all": avg([c.get("hybrid_score") for c in all_chunks]),
        "avg_bm25_score_all": avg([c.get("bm25_score") for c in all_chunks]),
        "avg_vector_similarity_top1": avg([c.get("vector_similarity") for c in top1_chunks]),
        "avg_hybrid_score_top1": avg([c.get("hybrid_score") for c in top1_chunks]),
        "avg_bm25_score_top1": avg([c.get("bm25_score") for c in top1_chunks]),
    }


def fmt_int(value):
    if value is None:
        return "N/A"
    return f"{int(round(value))}"


def fmt_float(value, digits=4):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def fmt_usd(value):
    if value is None:
        return "N/A"
    return f"${value:.8f}"


def safe_ratio(numerator, denominator):
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if denominator == 0:
        return None
    return numerator / denominator

chunking_strategy_stats = load_last_chunking_strategies() or load_json_file(CHUNKING_STRATEGIES_PATH)
hybrid_search_results = load_json_file(HYBRID_RESULTS_PATH)
vector_search_results = load_json_file(VECTOR_RESULTS_PATH)

st.write("## Chunking Strategy Benchmark")
if chunking_strategy_stats:
    regular = chunking_strategy_stats.get("regular_chunking", {})
    syntax = chunking_strategy_stats.get("syntax_chunking", {})

    cmp_col1, cmp_col2 = st.columns(2, gap="medium")

    with cmp_col1:
        with st.container(border=True):
            st.subheader("Regular Chunking")
            st.caption("Strategy: chunk_text")
            st.write(f"Chunks generated: {int(regular.get('chunks_generated', 0))}")
            st.write(f"Total tokens: {int(regular.get('total_tokens_used', 0))}")
            st.write(f"Avg tokens / chunk: {float(regular.get('avg_tokens_per_chunk', 0.0)):.2f}")
            st.write(
                f"Embedding cost (USD): ${float(regular.get('total_embedding_cost_usd', 0.0)):.8f}"
            )

    with cmp_col2:
        with st.container(border=True):
            st.subheader("Syntax/Code Chunking")
            st.caption("Strategy: chunk_code_by_language")
            st.write(f"Chunks generated: {int(syntax.get('chunks_generated', 0))}")
            st.write(f"Total tokens: {int(syntax.get('total_tokens_used', 0))}")
            st.write(f"Avg tokens / chunk: {float(syntax.get('avg_tokens_per_chunk', 0.0)):.2f}")
            st.write(
                f"Embedding cost (USD): ${float(syntax.get('total_embedding_cost_usd', 0.0)):.8f}"
            )
            if bool(syntax.get("used_for_rag", False)):
                st.success("This is the strategy currently used for RAG indexing.")

    # with st.expander("Raw chunking_strategies.json", expanded=False):
    #     st.json(chunking_strategy_stats)
else:
    st.info("No chunking strategy benchmark data available yet.")

st.write("### Summary of chunking strategies comparison")
st.write("For chunking strategy, the metrics indicate that syntax/code-aware chunking is more efficient and more retrieval-friendly on this repo snapshot: it produced more chunks (7 vs 4), but with smaller chunks on average (31.43 vs 58.5 tokens), and still used slightly fewer total embedding tokens (220 vs 234) and lower total embedding cost ($0.00000440 vs $0.00000468). That combination is usually desirable for RAG because you get finer-grained context boundaries without increasing embedding spend, so the current choice to keep syntax-aware chunks for indexing is well supported by the data.")

st.write("## Chunking Strategy Benchmark")
st.write("## Search Strategy Benchmark")
if not hybrid_search_results and not vector_search_results:
    st.info("No search strategy benchmark data available yet.")
else:
    hybrid_summary = summarize_query_metrics(hybrid_search_results)
    vector_summary = summarize_query_metrics(vector_search_results)
    hybrid_chunk_summary = summarize_chunk_metrics(hybrid_search_results)
    vector_chunk_summary = summarize_chunk_metrics(vector_search_results)

    st.caption("Average metrics across the 5 benchmark queries")

    query_summary_rows = [
        {
            "metric": "Prompt tokens",
            "hybrid": fmt_int(hybrid_summary.get("avg_prompt_tokens")),
            "vector_only": fmt_int(vector_summary.get("avg_prompt_tokens")),
        },
        {
            "metric": "Completion tokens",
            "hybrid": fmt_int(hybrid_summary.get("avg_completion_tokens")),
            "vector_only": fmt_int(vector_summary.get("avg_completion_tokens")),
        },
        {
            "metric": "Total tokens",
            "hybrid": fmt_int(hybrid_summary.get("avg_total_tokens")),
            "vector_only": fmt_int(vector_summary.get("avg_total_tokens")),
        },
        {
            "metric": "Estimated chat cost (USD)",
            "hybrid": fmt_usd(hybrid_summary.get("avg_estimated_chat_cost_usd")),
            "vector_only": fmt_usd(vector_summary.get("avg_estimated_chat_cost_usd")),
        },
        {
            "metric": "Retrieval latency (s)",
            "hybrid": fmt_float(hybrid_summary.get("avg_retrieval_latency_s"), 3),
            "vector_only": fmt_float(vector_summary.get("avg_retrieval_latency_s"), 3),
        },
        {
            "metric": "Generation latency (s)",
            "hybrid": fmt_float(hybrid_summary.get("avg_generation_latency_s"), 3),
            "vector_only": fmt_float(vector_summary.get("avg_generation_latency_s"), 3),
        },
    ]
    st.dataframe(query_summary_rows, use_container_width=True, hide_index=True)

    st.caption("Chunk score comparison")
    chunk_summary_rows = [
        {
            "metric": "Avg vector similarity (all retrieved chunks)",
            "hybrid": fmt_float(hybrid_chunk_summary.get("avg_vector_similarity_all"), 6),
            "vector_only": fmt_float(vector_chunk_summary.get("avg_vector_similarity_all"), 6),
        },
        {
            "metric": "Avg hybrid score (all retrieved chunks)",
            "hybrid": fmt_float(hybrid_chunk_summary.get("avg_hybrid_score_all"), 6),
            "vector_only": fmt_float(vector_chunk_summary.get("avg_hybrid_score_all"), 6),
        },
        {
            "metric": "Avg BM25 score (all retrieved chunks)",
            "hybrid": fmt_float(hybrid_chunk_summary.get("avg_bm25_score_all"), 6),
            "vector_only": fmt_float(vector_chunk_summary.get("avg_bm25_score_all"), 6),
        },
        {
            "metric": "Avg vector similarity (top-1 chunk/query)",
            "hybrid": fmt_float(hybrid_chunk_summary.get("avg_vector_similarity_top1"), 6),
            "vector_only": fmt_float(vector_chunk_summary.get("avg_vector_similarity_top1"), 6),
        },
        {
            "metric": "Avg hybrid score (top-1 chunk/query)",
            "hybrid": fmt_float(hybrid_chunk_summary.get("avg_hybrid_score_top1"), 6),
            "vector_only": fmt_float(vector_chunk_summary.get("avg_hybrid_score_top1"), 6),
        },
        {
            "metric": "Avg BM25 score (top-1 chunk/query)",
            "hybrid": fmt_float(hybrid_chunk_summary.get("avg_bm25_score_top1"), 6),
            "vector_only": fmt_float(vector_chunk_summary.get("avg_bm25_score_top1"), 6),
        },
    ]
    st.dataframe(chunk_summary_rows, use_container_width=True, hide_index=True)



    hybrid_queries = (hybrid_search_results or {}).get("queries", [])
    vector_queries = (vector_search_results or {}).get("queries", [])
    by_query = {}

    for entry in hybrid_queries:
        by_query.setdefault(entry.get("query"), {})["hybrid"] = entry
    for entry in vector_queries:
        by_query.setdefault(entry.get("query"), {})["vector"] = entry

    per_query_rows = []
    for query_text, pair in by_query.items():
        h = pair.get("hybrid", {})
        v = pair.get("vector", {})
        per_query_rows.append(
            {
                "query": query_text,
                "hybrid_retrieval_s": fmt_float(h.get("retrieval_latency_s"), 3),
                "vector_retrieval_s": fmt_float(v.get("retrieval_latency_s"), 3),
                "hybrid_generation_s": fmt_float(h.get("generation_latency_s"), 3),
                "vector_generation_s": fmt_float(v.get("generation_latency_s"), 3),
                "hybrid_total_tokens": fmt_int(h.get("total_tokens")),
                "vector_total_tokens": fmt_int(v.get("total_tokens")),
                "hybrid_cost_usd": fmt_usd(h.get("estimated_chat_cost_usd")),
                "vector_cost_usd": fmt_usd(v.get("estimated_chat_cost_usd")),
            }
        )

    if per_query_rows:
        st.caption("Per-query side-by-side metrics")
        st.dataframe(per_query_rows, use_container_width=True, hide_index=True)

    # with st.expander("Raw hybrid_search_results.json", expanded=False):
    #     st.json(hybrid_search_results)

    # with st.expander("Raw vector_search_results.json", expanded=False):
    #     st.json(vector_search_results)

st.write("### Summary of search strategies comparison")
st.write("In this benchmark on 5 queries against the repo snapshot, Hybrid shows stronger top-rank concentration than vector-only: hybrid top-1 vs average = 1.197746 vs 0.340179 (x3.52), while vector top-1 vs average = 0.332302 vs 0.142390 (x2.33). This suggests hybrid ranking emphasizes the best-matching chunk more strongly. The two approaches are very close in generation-side cost/usage, but vector-only is clearly faster on retrieval latency in this benchmark run: average retrieval latency is about 0.182s for vector-only vs 0.295s for hybrid")