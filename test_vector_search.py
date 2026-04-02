import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ingest import ingest_codebase, init_collection
from rag import generate_answer_stream, search


QUERIES = [
    "is there any class structure used in the code base (with the __init__ METHOD) ?",
    "Trace user data flow across files: where is a user created and where is a user fetched, including language differences.",
    "Which function validates credentials, what are the accepted values, and what token is returned on success?",
    "Explain how connection state is managed in the database component before and after connect/disconnect calls.",
    "Compare utility helpers that format payloads versus compute arithmetic; include expected output structure.",
]


def run_vector_search_benchmark():
    project_root = Path(__file__).resolve().parent
    repo_path = project_root / "repo"
    output_path = project_root / "vector_search_results.json"

    if not repo_path.exists() or not repo_path.is_dir():
        raise FileNotFoundError(f"Expected repo folder at: {repo_path}")

    print("[benchmark-vector] re-initializing vector collection and ingesting repo/")
    init_collection()
    ingest_stats = ingest_codebase(str(repo_path))

    results_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "repo",
        "strategy": "vector_only",
        "ingest_stats": ingest_stats,
        "queries": [],
    }

    for index, query in enumerate(QUERIES, start=1):
        print(f"[benchmark-vector] ({index}/{len(QUERIES)}) query: {query}")

        retrieval_started_at = time.perf_counter()
        retrieved_chunks = search(query, use_hybrid=False)
        retrieval_latency_s = time.perf_counter() - retrieval_started_at

        answer_metrics = {}
        answer_text = "".join(generate_answer_stream(query, retrieved_chunks, answer_metrics))

        chunk_rows = []
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            chunk_rows.append(
                {
                    "rank": rank,
                    "id": chunk.get("id"),
                    "source": chunk.get("source"),
                    "hybrid_score": None,
                    "vector_similarity": chunk.get("vector_score", chunk.get("score")),
                    "bm25_score": None,
                }
            )

        results_payload["queries"].append(
            {
                "query": query,
                "answer": answer_text,
                "prompt_tokens": answer_metrics.get("prompt_tokens"),
                "completion_tokens": answer_metrics.get("completion_tokens"),
                "total_tokens": answer_metrics.get("total_tokens"),
                "estimated_chat_cost_usd": answer_metrics.get("estimated_chat_cost_usd"),
                "retrieval_latency_s": retrieval_latency_s,
                "generation_latency_s": answer_metrics.get("generation_latency_s"),
                "chunks_used": chunk_rows,
            }
        )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results_payload, file, indent=2)

    print(f"[benchmark-vector] results written to {output_path}")


if __name__ == "__main__":
    run_vector_search_benchmark()
