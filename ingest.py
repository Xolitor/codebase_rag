import os
import json
from qdrant_client import QdrantClient
from config import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PRICE_PER_1M_TOKENS,
)
from openai import OpenAI

client = OpenAI()
INGEST_STATS_PATH = os.path.join(os.path.dirname(__file__), "ingest_stats.json")

#LOAD
def load_files(path):
    docs = []
    for root,_,files in os.walk(path):
        for f in files:
            if f.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rb", ".php", ".html", ".css")):
                with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as file:
                    docs.append({
                        "text": file.read(),
                        "source": f,
                    })
    return docs

#CHUNK
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
    return chunks

#EMBED
def estimate_embedding_cost(tokens_used):
    return (tokens_used / 1_000_000) * EMBEDDING_PRICE_PER_1M_TOKENS


def embed(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    data = response.data[0]
    usage = getattr(response, "usage", None)
    tokens_used = getattr(usage, "total_tokens", None)
    if tokens_used is None:
        tokens_used = getattr(usage, "prompt_tokens", 0) if usage else 0

    cost_usd = estimate_embedding_cost(tokens_used)

    print(
        f"[embed] model={EMBEDDING_MODEL} tokens={tokens_used} "
        f"cost_usd={cost_usd:.8f} dims={len(data.embedding)}"
    )
    print(f"[embed] embedding_preview={data.embedding[:8]}")

    return {
        "vector": data.embedding,
        "tokens_used": tokens_used,
        "cost_usd": cost_usd,
    }

#STORE
qdrant = QdrantClient(
    host="localhost",
    port=6333
)


def save_last_ingest_stats(stats):
    try:
        with open(INGEST_STATS_PATH, "w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2)
    except Exception as exc:
        print(f"[ingest] failed to persist stats: {exc}")


def load_last_ingest_stats():
    try:
        if not os.path.exists(INGEST_STATS_PATH):
            return None

        with open(INGEST_STATS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print(f"[ingest] failed to load persisted stats: {exc}")
        return None

def init_collection():
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": VECTOR_SIZE, "distance": "Cosine"}
    )

def ingest_codebase(path):
    docs = load_files(path)
    points = []
    idx = 0
    total_tokens_used = 0
    total_embedding_cost_usd = 0.0

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            embedding_result = embed(chunk)
            vector = embedding_result["vector"]
            chunk_tokens = embedding_result["tokens_used"]
            chunk_cost_usd = embedding_result["cost_usd"]

            total_tokens_used += chunk_tokens
            total_embedding_cost_usd += chunk_cost_usd

            points.append({
                "id": idx,
                "vector": vector,
                "payload": {
                    "text": chunk,
                    "source": doc["source"],
                    "embedding_tokens": chunk_tokens,
                    "embedding_cost_usd": chunk_cost_usd,
                }
            })
            idx += 1

    if not points:
        stats = {
            "chunks_indexed": 0,
            "total_tokens_used": 0,
            "avg_tokens_per_chunk": 0.0,
            "total_embedding_cost_usd": 0.0,
        }
        save_last_ingest_stats(stats)
        return stats

    qdrant.upsert(
        collection_name = COLLECTION_NAME,
        points = points
    )

    chunks_indexed = len(points)
    avg_tokens_per_chunk = total_tokens_used / chunks_indexed

    print(
        f"[ingest] chunks={chunks_indexed} total_tokens={total_tokens_used} "
        f"avg_tokens_per_chunk={avg_tokens_per_chunk:.2f} "
        f"total_embedding_cost_usd={total_embedding_cost_usd:.8f}"
    )

    stats = {
        "chunks_indexed": chunks_indexed,
        "total_tokens_used": total_tokens_used,
        "avg_tokens_per_chunk": avg_tokens_per_chunk,
        "total_embedding_cost_usd": total_embedding_cost_usd,
    }
    save_last_ingest_stats(stats)
    return stats

