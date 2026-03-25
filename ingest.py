import os
import json
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PRICE_PER_1M_TOKENS,
    MODE
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

def load_files_from_github(url):
    docs = []
    allowed_exts = (".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rb", ".php", ".html", ".css")
    print(f"[ingest] loading files from GitHub URL: {url}")
    started_at = time.perf_counter()
    try:
        print("[ingest][github] step=1 parse_url")
        parsed = urlparse(url)
        if parsed.netloc.lower() != "github.com":
            print(f"[ingest] invalid GitHub URL: {url}")
            return docs

        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) < 2:
            print(f"[ingest] invalid GitHub repo URL: {url}")
            return docs

        owner, repo = parts[0], parts[1].replace(".git", "")
        print(f"[ingest][github] owner={owner} repo={repo}")

        def get_json(api_url):
            print(f"[ingest][github] GET {api_url}")
            req = Request(
                api_url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "codebase-rag-ingest",
                },
            )
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))

        def get_text(raw_url):
            req = Request(raw_url, headers={"User-Agent": "codebase-rag-ingest"})
            with urlopen(req, timeout=30) as response:
                return response.read().decode("utf-8", errors="ignore")

        print("[ingest][github] step=2 fetch_repo_metadata")
        repo_meta = get_json(f"https://api.github.com/repos/{owner}/{repo}")
        default_branch = repo_meta.get("default_branch", "main")
        print(f"[ingest][github] default_branch={default_branch}")

        print("[ingest][github] step=3 fetch_repo_tree")
        tree = get_json(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
        )
        tree_nodes = tree.get("tree", [])
        print(f"[ingest][github] tree_nodes={len(tree_nodes)}")

        print("[ingest][github] step=4 filter_blob_code_files")
        candidate_paths = []
        for node in tree_nodes:
            if node.get("type") != "blob":
                continue
            file_path = node.get("path", "")
            if file_path.endswith(allowed_exts):
                candidate_paths.append(file_path)

        print(f"[ingest][github] candidate_code_files={len(candidate_paths)}")
        if candidate_paths:
            print(f"[ingest][github] first_candidate={candidate_paths[0]}")

        print("[ingest][github] step=5 download_candidate_files")
        failed_files = 0
        for i, file_path in enumerate(candidate_paths, start=1):
            if i == 1 or i % 25 == 0:
                print(f"[ingest][github] downloading {i}/{len(candidate_paths)}: {file_path}")

            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{file_path}"
            try:
                docs.append({
                    "text": get_text(raw_url),
                    "source": file_path,
                })
            except Exception as exc:
                failed_files += 1
                print(f"[ingest] failed to fetch {file_path}: {exc}")

        elapsed = time.perf_counter() - started_at
        print(
            f"[ingest] loaded {len(docs)} files from GitHub repo {owner}/{repo} "
            f"(failed={failed_files}, elapsed_s={elapsed:.2f})"
        )
        return docs
    

    except Exception as exc:
        print(f"[ingest] failed to load files from GitHub URL '{url}': {exc}")
        return docs


def load_files_drag_and_drop(uploaded_files):
    docs = []
    allowed_exts = (".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rb", ".php", ".html", ".css")

    if not uploaded_files:
        print("[ingest][upload] no uploaded files")
        return docs

    print(f"[ingest][upload] received_files={len(uploaded_files)}")
    skipped = 0

    for uploaded_file in uploaded_files:
        file_name = getattr(uploaded_file, "name", "")
        if not file_name.endswith(allowed_exts):
            skipped += 1
            print(f"[ingest][upload] skipped unsupported file: {file_name}")
            continue

        try:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            docs.append({
                "text": content,
                "source": file_name,
            })
        except Exception as exc:
            print(f"[ingest][upload] failed to read {file_name}: {exc}")

    print(f"[ingest][upload] loaded={len(docs)} skipped={skipped}")
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
if MODE == "demo":
    qdrant = QdrantClient(":memory:")
else:
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

def _ingest_docs(docs):
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

            points.append(models.PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "text": chunk,
                    "source": doc["source"],
                    "embedding_tokens": chunk_tokens,
                    "embedding_cost_usd": chunk_cost_usd,
                },
            ))
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


def ingest_codebase(path):
    docs = load_files(path)
    return _ingest_docs(docs)

def ingest_codebase_from_github(url):
    docs = load_files_from_github(url)
    if not docs:
        print(f"[ingest] no documents loaded from GitHub URL: {url}")
        return ingest_codebase("./repo")

    return _ingest_docs(docs)


def ingest_codebase_from_uploads(uploaded_files):
    docs = load_files_drag_and_drop(uploaded_files)
    return _ingest_docs(docs)

