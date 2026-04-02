import os
import json
import re
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from qdrant_client import QdrantClient
from qdrant_client.http import models
import chunk
from config import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PRICE_PER_1M_TOKENS,
    MODE,
    MAX_INGEST_FILES,
)
from openai import OpenAI
from chunk import chunk_code_by_language, chunk_text
from hybrid_retrieval import add_to_bm25, build_bm25, reset_bm25, get_bm25_entries

client = OpenAI()
INGEST_STATS_PATH = os.path.join(os.path.dirname(__file__), "ingest_stats.json")
INGEST_BM25_PATH = os.path.join(os.path.dirname(__file__), "ingest_bm25.json")
CHUNKING_STRATEGIES_PATH = os.path.join(os.path.dirname(__file__), "chunking_strategies.json")

#LOAD
def infer_code_language(source):
    source = (source or "").lower()
    if source.endswith(".py"):
        return "python"
    if source.endswith(".js"):
        return "javascript"
    if source.endswith(".ts"):
        return "typescript"
    if source.endswith(".java"):
        return "java"
    if source.endswith(".cpp"):
        return "cpp"
    if source.endswith(".c"):
        return "c"
    if source.endswith(".go"):
        return "go"
    if source.endswith(".rb"):
        return "ruby"
    if source.endswith(".php"):
        return "php"
    if source.endswith(".html"):
        return "html"
    if source.endswith(".css"):
        return "css"
    return "text"


def load_files(path):
    docs = []
    capped = False
    for root,_,files in os.walk(path):
        for f in files:
            if f.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rb", ".php", ".html", ".css")):
                with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as file:
                    docs.append({
                        "text": file.read(),
                        "source": f,
                    })
                if len(docs) >= MAX_INGEST_FILES:
                    capped = True
                    break
        if capped:
            break

    if capped:
        print(f"[ingest] capped local ingest to first {MAX_INGEST_FILES} supported files")
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

        if len(candidate_paths) > MAX_INGEST_FILES:
            print(
                f"[ingest][github] capping files from {len(candidate_paths)} "
                f"to {MAX_INGEST_FILES}"
            )
        candidate_paths = candidate_paths[:MAX_INGEST_FILES]

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
        if len(docs) >= MAX_INGEST_FILES:
            print(f"[ingest][upload] capped uploaded ingest to first {MAX_INGEST_FILES} supported files")
            break

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


def save_last_bm25_stats(stats):
    try:
        with open(INGEST_BM25_PATH, "w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2)
    except Exception as exc:
        print(f"[ingest] failed to persist bm25 stats: {exc}")


def load_last_bm25_stats():
    try:
        if not os.path.exists(INGEST_BM25_PATH):
            return None

        with open(INGEST_BM25_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print(f"[ingest] failed to load persisted bm25 stats: {exc}")
        return None


def save_last_chunking_strategies(stats, overwrite=False):
    try:
        if os.path.exists(CHUNKING_STRATEGIES_PATH) and not overwrite:
            print(
                "[ingest] chunking_strategies.json already exists; "
                "keeping existing benchmark snapshot"
            )
            return
        with open(CHUNKING_STRATEGIES_PATH, "w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2)
    except Exception as exc:
        print(f"[ingest] failed to persist chunking strategy stats: {exc}")


def load_last_chunking_strategies():
    try:
        if not os.path.exists(CHUNKING_STRATEGIES_PATH):
            return None

        with open(CHUNKING_STRATEGIES_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print(f"[ingest] failed to load chunking strategy stats: {exc}")
        return None

def init_collection():
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": VECTOR_SIZE, "distance": "Cosine"}
    )
    reset_bm25()

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def _ingest_docs(docs):
    points = []
    idx = 0
    total_tokens_used = 0
    total_embedding_cost_usd = 0.0
    regular_total_tokens_used = 0
    regular_total_embedding_cost_usd = 0.0
    regular_chunks_generated = 0
    syntax_chunks_generated = 0

    for doc in docs:
        regular_chunks = chunk_text(doc["text"])
        syntax_chunks = chunk_code_by_language(doc["text"], doc["source"])

        regular_chunks_generated += len(regular_chunks)
        syntax_chunks_generated += len(syntax_chunks)

        for regular_chunk in regular_chunks:
            regular_embedding_result = embed(regular_chunk)
            regular_total_tokens_used += regular_embedding_result["tokens_used"]
            regular_total_embedding_cost_usd += regular_embedding_result["cost_usd"]

        for chunk in syntax_chunks:
            #BM25 CORPUS PREPARATION
            add_to_bm25(chunk, doc["source"], point_id=idx)

            #REGULAR EMBEDDING
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
        save_last_bm25_stats({
            "tokenizer": r"re.findall(r\"\\w+\", text.lower())",
            "chunks_indexed": 0,
            "entries": [],
        })
        save_last_chunking_strategies({
            "regular_chunking": {
                "strategy": "chunk_text",
                "chunks_generated": 0,
                "total_tokens_used": 0,
                "avg_tokens_per_chunk": 0.0,
                "total_embedding_cost_usd": 0.0,
            },
            "syntax_chunking": {
                "strategy": "chunk_code_by_language",
                "chunks_generated": 0,
                "total_tokens_used": 0,
                "avg_tokens_per_chunk": 0.0,
                "total_embedding_cost_usd": 0.0,
                "used_for_rag": True,
            },
        })
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
    build_bm25()

    bm25_entries = get_bm25_entries(limit_tokens=30)
    save_last_bm25_stats({
        "tokenizer": r"re.findall(r\"\\w+\", text.lower())",
        "chunks_indexed": chunks_indexed,
        "entries": bm25_entries,
    })

    regular_avg_tokens_per_chunk = (
        regular_total_tokens_used / regular_chunks_generated if regular_chunks_generated else 0.0
    )
    syntax_avg_tokens_per_chunk = (
        total_tokens_used / syntax_chunks_generated if syntax_chunks_generated else 0.0
    )

    save_last_chunking_strategies({
        "regular_chunking": {
            "strategy": "chunk_text",
            "chunks_generated": regular_chunks_generated,
            "total_tokens_used": regular_total_tokens_used,
            "avg_tokens_per_chunk": regular_avg_tokens_per_chunk,
            "total_embedding_cost_usd": regular_total_embedding_cost_usd,
        },
        "syntax_chunking": {
            "strategy": "chunk_code_by_language",
            "chunks_generated": syntax_chunks_generated,
            "total_tokens_used": total_tokens_used,
            "avg_tokens_per_chunk": syntax_avg_tokens_per_chunk,
            "total_embedding_cost_usd": total_embedding_cost_usd,
            "used_for_rag": True,
        },
    })

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

