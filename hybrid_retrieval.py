
import re
from rank_bm25 import BM25Okapi

bm25 = None
bm25_corpus = []
bm25_chunks = []


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def add_to_bm25(chunk, source, point_id=None):
    global bm25_corpus, bm25_chunks

    tokens = tokenize(chunk)
    bm25_corpus.append(tokens)

    if point_id is None:
        point_id = len(bm25_chunks)

    bm25_chunks.append({
        "id": point_id,
        "text": chunk,
        "source": source,
    })


def reset_bm25():
    global bm25, bm25_corpus, bm25_chunks
    bm25 = None
    bm25_corpus = []
    bm25_chunks = []


def build_bm25():
    global bm25
    if bm25_corpus:
        bm25 = BM25Okapi(bm25_corpus)


def get_bm25_entries(limit_tokens=30):
    entries = []
    for i, chunk in enumerate(bm25_chunks):
        tokens = bm25_corpus[i] if i < len(bm25_corpus) else tokenize(chunk.get("text", ""))
        entries.append({
            "id": chunk.get("id", i),
            "source": chunk.get("source", ""),
            "text_preview": chunk.get("text", "")[:300],
            "token_count": len(tokens),
            "tokens_preview": tokens[:limit_tokens],
        })
    return entries