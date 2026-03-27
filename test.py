from rag import search
from config import TOP_K

TEST_QUERIES = [
    # 1. Direct lookup (easy retrieval)
    "Where is the embedding model defined?",

    # 2. Function-level understanding (tests chunking)
    "What does the sanitize function do?",

    # 3. Cross-file reasoning (tests retrieval depth)
    "How does the search pipeline work from query to answer?",

    # 4. Keyword-sensitive (tests hybrid search)
    "Where is TOP_K used?",

    # 5. Vague semantic query (tests embeddings)
    "How does the system prevent malicious input?"
]

for q in TEST_QUERIES:
    results = search(q)
    print("\nQUESTION:", q)
    for r in results:
        print("-", r["file"], "| score:", r["score"])