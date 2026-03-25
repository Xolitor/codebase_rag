from ingest import embed, qdrant
from config import COLLECTION_NAME, TOP_K, LLM_MODEL
from openai import OpenAI


def search(query):
    q_emb = embed(query)

    # qdrant-client >= 1.10+ uses query_points for vector similarity search.
    if hasattr(qdrant, "query_points"):
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=q_emb,
            limit=TOP_K,
        )
        points = getattr(response, "points", [])
    # else:
    #     # Backward compatibility with older qdrant-client versions.
    #     points = qdrant.search(
    #         collection_name=COLLECTION_NAME,
    #         query_vector=q_emb,
    #         limit=TOP_K,
    #     )

    return [p.payload for p in points]


client = OpenAI()

def generate_answer(query, context_chunk):
    context = "\n\n".join([c.get("text", "") for c in context_chunk])
    prompt = f"""
        Answer the question based on the context below.
        Mention file names when relevant.

        Context:
        {context}

        Question:
        {query}
        """
    
    response = client.chat.completions.create(
        model = LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def generate_answer_stream(query, context_chunk):
    context = "\n\n".join([c.get("text", "") for c in context_chunk])
    prompt = f"""
        Answer the question based on the context below.
        Mention file names when relevant.

        Context:
        {context}

        Question:
        {query}
        """

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content