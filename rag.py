from ingest import embed, qdrant
from config import (
    COLLECTION_NAME,
    TOP_K,
    LLM_MODEL,
    LLM_INPUT_PRICE_PER_1M_TOKENS,
    LLM_OUTPUT_PRICE_PER_1M_TOKENS,
)
from openai import OpenAI
import time


def search(query):
    q_emb_result = embed(query)
    q_vector = q_emb_result["vector"]

    # qdrant-client >= 1.10+ uses query_points for vector similarity search.
    if hasattr(qdrant, "query_points"):
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vector,
            limit=TOP_K,
        )
        points = getattr(response, "points", [])
    # else:
    #     # Backward compatibility with older qdrant-client versions.
    #     points = qdrant.search(
    #         collection_name=COLLECTION_NAME,
    #         query_vector=q_vector,
    #         limit=TOP_K,
    #     )

    results = []
    for p in points:
        payload = dict(getattr(p, "payload", {}) or {})
        payload["score"] = getattr(p, "score", None)
        results.append(payload)

    return results


client = OpenAI()


def estimate_chat_cost(prompt_tokens, completion_tokens):
    if prompt_tokens is None or completion_tokens is None:
        return None

    input_cost = (prompt_tokens / 1_000_000) * LLM_INPUT_PRICE_PER_1M_TOKENS
    output_cost = (completion_tokens / 1_000_000) * LLM_OUTPUT_PRICE_PER_1M_TOKENS
    return input_cost + output_cost

# def generate_answer(query, context_chunk):
#     context = "\n\n".join([c.get("text", "") for c in context_chunk])
#     prompt = f"""
#         Answer the question based on the context below.
#         Mention file names when relevant.

#         Context:
#         {context}

#         Question:
#         {query}
#         """
    
#     response = client.chat.completions.create(
#         model = LLM_MODEL,
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return response.choices[0].message.content


def generate_answer_stream(query, context_chunk, metrics=None):
    context = "\n\n".join([c.get("text", "") for c in context_chunk])
    prompt = f"""
        Answer the question based on the context below.
        Mention file names when relevant.

        Context:
        {context}

        Question:
        {query}
        """

    if metrics is None:
        metrics = {}

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    generation_started_at = time.perf_counter()

    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
        )
    except TypeError:
        # Backward compatibility if include_usage is unavailable in stream options.
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

    try:
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
                completion_tokens = getattr(usage, "completion_tokens", completion_tokens)
                total_tokens = getattr(usage, "total_tokens", total_tokens)

            choices = getattr(chunk, "choices", [])
            if not choices:
                continue

            delta = choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content
    finally:
        generation_latency_s = time.perf_counter() - generation_started_at
        metrics["prompt_tokens"] = prompt_tokens
        metrics["completion_tokens"] = completion_tokens
        metrics["total_tokens"] = total_tokens
        metrics["estimated_chat_cost_usd"] = estimate_chat_cost(prompt_tokens, completion_tokens)
        metrics["generation_latency_s"] = generation_latency_s