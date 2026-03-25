import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

EMBEDDING_PRICE_PER_1M_TOKENS = 0.02

COLLECTION_NAME = "codebase"
VECTOR_SIZE = 1536
TOP_K = 5