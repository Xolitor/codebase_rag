# Codebase RAG Chatbot

RAG chatbot for source code using Streamlit + OpenAI + Qdrant.

## Live Demo
- [Open Live Demo](https://codebase-rag-av.streamlit.app/)

## What This Project Does
- Ingests source files from local folders, GitHub repositories, or uploaded files.
- Uses syntax-aware/code-based chunking for supported languages and fallback text chunking for others.
- Embeds chunks with OpenAI and stores vectors in Qdrant.
- Builds a BM25 lexical index alongside vector storage for hybrid retrieval.
- Runs hybrid search by combining vector similarity scores and BM25 scores.
- Streams retrieval-augmented answers with source and chunk-level inspection.
- Applies safety nets: rate limiting, input length guard, query sanitization, and prompt hardening directives.
- Shows vector and BM25 database metrics for debugging and observability.

## Current App Structure
- [👋_Introduction.py](%F0%9F%91%8B_Introduction.py)
  - Main Streamlit page for ingestion and chat.
  - Supports GitHub URL and uploaded-file ingestion.
  - Enforces input guardrails (rate limiting and max input length).
  - Displays hybrid retrieval diagnostics (hybrid/vector/BM25 scores and BM25 term metrics).
- [pages/3_📊_Vector_Database.py](pages/3_%F0%9F%93%8A_Vector_Database.py)
  - Displays vector database metrics and ingestion usage (tokens/cost).
  - Shows vector chunk inspection with source, preview, embedding usage, and vector preview.
- [pages/4_🔎_BM25_Database.py](pages/4_%F0%9F%94%8E_BM25_Database.py)
  - Displays BM25 corpus metadata and tokenizer details.
  - Supports source filtering and chunk token/text preview inspection.
- [ingest.py](ingest.py)
  - Loads files from local path, GitHub, or uploads.
  - Performs chunking, embedding, Qdrant upsert, and BM25 corpus/index build.
  - Tracks and returns ingest stats:
    - `chunks_indexed`
    - `total_tokens_used`
    - `avg_tokens_per_chunk`
    - `total_embedding_cost_usd`
  - Persists latest stats to [ingest_stats.json](ingest_stats.json) and BM25 stats to [ingest_bm25.json](ingest_bm25.json).
- [chunk.py](chunk.py)
  - Implements syntax-aware chunking for Python and JavaScript/TypeScript.
  - Falls back to generic text chunking when syntax chunking is unavailable.
- [hybrid_retrieval.py](hybrid_retrieval.py)
  - Maintains BM25 corpus/chunk metadata.
  - Tokenizes code/text and builds the BM25 index used by hybrid search.
- [rag.py](rag.py)
  - Executes query sanitization and retrieval.
  - Performs vector search, BM25 search, and weighted hybrid merge.
  - Streams final answer generation with defensive prompt directives.
- [config.py](config.py)
  - Central config for model names, vector size, collection name, retrieval settings, and embedding price.

## Tech Stack
- Python
- Streamlit
- OpenAI API
  - Embeddings: `text-embedding-3-small`
  - Chat model: `gpt-4o-mini`
- Qdrant
- python-dotenv

## Setup
1. Create and activate a virtual environment.
2. Install dependencies.
3. Start Qdrant on `localhost:6333`. (for local testing if selected else it will be in memory)
4. Add `OPENAI_API_KEY` to `.env`.
5. Run the Streamlit app.

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
docker run -p 6333:6333 qdrant/qdrant
streamlit run "👋_Introduction.py"
```

Create `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage
1. Open the local Streamlit URL (usually `http://localhost:8501`).
2. Click **Ingest codebase** on the introduction page.
3. Ask questions about the indexed code.
4. Open **Vector Database** in the sidebar to inspect metrics and chunk-level details

## Notes
- Re-ingesting recreates the Qdrant collection and replaces previous vectors.
- Ingest stats are loaded from [ingest_stats.json](ingest_stats.json), so no full recomputation is needed on the vector page.
- If you indexed data before stats tracking was added, re-ingest once to populate latest usage stats.
