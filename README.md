# Codebase RAG Chatbot

RAG chatbot for source code using Streamlit + OpenAI + Qdrant.

## Live Demo
- [Open Live Demo](https://your-live-demo-url.streamlit.app)

## What This Project Does
- Ingests source files from the local [repo](repo) folder.
- Splits files into chunks and embeds each chunk with OpenAI.
- Stores vectors in Qdrant.
- Answers user questions with retrieval-augmented generation.
- Shows vector/database metrics and per-chunk inspection in a dedicated demo page.

## Current App Structure
- [👋_Introduction.py](%F0%9F%91%8B_Introduction.py)
  - Main Streamlit page for ingesting and chatting.
  - Validates that vector data exists before running retrieval.
- [pages/3_📊_Vector_Demo.py](pages/3_%F0%9F%93%8A_Vector_Demo.py)
  - Displays database/vector metrics.
  - Shows ingest usage stats (tokens and cost) loaded from persisted ingest summary.
  - Shows chunk cards with expanders (text preview, source, vector preview, embedding usage).
- [ingest.py](ingest.py)
  - Loads files, chunks text, generates embeddings, and upserts to Qdrant.
  - Tracks and returns ingest stats:
    - `chunks_indexed`
    - `total_tokens_used`
    - `avg_tokens_per_chunk`
    - `total_embedding_cost_usd`
  - Persists latest stats to [ingest_stats.json](ingest_stats.json).
- [rag.py](rag.py)
  - Performs similarity search and streams final answer generation.
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
3. Start Qdrant on `localhost:6333`.
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
4. Open **Vector Demo** in the sidebar to inspect metrics and chunk-level details.

## Notes
- Re-ingesting recreates the Qdrant collection and replaces previous vectors.
- Ingest stats are loaded from [ingest_stats.json](ingest_stats.json), so no full recomputation is needed on the vector page.
- If you indexed data before stats tracking was added, re-ingest once to populate latest usage stats.
