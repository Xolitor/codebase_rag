# Codebase RAG Chatbot

## Project Aim
This project is a simple Retrieval-Augmented Generation (RAG) chatbot for source code.
It indexes files from a local repository folder, stores vector embeddings in Qdrant, retrieves relevant code snippets for a user question, and generates an answer with an OpenAI chat model.

## Tech Stack
- Python: Core implementation language for ingestion, retrieval, and app orchestration.
- Streamlit: Web UI for indexing and asking questions.
- OpenAI API:
  - Embeddings (`text-embedding-3-small`) to convert code chunks into vectors.
  - Chat completion model (`gpt-4o-mini`) to generate final answers from retrieved context.
- Qdrant: Vector database to store and search code embeddings.
- python-dotenv: Loads environment variables (API key) from `.env`.
- tiktoken: Installed dependency for tokenization-related workflows.

## Requirements
- Python 3.14+ recommended.
- A running Qdrant instance on `localhost:6333`.
- An OpenAI API key in a `.env` file.
- Python packages from `requirements.txt`:
  - openai==2.29.0
  - qdrant-client==1.17.1
  - streamlit==1.55.0
  - python-dotenv==1.2.2
  - tiktoken==0.12.0

## Technical Explanation (File Responsibilities)
- `app.py`
  - Streamlit entrypoint/UI.
  - Provides the "Ingest codebase" button.
  - Accepts a user query, runs retrieval, then displays the generated answer and sources.

- `ingest.py`
  - Loads source files from `./repo`.
  - Splits each file into overlapping chunks.
  - Creates embeddings for each chunk.
  - Creates/recreates the Qdrant collection and upserts points.

- `rag.py`
  - Runs similarity retrieval against Qdrant using `query_points`.
  - Builds the final context from retrieved chunk payloads.
  - Calls OpenAI chat completions to produce the answer.

- `config.py`
  - Central configuration values:
    - model names
    - collection name
    - vector size
    - top-k retrieval size
  - Loads `OPENAI_API_KEY` from environment.

- `requirements.txt`
  - Project dependencies and versions.

- `repo/`
  - Example codebase to index (currently includes `api.js`, `auth.py`, `database.py`, `utils.py`).

## How to Run
1. Create and activate a virtual environment.
2. Install dependencies.
3. Start Qdrant.
4. Set `OPENAI_API_KEY` in `.env`.
5. Run Streamlit.

Example commands (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start Qdrant with Docker (if you do not already run it):

```powershell
docker run -p 6333:6333 qdrant/qdrant
```

Create `.env` in project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Run the app:

```powershell
streamlit run app.py
```

## How to Access Components
- Streamlit UI:
  - Open the local URL shown by Streamlit (typically `http://localhost:8501`).
  - Click "Ingest codebase" to index files from `./repo`.
  - Ask a question in the input box.

- Qdrant:
  - API endpoint at `http://localhost:6333/dashboard#/collections`.
  - Stores vectors in the collection named `codebase`.

- OpenAI:
  - Used by ingestion (embeddings) and answer generation (chat model).

## Notes
- Re-ingesting recreates the collection, so previous vectors are replaced.
- If indexing returns zero points, retrieval will not return useful context.
