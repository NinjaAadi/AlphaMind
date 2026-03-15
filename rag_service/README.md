# RAG Service (AlphaMind)

**Port: 8002** — Natural-language Q&A over stock data and model predictions.

Part of [AlphaMind](../README.md): this service answers questions like “What did the model predict for TCS?” by fetching context from the **scraper service** (8000) and **model service** (8001), then sending it to a local **Ollama** LLM (e.g. Llama 3.2). Before 3:15 PM IST it uses **live APIs**; after 3:15 PM IST it uses a **vector DB** (ChromaDB) populated at startup or via `POST /ingest`, with fallback to live APIs if the DB has no hits.

---

## How it works

1. **Query**: You ask a question (e.g. “What did the model predict for Reliance?”).
2. **Fresh fetch**: The service infers which stocks you’re asking about and **calls model_service and scraper_service right then** to get **current** predictions and stock data.
3. **Answer**: That fresh context is sent to **Ollama** (Llama); the LLM answers only from this data.

- **Before 3:15 PM IST**: Every answer is based on **live** data from model_service and scraper_service APIs.
- **After 3:15 PM IST**: The service first runs **ingest** once per day (loads all configured stocks into the vector DB), then answers **from the vector DB** instead of calling the APIs.

**How the vector DB gets loaded**
- **On server startup** (default): If `PRELOAD_VECTOR_DB=1`, the server runs ingest once at startup and fills the vector DB with all configured stocks (from `INGEST_STOCKS_FILE` or default list). Ensure model_service (8001) and scraper_service (8000) are running before starting the RAG service.
- **Manually**: `curl -X POST http://localhost:8002/ingest` or `POST /ingest` with body `{"stocks": ["Reliance", "TCS", ...]}` to load or refresh the DB.
- **After 3:15 PM**: On the first query of the day after 3:15 PM IST, the server runs ingest if it hasn’t run yet that day, then answers from the DB.

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running (free, local Llama):
  - Install: https://ollama.com
  - Then: `ollama run llama3.2` (default; or `llama4`, `mistral`, etc.)
- **model_service** and **scraper_service** running (for ingestion and for the pipeline to make sense)

## Setup

```bash
cd rag_service
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` and set URLs if your services are not on default ports.

```bash
cp .env.example .env
# Edit .env if needed: MODEL_SERVICE_URL, SCRAPER_API_URL, OLLAMA_HOST, OLLAMA_MODEL
```

## Run the RAG API

```bash
# From rag_service/ with venv activated
uvicorn api.server:app --host 0.0.0.0 --port 8002
```

API will be at **http://localhost:8002**.

### Testing: single stock

To test with **one stock only** (faster startup, smaller vector DB), set either env var:

- `SINGLE_STOCK=TCS`  
- `TEST_SINGLE_STOCK=TCS`

Then start the server. Startup preload and ingest will use only that stock. Unset or leave empty for the full stock list.

```bash
SINGLE_STOCK=TCS uvicorn api.server:app --host 0.0.0.0 --port 8002
# or in .env: SINGLE_STOCK=TCS
```

## Usage

### Ask questions (uses fresh API data)

When you query, the service **fetches current data** from model_service and scraper_service for the stocks mentioned in your question. No ingest step is required for that.

```bash
curl -X POST http://localhost:8002/query -H "Content-Type: application/json" \
  -d '{"question": "What did the model predict for Reliance?"}'
```

Example response:

```json
{
  "answer": "For Reliance, the model predicts ...",
  "stocks_fetched": 1,
  "collection_total": 0
}
```

Example questions:

- "What did the model predict for Reliance?"
- "What is the current price and PE for TCS?"
- "Compare Infosys and TCS predictions."

Stocks are inferred from the question text (e.g. “Reliance”, “TCS”, “Infosys”). Only those stocks are fetched, so data is always up to date.

### Optional: ingest (for RAG / batch use)

If you want to pre-index data into the vector store (e.g. for other tools or batch jobs), you can still call ingest. **Query does not use this**; it always fetches fresh.

```bash
curl -X POST http://localhost:8002/ingest
curl -X POST http://localhost:8002/ingest -H "Content-Type: application/json" -d '{"stocks": ["Reliance", "TCS", "Infosys"]}'
curl "http://localhost:8002/ingest?stocks=Reliance,TCS"
```

### Cron: daily RAG ingest at 3:16 AM

To store **all** data in the RAG vector DB once per day (e.g. after market close), run the ingest script via cron at **3:16 AM IST**.

1. **Stocks list**: By default the script reads from `model_service/training_stocks.txt` (one symbol per line). Override with env `INGEST_STOCKS_FILE`.

2. **Run manually** (from the project directory, i.e. current dir = `rag_service`):

   ```bash
   cd rag_service
   source venv/bin/activate
   python -m run_ingest
   ```

3. **Crontab** (3:16 AM IST). Use the **full path to the current (project) directory** — i.e. the folder that contains `rag_service`.

   **Option A – wrapper script** (script uses its own directory; only one path in crontab). From project root, make it executable once: `chmod +x rag_service/run_ingest_cron.sh`

   If your server uses **IST**:

   ```cron
   16 3 * * * /Users/aaditya.pal/Documents/AlphaMind/rag_service/run_ingest_cron.sh >> /tmp/rag_ingest.log 2>&1
   ```

   If your server uses **UTC** (3:16 AM IST = 21:46 UTC previous day):

   ```cron
   46 21 * * * /Users/aaditya.pal/Documents/AlphaMind/rag_service/run_ingest_cron.sh >> /tmp/rag_ingest.log 2>&1
   ```

   **Option B – inline** (same path for `cd` and python):

   ```cron
   16 3 * * * cd /Users/aaditya.pal/Documents/AlphaMind/rag_service && ./venv/bin/python -m run_ingest >> /tmp/rag_ingest.log 2>&1
   ```

   Ensure `model_service` and `scraper_service` are running (or reachable) when the cron runs so the script can fetch data.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check; reports Ollama and collection size |
| POST | `/query` | Body: `{"question": "..."}` → **fetch fresh data from APIs** + LLM answer (no RAG) |
| POST | `/ingest` | Body: `{"stocks": ["Reliance", "TCS"]}` (optional) → index into RAG store (optional) |
| GET | `/ingest?stocks=Reliance,TCS` | Same as POST /ingest with default or comma-separated stocks |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_SERVICE_URL` | http://localhost:8001 | model_service API base URL |
| `SCRAPER_API_URL` | http://localhost:8000 | scraper_service API base URL |
| `OLLAMA_HOST` | http://localhost:11434 | Ollama API URL |
| `OLLAMA_MODEL` | llama3.2 | Model name (e.g. llama3.2, llama4, mistral) |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB persistence directory |
| `INGEST_STOCKS_FILE` | `../model_service/training_stocks.txt` | File with one stock symbol per line for post-market ingest |
| `RAG_TOP_K` | 5 | Number of chunks to retrieve per query (vector DB) |
| `PRELOAD_VECTOR_DB` | 1 | If 1/true, load vector DB on server startup (needs 8000, 8001 running) |
| `MARKET_CLOSE_HOUR` | 15 | Hour (24h) after which to use vector DB — default 3 PM |
| `MARKET_CLOSE_MINUTE` | 15 | Minute — default 3:15 PM |
| `TIMEZONE` | Asia/Kolkata | Timezone for market-close check |
| `SINGLE_STOCK` or `TEST_SINGLE_STOCK` | (empty) | If set (e.g. `TCS`), ingest and preload only this stock — for testing |

## Architecture

- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2` (local, no API key).
- **Vector store**: ChromaDB (local, persistent).
- **LLM**: Ollama (e.g. Llama 3.2) via `POST /api/generate`.

All components run locally; no paid APIs required.

For **end-to-end flow diagrams** (live path, vector-DB path, startup preload), see the [main project README](../README.md#flow-diagrams).
