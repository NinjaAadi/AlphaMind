# AlphaMind

**Stock prediction and natural-language Q&A** for Indian equities. Three microservices: **scraper** (market + fundamentals + news), **model** (TFT predictions), and **RAG** (LLM answers from live data or vector DB).

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Flow diagrams](#flow-diagrams)
- [Quick start](#quick-start)
- [Services](#services)
- [Environment summary](#environment-summary)

---

## Overview

| Goal | How AlphaMind does it |
|------|------------------------|
| **Stock data** | Scraper service aggregates live data from YFinance, Screener.in, and NewsAPI. |
| **Predictions** | Model service runs a trained Temporal Fusion Transformer (TFT) to predict 5-day returns. |
| **Natural-language Q&A** | RAG service fetches context from scraper + model (or from vector DB after market close), then uses a local LLM (Ollama/Llama) to answer in plain English. |

**Time-based behaviour**

- **Before 3:15 PM IST:** Every query uses **live APIs** (scraper + model) so answers reflect current data.
- **After 3:15 PM IST:** RAG uses a **vector DB** (preloaded at startup or via cron). If the DB has no hits, it falls back to live APIs.

---

## Architecture

### High-level

```
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                         AlphaMind                                 │
                    └─────────────────────────────────────────────────────────────────┘
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         │                                            │                                            │
         ▼                                            ▼                                            ▼
┌─────────────────┐                        ┌─────────────────┐                        ┌─────────────────┐
│  Scraper (8000)  │                        │  Model (8001)   │                        │   RAG (8002)    │
│  Stock data API  │◄───────────────────────│  TFT Predictions│                        │  LLM Q&A API    │
│  + UI dashboard  │   GET /stock           │                 │                        │  + Vector DB     │
└────────┬─────────┘                        └────────┬────────┘                        └────────┬────────┘
         │                                            │                                            │
         │  YFinance, Screener, NewsAPI               │  Loads .ckpt / .pt                       │  Ollama (Llama)
         ▼                                            ▼                                            ▼
┌─────────────────┐                        ┌─────────────────┐                        ┌─────────────────┐
│  External       │                        │  Scraper (8000) │                        │  ChromaDB       │
│  data sources   │                        │  for features   │                        │  + Ollama 11434  │
└─────────────────┘                        └─────────────────┘                        └─────────────────┘
```

### Component roles

| Component | Port | Role |
|-----------|------|------|
| **Scraper service** | 8000 | Aggregates stock data (prices, technicals, fundamentals, news). Serves the main **UI** (dashboard + “Ask the LLM” chat). Proxies LLM requests to RAG. |
| **Model service** | 8001 | Loads a trained TFT model; fetches current features from scraper; returns 5-day return predictions per stock. |
| **RAG service** | 8002 | Infers stocks from the question; gets context from scraper + model (or from vector DB after 3:15 PM IST); calls Ollama to produce the answer. |
| **Ollama** | 11434 | Local LLM (e.g. Llama 3.2). Not part of the repo; must be installed and running. |

---

## Tech stack

| Layer | Technology |
|-------|------------|
| **API / Web** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Static HTML/JS (scraper `static/`), no framework |
| **Data ingestion** | YFinance, Screener.in (scraping), NewsAPI |
| **ML / Prediction** | PyTorch, PyTorch Forecasting (TFT), Lightning |
| **RAG / Embeddings** | ChromaDB, sentence-transformers (all-MiniLM-L6-v2) |
| **LLM** | Ollama (Llama 3.2 default) |
| **Config** | Environment variables, python-dotenv |

---

## Flow diagrams

### 1. User asks a question (before 3:15 PM IST — live path)

```
  User                UI (8000)           RAG (8002)           Scraper (8000)    Model (8001)    Ollama (11434)
    │                     │                    │                     │                 │                │
    │  "TCS prediction?"  │                    │                     │                 │                │
    │────────────────────►│                    │                     │                 │                │
    │                     │  POST /api/llm/query (question)          │                 │                │
    │                     │───────────────────►│                     │                 │                │
    │                     │                    │  GET /stock?company=TCS               │                │
    │                     │                    │────────────────────►                 │                │
    │                     │                    │  GET /predict?stock=TCS             │                │
    │                     │                    │─────────────────────────────────────►│                │
    │                     │                    │  (model may call scraper for features)                │
    │                     │                    │  POST /api/generate (context + prompt) │                │
    │                     │                    │───────────────────────────────────────────────────────►│
    │                     │                    │  answer                              │                │
    │                     │                    │◄───────────────────────────────────────────────────────│
    │                     │  JSON { answer }    │                     │                 │                │
    │                     │◄───────────────────│                     │                 │                │
    │  Answer text        │                    │                     │                 │                │
    │◄────────────────────│                    │                     │                 │                │
```

### 2. User asks a question (after 3:15 PM IST — vector DB path)

```
  User                RAG (8002)           ChromaDB            Ollama
    │                     │                    │                 │
    │  "TCS prediction?"  │                    │                 │
    │────────────────────►│                    │                 │
    │                     │  ensure_post_market_ingest() if needed │
    │                     │  query_documents(question) → top-k     │
    │                     │───────────────────►│                 │
    │                     │  context chunks    │                 │
    │                     │◄───────────────────│                 │
    │                     │  (if no context → fallback to live APIs as in diagram 1)
    │                     │  POST /api/generate                  │
    │                     │──────────────────────────────────────►│
    │                     │  answer                               │
    │                     │◄──────────────────────────────────────│
    │  Answer             │                    │                 │
    │◄────────────────────│                    │                 │
```

### 3. Startup preload (vector DB)

```
  RAG server start       ingest_stocks(stocks)        Scraper (8000)    Model (8001)    ChromaDB
        │                         │                         │                 │            │
        │  PRELOAD_VECTOR_DB=1     │                         │                 │            │
        │  get_stocks_for_post_market_ingest() → e.g. [TCS]  │                 │            │
        │────────────────────────►│                         │                 │            │
        │                         │  for each stock:        │                 │            │
        │                         │  GET /stock?company=... │                 │            │
        │                         │────────────────────────►                 │            │
        │                         │  GET /predict?stock=...  │                 │            │
        │                         │─────────────────────────────────────────►│            │
        │                         │  build docs, add_documents()             │            │
        │                         │─────────────────────────────────────────────────────►│
        │  "Server ready"         │                         │                 │            │
        │◄────────────────────────│                         │                 │            │
```

---

## Quick start

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and a model pulled (e.g. `ollama run llama3.2`)
- **NewsAPI key** (for scraper news) — set `NEWS_API_KEY` in scraper `.env`

### 1. Start all three services

```bash
# Terminal 1 — Scraper (UI + stock API)
cd scraper_service && source venv/bin/activate && uvicorn api.server:app --host 0.0.0.0 --port 8000

# Terminal 2 — Model (predictions)
cd model_service && ./venv/bin/python -m uvicorn api.server:app --host 0.0.0.0 --port 8001

# Terminal 3 — RAG (LLM Q&A)
cd rag_service && source venv/bin/activate && uvicorn api.server:app --host 0.0.0.0 --port 8002
```

### 2. Use the app

- **Main UI (chat + stock data):** [http://localhost:8000](http://localhost:8000)
- **RAG API docs:** [http://localhost:8002/docs](http://localhost:8002/docs)
- **Model API docs:** [http://localhost:8001/docs](http://localhost:8001/docs)

### 3. Single-stock testing (RAG)

To preload only one stock (e.g. TCS) for faster startup:

```bash
# In rag_service/.env or when starting:
SINGLE_STOCK=TCS uvicorn api.server:app --host 0.0.0.0 --port 8002
```

---

## Services

| Service | Path | README | Purpose |
|---------|------|--------|---------|
| **Scraper** | `scraper_service/` | [scraper_service/README.md](scraper_service/README.md) | Stock data aggregation (YFinance, Screener, NewsAPI), main UI, LLM proxy. |
| **Model** | `model_service/` | [model_service/README.md](model_service/README.md) | TFT training, prediction API, depends on scraper for features. |
| **RAG** | `rag_service/` | [rag_service/README.md](rag_service/README.md) | Natural-language Q&A: context from APIs or vector DB, answer via Ollama. |

---

## Environment summary

| Service | Key variables | Defaults |
|---------|----------------|----------|
| **Scraper** | `NEWS_API_KEY`, `PORT` | Port 8000 |
| **Model** | `SCRAPER_API_URL`, `PORT`, `MODEL_PATH` | 8001, scraper at 8000 |
| **RAG** | `MODEL_SERVICE_URL`, `SCRAPER_API_URL`, `OLLAMA_HOST`, `OLLAMA_MODEL`, `PRELOAD_VECTOR_DB`, `SINGLE_STOCK` / `TEST_SINGLE_STOCK` | 8001, 8000, 11434, llama3.2, 1, (empty) |

See each service’s README and `.env.example` for full lists.

---

## Repository layout

```
AlphaMind/
├── README.md                 # This file — project overview and architecture
├── scraper_service/          # Stock data + UI (port 8000)
│   ├── README.md
│   ├── api/server.py
│   ├── providers/             # YFinance, News, Screener
│   ├── services/              # Aggregator
│   └── static/                # Dashboard UI
├── model_service/             # TFT predictions (port 8001)
│   ├── README.md
│   ├── api/server.py         # Prediction API
│   ├── model_train/           # Training script (TFT)
│   └── run_pipeline.py       # Data pipeline (fetch → CSV)
└── rag_service/               # RAG + LLM Q&A (port 8002)
    ├── README.md
    ├── api/server.py         # Query, ingest, health
    ├── ingest.py             # Fetch from 8000/8001 → ChromaDB
    ├── vector_store.py       # ChromaDB + embeddings
    └── llm_client.py         # Ollama client
```
