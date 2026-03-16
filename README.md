# AlphaMind

**Stock prediction and natural-language Q&A** for Indian equities. Three microservices: **scraper** (market + fundamentals + news), **model** (TFT predictions), and **RAG** (LLM answers from live data or vector DB).

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Flow diagrams](#flow-diagrams)
- [Quick start](#quick-start)
- [Complete ML pipeline](#complete-ml-pipeline)
- [Services](#services)
- [Environment setup](#environment-setup)

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

## Complete ML pipeline

This section describes the **end-to-end machine learning pipeline** from data collection to serving predictions.

### Pipeline overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AlphaMind ML Pipeline                                │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: SETUP & INSTALLATION
  1. Install prerequisites (Python 3.10+, Ollama)
  2. Set up virtual environments for all services
  3. Configure environment variables (.env files)
  4. Start Ollama and pull the LLM model

PHASE 2: DATA COLLECTION (Training Data)
  5. Start scraper service (port 8000)
  6. Run data pipeline to collect stock features
  7. Generate training CSV (training_data.csv)

PHASE 3: MODEL TRAINING
  8. Train Temporal Fusion Transformer (TFT) model
  9. Evaluate model performance
  10. Save trained model weights (.pt file)

PHASE 4: DEPLOYMENT & INFERENCE
  11. Start all three services (scraper, model, RAG)
  12. Model service loads trained weights
  13. RAG service preloads vector DB (optional)
  14. Services are ready for predictions and Q&A
```

### Step-by-step instructions

#### PHASE 1: Setup & Installation

**1. Install prerequisites**

```bash
# Python 3.10 or higher required
python3 --version

# Install Ollama (macOS)
brew install ollama

# Or download from https://ollama.com for other platforms
```

**2. Pull the LLM model**

```bash
# Start Ollama service
ollama serve

# In another terminal, pull the model
ollama pull llama3.2
```

**3. Set up virtual environments**

```bash
# Create virtual environments for each service
cd scraper_service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate

cd ../model_service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate

cd ../rag_service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

**4. Configure environment variables**

Create `.env` files for each service by copying from `.env.example`:

```bash
# Scraper service
cd scraper_service
cp .env.example .env
# Edit .env and add your NewsAPI key
# NEWS_API_KEY=your_newsapi_key_here

# Model service
cd ../model_service
cp .env.example .env
# Edit if needed (defaults work for local setup)

# RAG service
cd ../rag_service
cp .env.example .env
# Edit if needed (defaults work for local setup)
```

#### PHASE 2: Data Collection

**5. Start the scraper service**

```bash
cd scraper_service
source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

The scraper service must be running for the data pipeline to work.

**6. Run the data pipeline**

Open a new terminal:

```bash
cd model_service
source venv/bin/activate

# Run the pipeline to fetch training data
python run_pipeline.py

# This will:
# - Read stock symbols from training_stocks.txt (100 stocks)
# - Fetch historical data from scraper service
# - Generate training_data.csv in the output/ directory
# - Take ~5-10 minutes depending on API response times
```

**Pipeline options:**

```bash
# Force regenerate even if CSV exists
python run_pipeline.py --force

# Use custom stock list
python run_pipeline.py --stocks-file my_stocks.txt

# Use more parallel workers (faster)
python run_pipeline.py --workers 10

# Custom output file
python run_pipeline.py --output data/my_training_data.csv
```

**7. Verify training data**

```bash
cd model_service
ls -lh output/training_data.csv

# Check the data
head output/training_data.csv
```

Expected columns: `ticker`, `date`, `time_idx`, `open`, `high`, `low`, `close`, `volume`, technical indicators (RSI, SMA, MACD), fundamental ratios, and `target_return`.

#### PHASE 3: Model Training

**8. Train the TFT model**

```bash
cd model_service/model_train
source ../venv/bin/activate

# Train the model
python train_tft.py

# This will:
# - Load and clean training_data.csv
# - Create train/validation splits (80/20)
# - Train Temporal Fusion Transformer
# - Save model to ../models/tft_stock_model.pt
# - Training takes ~30-60 minutes (20 epochs, depends on hardware)
```

**Training options:**

```bash
# Resume training from checkpoint
python train_tft.py --resume lightning_logs/version_0/checkpoints/epoch=10.ckpt

# Train for more epochs
python train_tft.py --max-epochs 50
```

**9. Monitor training**

The training script will output:
- Data preprocessing statistics
- Training progress (loss per epoch)
- Validation metrics
- Early stopping (if validation loss doesn't improve)

**10. Verify model weights**

```bash
cd model_service
ls -lh models/tft_stock_model.pt

# Should be ~1-5 MB depending on model complexity
```

#### PHASE 4: Deployment & Inference

**11. Start all services**

You need **three terminals** (or use tmux/screen):

**Terminal 1 - Scraper service:**

```bash
cd scraper_service
source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Model service:**

```bash
cd model_service
source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8001

# On startup, the model service will:
# - Look for models/tft_stock_model.pt
# - Load the trained model weights
# - Be ready to serve predictions
```

**Terminal 3 - RAG service:**

```bash
cd rag_service
source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8002

# On startup, the RAG service will:
# - Connect to ChromaDB
# - Optionally preload vector DB (if PRELOAD_VECTOR_DB=1)
# - Connect to Ollama
# - Be ready to answer questions
```

**Terminal 4 (optional) - Ollama:**

If Ollama isn't running as a service:

```bash
ollama serve
```

**12. Verify all services are running**

```bash
# Check scraper
curl http://localhost:8000/health

# Check model
curl http://localhost:8001/health

# Check RAG
curl http://localhost:8002/health
```

**13. Open the UI**

Navigate to: [http://localhost:8000](http://localhost:8000)

You should see:
- Stock data dashboard
- "Ask the LLM" chat interface

### Testing the pipeline

**Test stock data retrieval:**

```bash
curl "http://localhost:8000/stock?company=TCS"
```

**Test model predictions:**

```bash
curl "http://localhost:8001/predict?stock=TCS"
```

**Test RAG Q&A:**

```bash
curl -X POST http://localhost:8002/api/llm/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the prediction for TCS?"}'
```

**Test via UI:**
1. Go to http://localhost:8000
2. Click "Ask the LLM"
3. Type: "What is the 5-day prediction for TCS?"
4. The system will:
   - Extract stock symbol (TCS)
   - Fetch current data from scraper
   - Get prediction from model service
   - Use Ollama to generate natural language answer

### Pipeline maintenance

**Retraining the model:**

To retrain with fresh data:

```bash
# 1. Regenerate training data
cd model_service
source venv/bin/activate
python run_pipeline.py --force

# 2. Delete old model
rm models/tft_stock_model.pt

# 3. Train new model
cd model_train
python train_tft.py
```

**Adding new stocks:**

Edit `model_service/training_stocks.txt` and add stock symbols (one per line), then regenerate training data.

**Updating vector DB:**

The RAG service auto-updates the vector DB after market close (3:15 PM IST). To manually trigger:

```bash
curl -X POST http://localhost:8002/api/ingest
```

---

## Services

| Service | Path | README | Purpose |
|---------|------|--------|---------|
| **Scraper** | `scraper_service/` | [scraper_service/README.md](scraper_service/README.md) | Stock data aggregation (YFinance, Screener, NewsAPI), main UI, LLM proxy. |
| **Model** | `model_service/` | [model_service/README.md](model_service/README.md) | TFT training, prediction API, depends on scraper for features. |
| **RAG** | `rag_service/` | [rag_service/README.md](rag_service/README.md) | Natural-language Q&A: context from APIs or vector DB, answer via Ollama. |

---

## Environment setup

Each service requires environment variables configured via a `.env` file. Copy `.env.example` to `.env` in each service directory and customize as needed.

### Scraper service environment variables

Location: `scraper_service/.env`

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEWS_API_KEY` | NewsAPI key for fetching stock news. Get from [newsapi.org](https://newsapi.org/) | - | **Yes** |
| `RAG_SERVICE_URL` | URL of the RAG service for LLM queries | `http://localhost:8002` | No |
| `API_HOST` | Host to bind the API server | `0.0.0.0` | No |
| `API_PORT` | Port for the scraper service | `8000` | No |
| `API_RELOAD` | Enable auto-reload during development | `false` | No |
| `SSL_VERIFY` | Verify SSL certificates for external APIs | `true` | No |

**Example `.env`:**

```bash
# Required
NEWS_API_KEY=your_newsapi_key_here

# Optional (defaults work)
RAG_SERVICE_URL=http://localhost:8002
API_HOST=0.0.0.0
API_PORT=8000
```

**Getting NewsAPI key:**
1. Go to [https://newsapi.org/](https://newsapi.org/)
2. Sign up for free account
3. Get API key from dashboard
4. Add to `.env` file

### Model service environment variables

Location: `model_service/.env`

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SCRAPER_API_URL` | Base URL of the scraper service | `http://localhost:8000` | **Yes** |
| `PORT` | Port for the model service | `8001` | No |
| `MODEL_PATH` | Path to trained model file (.pt or .ckpt) | Auto-detected: `models/tft_stock_model.pt` or latest checkpoint | No |
| `API_URL` | Scraper endpoint for data pipeline | `http://localhost:8000/stock` | No (for `run_pipeline.py`) |

**Example `.env`:**

```bash
# Required
SCRAPER_API_URL=http://localhost:8000

# Optional
PORT=8001
MODEL_PATH=models/tft_stock_model.pt
API_URL=http://localhost:8000/stock
```

**Model path resolution:**

The model service looks for the model file in this order:
1. `MODEL_PATH` if set in `.env`
2. `models/tft_stock_model.pt` (recommended location)
3. Latest checkpoint in `lightning_logs/`

### RAG service environment variables

Location: `rag_service/.env`

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MODEL_SERVICE_URL` | URL of the model service | `http://localhost:8001` | **Yes** |
| `SCRAPER_API_URL` | URL of the scraper service | `http://localhost:8000` | **Yes** |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` | **Yes** |
| `OLLAMA_MODEL` | Ollama model name | `llama3.2` | **Yes** |
| `PORT` | Port for the RAG service | `8002` | No |
| `RAG_TOP_K` | Number of top results to retrieve from vector DB | `5` | No |
| `CHROMA_PERSIST_DIR` | Directory for ChromaDB persistence | `./chroma_data` | No |
| `CHROMA_COLLECTION_NAME` | ChromaDB collection name | `alphamind_rag` | No |
| `OLLAMA_TIMEOUT` | Timeout for Ollama requests (seconds) | `90` | No |
| `PRELOAD_VECTOR_DB` | Preload vector DB on startup (1=yes, 0=no) | `1` | No |
| `MARKET_CLOSE_HOUR` | Hour when market closes (IST, 24-hour format) | `15` (3 PM) | No |
| `MARKET_CLOSE_MINUTE` | Minute when market closes | `15` | No |
| `TIMEZONE` | Timezone for market hours | `Asia/Kolkata` | No |
| `INGEST_STOCKS_FILE` | File with stock symbols for vector DB ingestion | `../model_service/training_stocks.txt` | No |
| `SINGLE_STOCK` | Ingest only this stock (for testing) | - | No |
| `TEST_SINGLE_STOCK` | Test mode: preload only this stock | - | No |

**Example `.env`:**

```bash
# Required
MODEL_SERVICE_URL=http://localhost:8001
SCRAPER_API_URL=http://localhost:8000
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Optional
PORT=8002
RAG_TOP_K=5
PRELOAD_VECTOR_DB=1
MARKET_CLOSE_HOUR=15
MARKET_CLOSE_MINUTE=15
TIMEZONE=Asia/Kolkata

# Testing (optional)
# SINGLE_STOCK=TCS
```

**Vector DB behavior:**

- **Before 3:15 PM IST:** All queries use live APIs (scraper + model)
- **After 3:15 PM IST:** Queries use ChromaDB vector store (faster, offline-capable)
- **Preloading:** If `PRELOAD_VECTOR_DB=1`, the vector DB is populated on startup
- **Auto-ingestion:** After market close, the RAG service automatically ingests the day's data into ChromaDB

**Testing with single stock:**

For faster startup during development:

```bash
# In rag_service/.env
SINGLE_STOCK=TCS
```

This will only ingest TCS data instead of all 100 stocks.

### Common environment setup issues

**Issue: NewsAPI key not working**
- Verify key is correct in `scraper_service/.env`
- Check NewsAPI quota (free tier: 100 requests/day)
- Ensure no trailing spaces in `.env` file

**Issue: Model service can't find trained model**
- Verify `models/tft_stock_model.pt` exists
- Check `MODEL_PATH` in `.env`
- Run training: `cd model_service/model_train && python train_tft.py`

**Issue: Ollama connection failed**
- Ensure Ollama is running: `ollama serve`
- Verify model is pulled: `ollama list` (should show llama3.2)
- Check `OLLAMA_HOST` URL in `.env`

**Issue: Services can't communicate**
- Ensure all services use correct ports (8000, 8001, 8002)
- Check firewall settings
- Verify URLs in `.env` files match running services

### Environment variables quick reference

**Minimal setup (required only):**

```bash
# scraper_service/.env
NEWS_API_KEY=your_key_here

# model_service/.env
SCRAPER_API_URL=http://localhost:8000

# rag_service/.env
MODEL_SERVICE_URL=http://localhost:8001
SCRAPER_API_URL=http://localhost:8000
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

**All other variables use sensible defaults** and only need to be set if you want to customize ports, paths, or behavior.

### Production deployment notes

For production deployments, consider:

1. **Security:**
   - Use environment-specific `.env` files (never commit to git)
   - Store API keys in secure vault (AWS Secrets Manager, HashiCorp Vault)
   - Enable HTTPS/TLS for all service communication
   - Set `API_HOST=127.0.0.1` if services shouldn't be publicly accessible

2. **Performance:**
   - Use production ASGI server (gunicorn + uvicorn workers)
   - Enable caching in scraper service
   - Increase `RAG_TOP_K` for better context (higher latency)
   - Use dedicated ChromaDB instance (not embedded)

3. **Reliability:**
   - Run services with process manager (systemd, supervisord, PM2)
   - Configure auto-restart on failure
   - Set up monitoring and alerting
   - Use load balancer for high availability

4. **Scalability:**
   - Deploy services in containers (Docker)
   - Use orchestration platform (Kubernetes, Docker Swarm)
   - Scale RAG service horizontally for concurrent queries
   - Consider Redis for distributed caching

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

---

## Quick command reference

### Initial setup (one-time)

```bash
# 1. Install Ollama and pull model
ollama pull llama3.2

# 2. Set up all services
for service in scraper_service model_service rag_service; do
  cd $service
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  cp .env.example .env
  deactivate
  cd ..
done

# 3. Edit .env files (add NEWS_API_KEY in scraper_service/.env)
```

### Training workflow

```bash
# 1. Start scraper
cd scraper_service && source venv/bin/activate
uvicorn api.server:app --port 8000 &

# 2. Collect training data
cd model_service && source venv/bin/activate
python run_pipeline.py

# 3. Train model
cd model_train
python train_tft.py
```

### Running services

```bash
# Terminal 1
cd scraper_service && source venv/bin/activate && uvicorn api.server:app --port 8000

# Terminal 2
cd model_service && source venv/bin/activate && uvicorn api.server:app --port 8001

# Terminal 3
cd rag_service && source venv/bin/activate && uvicorn api.server:app --port 8002

# Terminal 4 (if needed)
ollama serve
```

### Testing commands

```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health

# Get stock data
curl "http://localhost:8000/stock?company=TCS"

# Get prediction
curl "http://localhost:8001/predict?stock=TCS"

# Ask LLM
curl -X POST http://localhost:8002/api/llm/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the prediction for TCS?"}'

# Manually trigger vector DB ingest
curl -X POST http://localhost:8002/api/ingest
```

### Common maintenance

```bash
# Retrain with fresh data
cd model_service && source venv/bin/activate
python run_pipeline.py --force
rm models/tft_stock_model.pt
cd model_train && python train_tft.py

# Add new stocks
echo "NEWSTOCK" >> model_service/training_stocks.txt
cd model_service && python run_pipeline.py --force

# View logs
tail -f model_service/pipeline.log

# Clear ChromaDB (fresh start)
rm -rf rag_service/chroma_data
```

---

**Questions?** See individual service READMs for detailed documentation or open an issue on GitHub.
