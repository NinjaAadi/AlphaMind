"""
RAG API: query your model's predictions and stock data via natural language using a free LLM (Ollama/Llama).
Before market close (3:15 PM IST): context from live APIs. After 3:15 PM: ingest all stocks into vector DB once, then query from DB.
"""
import os
import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config import (
    RAG_PROMPT_TEMPLATE,
    PREDICTION_FULL_PROMPT,
    PREDICTION_REWRITE_PROMPT,
    MODEL_SERVICE_URL,
    SCRAPER_API_URL,
    OLLAMA_HOST,
    RAG_TOP_K,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    TIMEZONE,
    CHROMA_PERSIST_DIR,
    PRELOAD_VECTOR_DB,
)
from vector_store import collection_count, query_documents
from llm_client import ollama_generate, ollama_health
from ingest import ingest_stocks, fetch_fresh_context_for_question, get_stocks_for_post_market_ingest, DEFAULT_STOCKS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about stocks or model predictions")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="LLM answer based on fresh API data")
    stocks_fetched: int = Field(..., description="Number of stocks fetched from APIs for context")
    collection_total: int = Field(0, description="Total documents in vector store (if RAG used)")


class IngestRequest(BaseModel):
    stocks: list[str] = Field(default_factory=lambda: DEFAULT_STOCKS, description="List of stock names/tickers to ingest")


class IngestResponse(BaseModel):
    documents_added: int
    message: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # One-time at startup: load vector DB so after-hours queries can use it. Progress is logged below.
    if PRELOAD_VECTOR_DB:
        try:
            stocks = get_stocks_for_post_market_ingest()
            n = len(stocks)
            logger.info(f"One-time startup: loading vector DB for {n} stocks (you will see progress below)...")
            added = ingest_stocks(stocks, log_progress=True)
            logger.info(f"Startup preload done: ingested {added} documents for {n} stocks. Server ready.")
        except Exception as e:
            logger.warning(f"Startup preload (vector DB) failed: {e}. Ensure model_service and scraper_service are running. You can load later via POST /ingest.")
    yield


app = FastAPI(
    title="AlphaMind RAG",
    description="Ask questions about your stock model predictions and data (RAG + Ollama/Llama)",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
def root():
    """Landing page: use the main UI on port 8000 for the chat interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>AlphaMind RAG</title></head>
    <body style="font-family: system-ui; max-width: 600px; margin: 2rem auto; padding: 1rem;">
        <h1>AlphaMind RAG API</h1>
        <p>This is the <strong>RAG/LLM API</strong> (port 8002). It has no chat UI here.</p>
        <ul>
            <li><strong>Use the main UI:</strong> <a href="http://localhost:8000">http://localhost:8000</a> — stock data and &quot;Ask the LLM&quot; chat.</li>
            <li><a href="/docs">API docs (Swagger)</a></li>
            <li><a href="/health">Health</a></li>
            <li><strong>Load vector DB:</strong> <code>POST /ingest</code> (or the server preloads on startup if PRELOAD_VECTOR_DB=1).</li>
        </ul>
    </body>
    </html>
    """


# ---------------------------------------------------------------------------
# Post-market: 3:15 PM IST → use vector DB; run ingest first if needed
# ---------------------------------------------------------------------------

def _now_ist() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def is_after_market_close() -> bool:
    """True if current time (IST) is at or after market close (default 3:15 PM)."""
    now = _now_ist()
    close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    return now >= close


def _post_market_ingest_date_file() -> Path:
    return Path(CHROMA_PERSIST_DIR) / ".last_post_market_ingest_date"


def ensure_post_market_ingest() -> None:
    """
    If it's after 3:15 PM IST and we haven't ingested today, run ingest for all stocks
    and store in vector DB. Idempotent per day.
    """
    if not is_after_market_close():
        return
    date_file = _post_market_ingest_date_file()
    today_str = _now_ist().strftime("%Y-%m-%d")
    try:
        if date_file.exists() and date_file.read_text().strip() == today_str:
            logger.info("Post-market ingest already done today, using vector DB.")
            return
    except Exception as e:
        logger.warning(f"Could not read post-market ingest date file: {e}")
    logger.info("Running post-market ingest: loading all stock details into vector DB...")
    stocks = get_stocks_for_post_market_ingest()
    added = ingest_stocks(stocks)
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    try:
        date_file.write_text(today_str)
    except Exception as e:
        logger.warning(f"Could not write post-market ingest date file: {e}")
    logger.info(f"Post-market ingest done: {added} documents for {len(stocks)} stocks.")


def get_context_from_vector_db(question: str) -> tuple[str, int, bool]:
    """
    Retrieve context from vector DB for the question. Returns (context, stocks_fetched, has_predictions).
    """
    top_k = min(RAG_TOP_K * 3, 20)  # more chunks for richer context
    results = query_documents(question, top_k=top_k)
    if not results:
        return "", 0, False
    parts = [r["content"] for r in results]
    context = "\n\n---\n\n".join(parts)
    stocks_seen = set()
    has_predictions = False
    for r in results:
        meta = r.get("metadata") or {}
        if meta.get("stock"):
            stocks_seen.add(meta["stock"])
        if meta.get("type") == "prediction" or "ALPHAMIND MODEL PREDICTION" in (r.get("content") or ""):
            has_predictions = True
    return context, len(stocks_seen), has_predictions


def _build_prediction_draft(context: str) -> str:
    """Build a draft (prices, trend, news, ratios, technicals) from context so the LLM can rewrite it without dropping data."""
    prices_lines = []
    direction_word = None
    return_pcts = []
    news_lines = []
    ratios_line = None
    technicals_line = None

    for block in context.split("\n\n---\n\n"):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("Current price:") and "INR" in s and not prices_lines:
                prices_lines.append(s)
            if s.startswith("Horizon ") and "day(s):" in s:
                prices_lines.append("  • " + s.rstrip("."))
                if "DOWN" in s.upper():
                    direction_word = "down"
                elif "UP" in s.upper():
                    direction_word = "up"
                if "return" in s.lower():
                    try:
                        idx = s.lower().find("return")
                        rest = s[idx:].replace(",", " ")
                        for part in rest.split():
                            if part.endswith("%"):
                                return_pcts.append(float(part[:-1]))
                    except (ValueError, IndexError):
                        pass
            if s.startswith("Recent news:"):
                for j in range(i + 1, len(lines)):
                    t = lines[j].strip()
                    if t and (t.startswith("-") or t.startswith("•") or not t.startswith("Valuation") and not t.startswith("Strengths") and not t.startswith("Concerns")):
                        if "Quarterly" not in t and "Stock data" not in t:
                            news_lines.append(t)
                    elif t.startswith("Valuation") or t.startswith("Strengths") or t.startswith("Concerns"):
                        break
            if s.startswith("Valuation ratios (fundamentals):"):
                ratios_line = s
            if s.startswith("Technical indicators (chart-based):"):
                technicals_line = s

    trend = "The model predicts a downward trend over the next 5 days."
    if direction_word == "up":
        trend = "The model predicts an upward trend over the next 5 days."
    if return_pcts:
        trend += f" Returns are approximately {min(return_pcts):.2f}% to {max(return_pcts):.2f}%."

    news_text = "\n".join(news_lines[:8]) if news_lines else "No recent news in the data."
    ratios_text = ratios_line if ratios_line else "No ratio data in the context."
    ratios_text += " PE reflects valuation; ROE and ROCE indicate profitability and capital efficiency."

    draft = (
        "**Predicted prices**\n"
        + "\n".join(prices_lines)
        + "\n\n**Trend**\n"
        + trend
        + "\n\n**News**\n"
        + news_text
        + "\n\n**Ratios**\n"
        + ratios_text
    )
    if technicals_line:
        draft += "\n\n**Technicals**\n" + technicals_line
    return draft


@app.get("/health")
def health():
    ollama_ok = ollama_health()
    return {
        "status": "healthy",
        "service": "alphamind-rag",
        "ollama_available": ollama_ok,
        "collection_documents": collection_count(),
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    """
    Ask a question in natural language. The service fetches fresh data from model_service and
    scraper_service (based on stocks mentioned in the question), then asks the local LLM to answer.
    Data is always current — nothing is read from RAG storage at query time.
    """
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    if not ollama_health():
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama run llama3.2",
        )

    # After 3:15 PM IST: try vector DB first; if no hits, fall back to live APIs. Before 3:15: live APIs only.
    if is_after_market_close():
        ensure_post_market_ingest()
        context, stocks_fetched, has_predictions = get_context_from_vector_db(question)
        if not context:
            context, stocks_fetched, has_predictions = fetch_fresh_context_for_question(question)
            if context:
                logger.info("After 3:15: no vector DB hits; used live API context for this query.")
    else:
        context, stocks_fetched, has_predictions = fetch_fresh_context_for_question(question)

    if not context:
        # Don't call the LLM when we have no data — return a clear, actionable message
        if is_after_market_close():
            answer = (
                "No relevant data found in the stored stock data for your query. "
                "Try asking about a specific stock by name (e.g. 'Technical details of TCS' or 'Reliance predictions')."
            )
        else:
            answer = (
                "No data could be fetched for your query. "
                "Make sure (1) the scraper service is running on port 8000 (stock data), "
                "and (2) the model service is running on port 8001 with a loaded model (predictions). "
                "Ask a question that mentions a stock by name (e.g. 'What are the predictions for Reliance?' or 'Reliance')."
            )
        return QueryResponse(
            answer=answer,
            stocks_fetched=stocks_fetched,
            collection_total=collection_count(),
        )

    # If user asked for prediction but we have no prediction data, tell them to start model service (no LLM)
    q_lower = question.lower()
    wants_prediction = (
        "predict" in q_lower or "prediction" in q_lower or "forecast" in q_lower
        or "next week" in q_lower or "next 5 days" in q_lower or "for the next" in q_lower
    )
    context_has_prediction = "ALPHAMIND MODEL PREDICTION" in context or "Model prediction for" in context
    if wants_prediction and (not has_predictions or not context_has_prediction):
        answer = (
            "For predictions, start the model service (port 8001). Stock data comes from this server. "
            "Prediction data is not available — start the model service and try again."
        )
        return QueryResponse(
            answer=answer,
            stocks_fetched=stocks_fetched,
            collection_total=collection_count(),
        )

    # Prediction question with data: build four-section draft from context, LLM rewrites for clarity (keeps numbers exact)
    if wants_prediction and context_has_prediction:
        draft = _build_prediction_draft(context)
        prompt = PREDICTION_REWRITE_PROMPT.format(draft=draft)
        try:
            answer = ollama_generate(prompt)
            # If LLM refused or said "not enough information" / "intervals are broad", return the draft so user gets the numbers
            refusal_phrases = (
                "do not have enough information",
                "intervals are quite broad",
                "cannot predict specific",
                "not have enough information",
            )
            if any(p in answer.lower() for p in refusal_phrases):
                answer = draft
            return QueryResponse(
                answer=answer,
                stocks_fetched=stocks_fetched,
                collection_total=collection_count(),
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    # Send full context to LLM for non-prediction questions (technicals, ratios, general)
    q = question.strip()
    if len(q) < 20 and any(c.isalpha() for c in q):
        question = f"What are the model predictions and stock data for {q}? Summarise briefly."
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        answer = ollama_generate(prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return QueryResponse(
        answer=answer,
        stocks_fetched=stocks_fetched,
        collection_total=collection_count(),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest | None = Body(None)):
    """
    Ingest data from model_service and scraper_service into the RAG vector store.
    Call this at least once (and optionally periodically) so /query has context.
    """
    if req is None:
        req = IngestRequest()
    stocks = req.stocks or DEFAULT_STOCKS
    try:
        added = ingest_stocks(stocks)
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")
    return IngestResponse(
        documents_added=added,
        message=f"Ingested {added} documents. Model service: {MODEL_SERVICE_URL}, Scraper: {SCRAPER_API_URL}.",
    )


@app.get("/ingest")
def ingest_get(stocks: str = Query(None, description="Comma-separated stock names (default: Reliance,TCS,...)")):
    """Same as POST /ingest but with optional ?stocks=Reliance,TCS,Infosys."""
    stock_list = [s.strip() for s in (stocks or "").split(",") if s.strip()] or DEFAULT_STOCKS
    try:
        added = ingest_stocks(stock_list)
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"documents_added": added, "stocks": stock_list}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8002")))
