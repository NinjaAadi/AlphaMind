"""
Ingest data from model_service and scraper_service into the RAG vector store.

Run this after model_service and scraper_service are up. Optionally run periodically
to refresh context (e.g. new predictions).
"""
import logging
import uuid
from typing import List

import httpx

from pathlib import Path

from config import MODEL_SERVICE_URL, SCRAPER_API_URL, INGEST_STOCKS_FILE, SINGLE_STOCK
from vector_store import add_documents

logger = logging.getLogger(__name__)

# Default stocks to ingest (can be overridden by env or CLI)
DEFAULT_STOCKS = [
    "Reliance", "TCS", "HDFCBANK", "Infosys", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
]

# Names to match in user questions (lowercase) -> API-friendly name
STOCK_ALIASES = {
    "reliance": "Reliance",
    "tcs": "TCS",
    "hdfc": "HDFCBANK",
    "hdfc bank": "HDFCBANK",
    "icici": "ICICIBANK",
    "icici bank": "ICICIBANK",
    "infosys": "Infosys",
    "infy": "Infosys",
    "hindunilvr": "HINDUNILVR",
    "hul": "HINDUNILVR",
    "itc": "ITC",
    "sbin": "SBIN",
    "sbi": "SBIN",
    "bharti": "BHARTIARTL",
    "bharti airtel": "BHARTIARTL",
    "airtel": "BHARTIARTL",
    "kotak": "KOTAKBANK",
    "kotak bank": "KOTAKBANK",
    "axis bank": "AXISBANK",
    "axis": "AXISBANK",
    "asian paint": "ASIANPAINT",
    "asianpaint": "ASIANPAINT",
    "maruti": "MARUTI",
    "sun pharma": "SUNPHARMA",
    "sunpharma": "SUNPHARMA",
    "titan": "TITAN",
    "wipro": "WIPRO",
    "hcl": "HCLTECH",
    "hcl tech": "HCLTECH",
    "lt": "LT",
    "larsen": "LT",
    "bajaj finance": "BAJFINANCE",
    "bajfinance": "BAJFINANCE",
}


def stocks_mentioned_in_question(question: str) -> List[str]:
    """
    Infer which stocks the user is asking about from the question text.
    Returns a list of unique stock names (API-friendly) to fetch. If none matched, returns default list so we still fetch some context.
    """
    q = (question or "").lower().strip()
    if not q:
        return list(DEFAULT_STOCKS)[:5]
    seen = set()
    out = []
    for alias, name in STOCK_ALIASES.items():
        if alias in q and name not in seen:
            seen.add(name)
            out.append(name)
    if not out:
        return list(DEFAULT_STOCKS)[:5]
    return out


def fetch_prediction(stock: str) -> dict | None:
    """Fetch prediction for one stock from model_service."""
    try:
        with httpx.Client(timeout=20) as client:
            r = client.get(f"{MODEL_SERVICE_URL.rstrip('/')}/predict", params={"stock": stock})
            if r.status_code == 200:
                return r.json()
    except Exception as e:
        logger.warning(f"Prediction fetch failed for {stock}: {e}")
    return None


def fetch_stock_summary(stock: str) -> dict | None:
    """Fetch stock summary (market, news, fundamentals) from scraper_service."""
    try:
        with httpx.Client(timeout=15) as client:
            r = client.get(f"{SCRAPER_API_URL.rstrip('/')}/stock", params={"company": stock})
            if r.status_code == 200:
                return r.json()
    except Exception as e:
        logger.warning(f"Scraper fetch failed for {stock}: {e}")
    return None


def build_document_from_prediction(data: dict) -> str:
    """Turn a prediction response into a single text chunk for RAG."""
    ticker = data.get("ticker", "")
    company = data.get("company", "")
    price = data.get("current_price")
    preds = data.get("predictions", [])
    lines = [
        f"ALPHAMIND MODEL PREDICTION (from prediction API) for {company} ({ticker}):",
        f"Current price: {price} INR." if price else "",
    ]
    for p in preds:
        h = p.get("horizon", 0)
        direction = p.get("direction", "")
        ret = p.get("predicted_return", 0)
        pprice = p.get("predicted_price", 0)
        low = p.get("price_low", 0)
        high = p.get("price_high", 0)
        lines.append(
            f"Horizon {h} day(s): direction {direction}, return {ret:.2f}%, "
            f"predicted price {pprice:.2f} INR (range {low:.2f}-{high:.2f})."
        )
    return "\n".join(l for l in lines if l).strip()


def build_document_from_stock_summary(data: dict) -> str:
    """Turn scraper stock summary into a rich text chunk (market, ratios, technicals, news, pros/cons)."""
    ticker = data.get("ticker", "")
    company = data.get("company", "")
    market = data.get("market", {}) or {}
    ratios = data.get("ratios", {}) or {}
    news = data.get("news", []) or []
    pros = data.get("pros", []) or []
    cons = data.get("cons", []) or []
    financials = data.get("financials", {}) or {}
    lines = [f"Stock data for {company} ({ticker})."]

    # Market: price, OHLC, volume, market cap
    p = market.get("price")
    if p is not None:
        lines.append(f"Current price: {p} INR.")
    o, h, l, v = market.get("open"), market.get("high"), market.get("low"), market.get("volume")
    if any(x is not None for x in (o, h, l, v)):
        parts = [f"Open: {o}", f"High: {h}", f"Low: {l}", f"Volume: {v}"]
        lines.append(" ".join(p for p in parts if "None" not in p))
    mc = market.get("market_cap")
    if mc is not None:
        lines.append(f"Market cap: {mc}.")

    # Technical indicators (chart-based: RSI, SMA, MACD — distinct from valuation ratios below)
    ti = market.get("technical_indicators")
    if isinstance(ti, dict):
        rsi = ti.get("rsi")
        sma50 = ti.get("sma_50")
        sma200 = ti.get("sma_200")
        macd = ti.get("macd")
        crossover = ti.get("sma_crossover")
        rsi_sig = ti.get("rsi_signal")
        parts = []
        if rsi is not None:
            parts.append(f"RSI: {rsi}")
        if sma50 is not None:
            parts.append(f"SMA 50: {sma50}")
        if sma200 is not None:
            parts.append(f"SMA 200: {sma200}")
        if macd is not None:
            parts.append(f"MACD: {macd}")
        if crossover:
            parts.append(f"SMA crossover: {crossover}")
        if rsi_sig:
            parts.append(f"RSI signal: {rsi_sig}")
        if parts:
            lines.append("Technical indicators (chart-based): " + ", ".join(parts) + ".")

    # Valuation ratios (fundamentals — PE, ROE, ROCE, etc.; not the same as technical indicators)
    if ratios:
        r_parts = []
        for k in ("pe", "roe", "roce", "book_value", "dividend_yield", "eps", "debt_to_equity", "face_value", "high_low"):
            val = ratios.get(k)
            if val is not None:
                r_parts.append(f"{k}: {val}")
        if r_parts:
            lines.append("Valuation ratios (fundamentals): " + ", ".join(r_parts) + ".")

    # Pros and cons
    if pros:
        lines.append("Strengths: " + "; ".join(pros[:8]))
    if cons:
        lines.append("Concerns: " + "; ".join(cons[:8]))

    # News (title + short description)
    if news:
        lines.append("Recent news:")
        for n in news[:5]:
            title = (n.get("title") or "").strip()
            desc = (n.get("description") or "").strip()[:150]
            if title:
                lines.append(f"  - {title}" + (f". {desc}" if desc else ""))
            source = n.get("source")
            if source:
                lines[-1] = lines[-1] + f" (Source: {source})"

    # Quarterly results
    qr = financials.get("quarterly_results") if isinstance(financials, dict) else []
    if isinstance(qr, list) and len(qr) > 0:
        lines.append("Quarterly results: available in data.")
    return "\n".join(l for l in lines if l).strip()


def load_stocks_from_file(path: str | Path | None = None) -> List[str]:
    """Load stock symbols/names from a file (one per line). Returns empty list if file missing or unreadable."""
    p = Path(path or INGEST_STOCKS_FILE)
    if not p.exists():
        logger.warning(f"Stocks file not found: {p}")
        return []
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        return [s.strip() for s in lines if s.strip()]
    except Exception as e:
        logger.warning(f"Failed to read stocks file {p}: {e}")
        return []


def get_stocks_for_post_market_ingest() -> List[str]:
    """Stocks to ingest after market close (for vector DB). SINGLE_STOCK overrides for testing."""
    if SINGLE_STOCK:
        return [SINGLE_STOCK]
    stocks = load_stocks_from_file()
    return stocks if stocks else list(DEFAULT_STOCKS)


def ingest_stocks(stocks: List[str], log_progress: bool = False) -> int:
    """
    For each stock: fetch prediction + stock summary, build documents, add to vector store.
    Returns number of documents added.
    If log_progress is True, logs progress after each stock (e.g. for startup preload).
    """
    doc_ids = []
    documents = []
    metadatas = []
    total = len([s for s in stocks if s and s.strip()])
    done = 0

    for stock in stocks:
        stock = stock.strip()
        if not stock:
            continue
        pred = fetch_prediction(stock)
        summary = fetch_stock_summary(stock)
        if pred:
            doc_ids.append(f"pred_{stock}_{uuid.uuid4().hex[:8]}")
            documents.append(build_document_from_prediction(pred))
            metadatas.append({"source": "model", "stock": stock, "type": "prediction"})
        if summary:
            doc_ids.append(f"summary_{stock}_{uuid.uuid4().hex[:8]}")
            documents.append(build_document_from_stock_summary(summary))
            metadatas.append({"source": "scraper", "stock": stock, "type": "summary"})
        done += 1
        if log_progress:
            logger.info(f"Preload progress: {done}/{total} — {stock} (pred={'✓' if pred else '-'} summary={'✓' if summary else '-'})")

    if documents:
        add_documents(doc_ids, documents, metadatas)
    return len(documents)


def fetch_fresh_context_for_question(question: str) -> tuple[str, int, bool]:
    """
    At query time: infer stocks from the question, call model + scraper APIs for fresh data,
    and return (context string, number of stocks fetched, whether any prediction data was found).
    Data is always current (no RAG/storage).
    """
    stocks = stocks_mentioned_in_question(question)
    parts = []
    has_predictions = False
    for stock in stocks:
        pred = fetch_prediction(stock)
        summary = fetch_stock_summary(stock)
        if pred:
            parts.append(build_document_from_prediction(pred))
            has_predictions = True
        if summary:
            parts.append(build_document_from_stock_summary(summary))
    context = "\n\n---\n\n".join(parts) if parts else ""
    return context, len(stocks), has_predictions
