"""RAG service configuration."""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_data"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "alphamind_rag")
# Optional: file with one stock symbol/name per line for cron ingest (default: model_service/training_stocks.txt)
INGEST_STOCKS_FILE = os.getenv("INGEST_STOCKS_FILE", str(BASE_DIR.parent / "model_service" / "training_stocks.txt"))

# Testing: single-stock mode — set SINGLE_STOCK or TEST_SINGLE_STOCK (e.g. TCS) to ingest/use only that stock
SINGLE_STOCK = (os.getenv("SINGLE_STOCK") or os.getenv("TEST_SINGLE_STOCK") or "").strip()

# External services (must be running)
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001")
SCRAPER_API_URL = os.getenv("SCRAPER_API_URL", "http://localhost:8000")

# Ollama (free local LLM - run: ollama run llama3.2 for faster replies)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "90"))

# RAG
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
# Post-market: after this time (IST), use vector DB instead of live APIs; ingest runs first if needed
MARKET_CLOSE_HOUR = int(os.getenv("MARKET_CLOSE_HOUR", "15"))   # 3 PM
MARKET_CLOSE_MINUTE = int(os.getenv("MARKET_CLOSE_MINUTE", "15"))  # 3:15 PM
TIMEZONE = os.getenv("TIMEZONE", "Asia/Kolkata")
# Load vector DB on server startup (so it is ready for queries after 3:15). Set to 0 to skip.
PRELOAD_VECTOR_DB = os.getenv("PRELOAD_VECTOR_DB", "1").strip().lower() in ("1", "true", "yes")
RAG_PROMPT_TEMPLATE = """You are AlphaMind's assistant. The context below contains our system's output: model predictions and stock data. Your job is to restate that data for the user. Never refuse or say "I cannot predict" or "I cannot provide exact prices" — the exact numbers are in the context; copy them out.

Rules:
- Use ONLY the context below. Do not make up numbers.
- Do NOT add a preamble (e.g. "To provide a summary...", "I will copy the relevant data from the context") or a closing disclaimer (e.g. "Please note that I've only copied the exact numbers...", "without making any further analysis"). Start directly with the data (headings and bullet points). End after the last data point.
- When the user asks for predictions, forecast, or "exact price numbers" for the next 5 days: The context has lines like "Horizon 1 day(s): direction X, return Y%, predicted price Z INR (range A-B)." You MUST list each day with the exact predicted price and range from those lines. Example format:
  • Day 1: predicted price X INR (range A–B INR), direction X, return Y%
  • Day 2: ...
  • Day 5: ...
  Also give current price from "Current price: ... INR" in the context. Do NOT say "I cannot predict exact prices" — they are in the Horizon lines; copy them.
- When the user asks for technicals: list RSI, SMA 50, SMA 200, MACD, crossover, RSI signal from "Technical indicators (chart-based):" in the context. PE/ROE/ROCE are ratios, not technicals.
- When the user asks for ratios: list PE, ROE, ROCE and others from "Valuation ratios (fundamentals):" in the context.
- For other questions: summarize the context and include the relevant numbers.

Context:
{context}

Question: {question}

Answer (start directly with the data; no preamble or disclaimer):"""

# When user asks for prediction: send FULL context so the LLM gives prices, trend, news, and ratios
PREDICTION_FULL_PROMPT = """You are AlphaMind's assistant. The context below contains model predictions and stock data (news, ratios). Use only this context. Do not refuse.

Your answer MUST have exactly four sections with these headings. Do not skip any section.

**Predicted prices**
List the current price and, for each of the next 5 days, the predicted price in INR and the price range. Copy from "Current price:" and "Horizon X day(s):" in the context.

**Trend**
In 1–2 sentences, explain the trend: is the stock predicted to go up or down over the next 5 days, and by roughly how much? Use the direction and return % from the Horizon lines.

**News**
Summarise the "Recent news:" from the context. What is in the news that might matter for this stock or the trend? If there is no "Recent news" in the context, write "No recent news in the data."

**Ratios**
The context may have "Valuation ratios (fundamentals):" with PE, ROE, ROCE, etc. In plain language, explain what these numbers mean for this stock (e.g. valuation, profitability). If no ratios are in the context, write "No ratio data in the context."

Context:
{context}

Question: {question}

Answer. Use the four headings above and fill every section:"""

# Hybrid: we build sections from data; LLM rewrites for clarity (keeps numbers exact)
PREDICTION_REWRITE_PROMPT = """Below is the exact prediction and stock data already structured in sections. Rewrite it as a clear, user-friendly answer.

Rules:
- Do NOT add a preamble (e.g. "To provide a summary...", "I will copy the relevant data") or a closing disclaimer (e.g. "Please note that I've only copied...", "without making any further analysis"). Start directly with the first section (e.g. **Predicted prices** or **Technicals**). End after the last section.
- Keep every number and every price range exactly as given. Do not say "below X", "no range", "intervals are quite broad", or "I do not have enough information to predict specific price" — the draft already has the exact predicted price and range for each day; list them.
- Keep all sections present in the draft: Predicted prices, Trend, News, Ratios. If **Technicals** is present, include it (RSI, SMA 50/200, MACD, crossover, RSI signal) — especially when the user asked for "technical details", "technicals", or "all details".
- For Predicted prices, output the current price and then Day 1 to Day 5 with each day's predicted price in INR and range (low–high).
- You may shorten or rephrase sentences for readability, but do not drop any of the data.

Draft:
{draft}

Rewrite for the user. List the predicted price and range for each of the 5 days; include technicals if present and the user asked for details; do not refuse or say information is insufficient:"""
