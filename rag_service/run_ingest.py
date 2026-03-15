#!/usr/bin/env python3
"""
Cron job script: fetch all stock data from model_service and scraper_service,
then store it in the RAG vector database.

Run daily at 3:16 AM IST (after market close) so RAG has the latest data for the next day.

Usage:
  # From repo root or rag_service dir, with venv activated:
  python -m rag_service.run_ingest
  # Or with explicit stocks file:
  INGEST_STOCKS_FILE=/path/to/stocks.txt python -m rag_service.run_ingest

Crontab (3:16 AM IST) — use full path to current (project) directory, e.g.:
  16 3 * * * /Users/aaditya.pal/Documents/AlphaMind/rag_service/run_ingest_cron.sh >> /tmp/rag_ingest.log 2>&1
Or inline:
  16 3 * * * cd /Users/aaditya.pal/Documents/AlphaMind/rag_service && ./venv/bin/python -m run_ingest >> /tmp/rag_ingest.log 2>&1
"""
import logging
import os
import sys
from pathlib import Path

# Ensure rag_service is on path when run as cron
_rag_dir = Path(__file__).resolve().parent
if str(_rag_dir) not in sys.path:
    sys.path.insert(0, str(_rag_dir))
os.chdir(_rag_dir)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_ingest")


def main() -> int:
    from ingest import ingest_stocks, load_stocks_from_file, DEFAULT_STOCKS
    from config import INGEST_STOCKS_FILE

    stocks = load_stocks_from_file(INGEST_STOCKS_FILE)
    if not stocks:
        logger.warning("No stocks from file; using DEFAULT_STOCKS")
        stocks = DEFAULT_STOCKS
    logger.info(f"Ingesting {len(stocks)} stocks into RAG vector DB")
    try:
        added = ingest_stocks(stocks)
        logger.info(f"RAG ingest done: {added} documents stored")
        return 0
    except Exception as e:
        logger.exception(f"RAG ingest failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
