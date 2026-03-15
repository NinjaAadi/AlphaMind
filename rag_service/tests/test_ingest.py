"""
Unit tests for ingest module: stock detection from prompts, document building from prediction/summary.
"""
import pytest

# Run from rag_service: pytest tests/ -v
import sys
from pathlib import Path
rag_root = Path(__file__).resolve().parent.parent
if str(rag_root) not in sys.path:
    sys.path.insert(0, str(rag_root))

from ingest import (
    stocks_mentioned_in_question,
    build_document_from_prediction,
    build_document_from_stock_summary,
    DEFAULT_STOCKS,
)


class TestStocksMentionedInQuestion:
    """Test stock extraction from different user prompts."""

    def test_predict_tcs(self):
        assert "TCS" in stocks_mentioned_in_question("Can you predict the price of TCS for next 5 days?")

    def test_reliance_forecast(self):
        assert "Reliance" in stocks_mentioned_in_question("Predict the price of Reliance for next 5 days.")

    def test_technicals_infosys(self):
        assert "Infosys" in stocks_mentioned_in_question("Give me the technicals for Infosys.")

    def test_multiple_stocks(self):
        result = stocks_mentioned_in_question("Compare TCS and Reliance")
        assert "TCS" in result
        assert "Reliance" in result

    def test_alias_hdfc_bank(self):
        assert "HDFCBANK" in stocks_mentioned_in_question("What about HDFC Bank?")

    def test_alias_infy(self):
        assert "Infosys" in stocks_mentioned_in_question("Show Infy ratios")

    def test_empty_question_returns_defaults(self):
        result = stocks_mentioned_in_question("")
        assert len(result) <= 5
        assert all(s in DEFAULT_STOCKS for s in result)

    def test_unrelated_question_returns_defaults(self):
        result = stocks_mentioned_in_question("What is the weather today?")
        assert len(result) <= 5


class TestBuildDocumentFromPrediction:
    """Test prediction API response -> text for context."""

    def test_full_prediction(self):
        data = {
            "ticker": "TCS.NS",
            "company": "TCS",
            "current_price": 2410.5,
            "predictions": [
                {"horizon": 1, "direction": "DOWN", "predicted_return": -0.48, "predicted_price": 2398.87, "price_low": 2354.44, "price_high": 2447.24},
                {"horizon": 2, "direction": "DOWN", "predicted_return": -0.52, "predicted_price": 2387.34, "price_low": 2296.15, "price_high": 2486.48},
            ],
        }
        doc = build_document_from_prediction(data)
        assert "ALPHAMIND MODEL PREDICTION" in doc
        assert "TCS" in doc
        assert "Current price: 2410.5 INR" in doc
        assert "Horizon 1 day(s):" in doc
        assert "Horizon 2 day(s):" in doc
        assert "2398.87" in doc
        assert "2354.44" in doc
        assert "2447.24" in doc

    def test_empty_predictions(self):
        data = {"ticker": "X.NS", "company": "X", "current_price": 100, "predictions": []}
        doc = build_document_from_prediction(data)
        assert "ALPHAMIND MODEL PREDICTION" in doc
        assert "Current price: 100 INR" in doc
        assert "Horizon" not in doc or "day(s):" not in doc


class TestBuildDocumentFromStockSummary:
    """Test scraper stock summary -> text for context."""

    def test_with_technicals_and_ratios(self):
        data = {
            "ticker": "TCS.NS",
            "company": "TCS",
            "market": {
                "price": 2410.5,
                "open": 2400,
                "high": 2420,
                "low": 2390,
                "volume": 1_000_000,
                "market_cap": 9e12,
                "technical_indicators": {
                    "rsi": 19.0,
                    "sma_50": 2908.1,
                    "sma_200": 3055.92,
                    "macd": -129.8,
                    "sma_crossover": "bearish",
                    "rsi_signal": "oversold",
                },
            },
            "ratios": {"pe": 17.8, "roe": 65.0, "roce": 78.4, "eps": 135.0},
            "news": [{"title": "TCS Q4 results", "description": "Strong growth", "source": "ET"}],
            "pros": ["Strong brand"],
            "cons": ["Competition"],
        }
        doc = build_document_from_stock_summary(data)
        assert "Stock data for TCS" in doc
        assert "Current price: 2410.5 INR" in doc
        assert "Technical indicators (chart-based):" in doc
        assert "RSI: 19.0" in doc
        assert "SMA 50: 2908.1" in doc
        assert "Valuation ratios (fundamentals):" in doc
        assert "pe: 17.8" in doc
        assert "Recent news:" in doc
        assert "TCS Q4 results" in doc
        assert "Strengths:" in doc
        assert "Concerns:" in doc

    def test_minimal_summary(self):
        data = {"ticker": "X.NS", "company": "X", "market": {"price": 100}, "ratios": {}, "news": []}
        doc = build_document_from_stock_summary(data)
        assert "Stock data for X" in doc
        assert "100" in doc
