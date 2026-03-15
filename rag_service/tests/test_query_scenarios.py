"""
Unit tests for /query with different prompts and scenarios.
Mocks external services (model, scraper, Ollama) so tests run without them.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

rag_root = Path(__file__).resolve().parent.parent
if str(rag_root) not in sys.path:
    sys.path.insert(0, str(rag_root))

from api.server import app, _build_prediction_draft


# Sample context strings for different scenarios
CONTEXT_EMPTY = ""
CONTEXT_NO_PREDICTION = """
Stock data for TCS (TCS.NS).
Current price: 2410.5 INR.
Valuation ratios (fundamentals): pe: 17.8, roe: 65.0, roce: 78.4.
Recent news:
  - TCS reports growth (Source: ET).
""".strip()

CONTEXT_WITH_PREDICTION = """
ALPHAMIND MODEL PREDICTION (from prediction API) for Reliance (RELIANCE.NS):
Current price: 1250.0 INR.
Horizon 1 day(s): direction DOWN, return -0.50%, predicted price 1243.75 INR (range 1228.53-1260.00).
Horizon 2 day(s): direction DOWN, return -0.52%, predicted price 1237.50 INR (range 1215.00-1265.00).
Horizon 3 day(s): direction DOWN, return -0.52%, predicted price 1231.25 INR (range 1200.00-1270.00).
Horizon 4 day(s): direction DOWN, return -0.55%, predicted price 1225.00 INR (range 1185.00-1275.00).
Horizon 5 day(s): direction DOWN, return -0.55%, predicted price 1218.75 INR (range 1170.00-1280.00).

---

Stock data for Reliance (RELIANCE.NS).
Current price: 1250.0 INR.
Technical indicators (chart-based): RSI: 45.0, SMA 50: 1280.0, SMA 200: 1300.0, MACD: -5.0, SMA crossover: bearish, RSI signal: neutral.
Valuation ratios (fundamentals): pe: 22.0, roe: 12.0, roce: 15.0.
Recent news:
  - Reliance announces new venture (Source: ET).
Strengths: Strong retail.
Concerns: Debt.
""".strip()


class TestBuildPredictionDraft:
    """Test draft building from context (no LLM)."""

    def test_draft_has_core_sections(self):
        draft = _build_prediction_draft(CONTEXT_WITH_PREDICTION)
        assert "**Predicted prices**" in draft
        assert "**Trend**" in draft
        assert "**News**" in draft
        assert "**Ratios**" in draft

    def test_draft_includes_technicals_when_in_context(self):
        draft = _build_prediction_draft(CONTEXT_WITH_PREDICTION)
        assert "**Technicals**" in draft
        assert "RSI" in draft
        assert "SMA" in draft or "MACD" in draft

    def test_draft_has_horizon_prices_and_ranges(self):
        draft = _build_prediction_draft(CONTEXT_WITH_PREDICTION)
        assert "1243.75" in draft
        assert "1228.53" in draft
        assert "1260.00" in draft
        assert "1250.0" in draft or "1250" in draft

    def test_draft_trend_down(self):
        draft = _build_prediction_draft(CONTEXT_WITH_PREDICTION)
        assert "downward" in draft.lower()

    def test_draft_with_no_news(self):
        ctx = "ALPHAMIND MODEL PREDICTION for X (X.NS):\nCurrent price: 100 INR.\nHorizon 1 day(s): direction UP, return 1.0%, predicted price 101 INR (range 99-103)."
        draft = _build_prediction_draft(ctx)
        assert "**News**" in draft
        assert "No recent news" in draft


@pytest.fixture
def client():
    return TestClient(app)


@patch("api.server.is_after_market_close", return_value=False)
@patch("api.server.collection_count", return_value=0)
@patch("api.server.ollama_health")
@patch("api.server.fetch_fresh_context_for_question")
class TestQueryScenarios:
    """Test /query with different prompts; Ollama and context are mocked."""

    def test_empty_question_returns_400(self, mock_fetch, mock_ollama_health, mock_collection_count, mock_after_market, client):
        mock_ollama_health.return_value = True
        r = client.post("/query", json={"question": ""})
        assert r.status_code == 400
        mock_fetch.assert_not_called()

    def test_no_context_returns_actionable_message(self, mock_fetch, mock_ollama_health, mock_collection_count, mock_after_market, client):
        mock_ollama_health.return_value = True
        mock_fetch.return_value = (CONTEXT_EMPTY, 5, False)
        with patch("api.server.ollama_generate"):
            r = client.post("/query", json={"question": "Tell me about XYZ stock"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "No data could be fetched" in data["answer"] or "Make sure" in data["answer"]

    def test_prediction_question_no_prediction_data_returns_start_model_service(
        self, mock_fetch, mock_ollama_health, mock_collection_count, mock_after_market, client
    ):
        mock_ollama_health.return_value = True
        mock_fetch.return_value = (CONTEXT_NO_PREDICTION, 1, False)
        r = client.post(
            "/query",
            json={"question": "Can you predict the price of TCS for the next 5 days?"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "Start the model service" in data["answer"] or "port 8001" in data["answer"]

    def test_prediction_question_with_data_llm_returns_good_answer(
        self, mock_fetch, mock_ollama_health, mock_collection_count, mock_after_market, client
    ):
        mock_ollama_health.return_value = True
        mock_fetch.return_value = (CONTEXT_WITH_PREDICTION, 1, True)
        with patch("api.server.ollama_generate") as mock_gen:
            mock_gen.return_value = (
                "**Predicted prices**\nCurrent price 1250 INR. Day 1: 1243.75 INR (1228-1260). "
                "**Trend** Down. **News** Reliance venture. **Ratios** PE 22."
            )
            r = client.post(
                "/query",
                json={"question": "Predict the price of Reliance for next 5 days."},
            )
        assert r.status_code == 200
        data = r.json()
        assert "1243.75" in data["answer"] or "1250" in data["answer"]
        mock_gen.assert_called_once()

    def test_prediction_question_llm_refusal_returns_draft(
        self, mock_fetch, mock_ollama_health, mock_collection_count, mock_after_market, client
    ):
        mock_ollama_health.return_value = True
        mock_fetch.return_value = (CONTEXT_WITH_PREDICTION, 1, True)
        with patch("api.server.ollama_generate") as mock_gen:
            mock_gen.return_value = (
                "Based on the context, I do not have enough information to predict specific prices. "
                "The intervals are quite broad."
            )
            r = client.post(
                "/query",
                json={"question": "Predict the price of Reliance for next 5 days."},
            )
        assert r.status_code == 200
        data = r.json()
        # Refusal detected -> draft should be used, so we get the actual numbers
        assert "**Predicted prices**" in data["answer"]
        assert "1243.75" in data["answer"]
        assert "1228.53" in data["answer"]
        assert "do not have enough information" not in data["answer"]

    def test_technicals_question_gets_full_context_to_llm(
        self, mock_fetch, mock_ollama_health, mock_collection_count, mock_after_market, client
    ):
        mock_ollama_health.return_value = True
        mock_fetch.return_value = (CONTEXT_NO_PREDICTION, 1, False)
        with patch("api.server.ollama_generate") as mock_gen:
            mock_gen.return_value = "RSI, SMA 50, etc. PE: 17.8, ROE: 65."
            r = client.post(
                "/query",
                json={"question": "Give me the technicals for TCS."},
            )
        assert r.status_code == 200
        mock_gen.assert_called_once()
        call_args = mock_gen.call_args[0][0]
        assert "Technical indicators" in call_args or "2410.5" in call_args

    def test_ollama_unavailable_returns_503(self, mock_fetch, mock_ollama_health, mock_collection_count, mock_after_market, client):
        mock_ollama_health.return_value = False
        r = client.post("/query", json={"question": "Predict TCS for 5 days."})
        assert r.status_code == 503
        mock_fetch.assert_not_called()


@patch("api.server.is_after_market_close", return_value=False)
@patch("api.server.collection_count", return_value=0)
@patch("api.server.ollama_health")
@patch("api.server.fetch_fresh_context_for_question")
class TestLlamaResponseContent:
    """Verify that when the LLM (Llama) returns a response, the API returns all expected info to the user."""

    def test_prediction_response_includes_all_four_sections_from_llm(
        self, mock_fetch, mock_ollama_health, mock_collection_count, mock_after_market, client
    ):
        mock_ollama_health.return_value = True
        mock_fetch.return_value = (CONTEXT_WITH_PREDICTION, 1, True)
        llm_answer = (
            "**Predicted prices**\nCurrent price 1250 INR. Day 1: 1243.75 INR (range 1228.53–1260). "
            "Day 2: 1237.50, Day 3: 1231.25, Day 4: 1225, Day 5: 1218.75.\n"
            "**Trend** Downward over the next 5 days.\n"
            "**News** Reliance announces new venture (ET).\n"
            "**Ratios** PE 22, ROE 12%, ROCE 15%."
        )
        with patch("api.server.ollama_generate") as mock_gen:
            mock_gen.return_value = llm_answer
            r = client.post(
                "/query",
                json={"question": "Predict Reliance for the next 5 days."},
            )
        assert r.status_code == 200
        data = r.json()
        answer = data["answer"]
        assert "**Predicted prices**" in answer
        assert "**Trend**" in answer
        assert "**News**" in answer
        assert "**Ratios**" in answer
        assert "1250" in answer
        assert "1243.75" in answer
        assert "1228.53" in answer or "1228" in answer
        assert "downward" in answer.lower() or "down" in answer.lower()
        assert "Reliance" in answer or "venture" in answer
        assert "22" in answer or "PE" in answer
        assert data.get("stocks_fetched") == 1
        assert "collection_total" in data

    def test_prediction_response_includes_exact_prices_and_ranges_from_llm(
        self, mock_fetch, mock_ollama_health, mock_collection_count, mock_after_market, client
    ):
        mock_ollama_health.return_value = True
        mock_fetch.return_value = (CONTEXT_WITH_PREDICTION, 1, True)
        llm_answer = (
            "Predicted prices: current 1250 INR; 1d 1243.75 (1228.53–1260), 2d 1237.50, 3d 1231.25, 4d 1225, 5d 1218.75. "
            "Trend: downward. News: Reliance new venture. Ratios: PE 22, ROE 12, ROCE 15."
        )
        with patch("api.server.ollama_generate") as mock_gen:
            mock_gen.return_value = llm_answer
            r = client.post(
                "/query",
                json={"question": "What are the predictions for Reliance?"},
            )
        assert r.status_code == 200
        data = r.json()
        answer = data["answer"]
        assert "1243.75" in answer
        assert "1237.50" in answer
        assert "1231.25" in answer
        assert "1228.53" in answer
        assert "1260" in answer
        assert "1218.75" in answer

    def test_technicals_response_includes_indicators_and_ratios_from_llm(
        self, mock_fetch, mock_ollama_health, mock_collection_count, mock_after_market, client
    ):
        mock_ollama_health.return_value = True
        mock_fetch.return_value = (CONTEXT_NO_PREDICTION, 1, False)
        llm_answer = (
            "TCS (2410.5 INR). Technical indicators: RSI 17.8, SMA 50, SMA 200, MACD. "
            "Valuation: PE 17.8, ROE 65.0, ROCE 78.4. Recent news: TCS reports growth (ET)."
        )
        with patch("api.server.ollama_generate") as mock_gen:
            mock_gen.return_value = llm_answer
            r = client.post(
                "/query",
                json={"question": "Give me technicals and ratios for TCS."},
            )
        assert r.status_code == 200
        data = r.json()
        answer = data["answer"]
        assert "2410.5" in answer
        assert "RSI" in answer
        assert "17.8" in answer
        assert "ROE" in answer or "65" in answer
        assert "TCS" in answer
        assert "news" in answer.lower() or "ET" in answer
        assert data.get("stocks_fetched") == 1
        assert "collection_total" in data

    def test_general_question_response_passes_through_llm_answer_unchanged(
        self, mock_fetch, mock_ollama_health, mock_collection_count, mock_after_market, client
    ):
        mock_ollama_health.return_value = True
        mock_fetch.return_value = (CONTEXT_NO_PREDICTION, 1, False)
        llm_answer = "TCS is at 2410.5 INR. PE 17.8, ROE 65, ROCE 78.4. News: TCS reports growth."
        with patch("api.server.ollama_generate") as mock_gen:
            mock_gen.return_value = llm_answer
            r = client.post(
                "/query",
                json={"question": "Summarise TCS."},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == llm_answer
        assert data["answer"].strip() != ""


class TestHealth:
    """Test /health endpoint."""

    def test_health_returns_200(self, client):
        with patch("api.server.ollama_health") as mock_health, patch(
            "api.server.collection_count", return_value=0
        ):
            mock_health.return_value = True
            r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "healthy"
        assert "ollama_available" in data
