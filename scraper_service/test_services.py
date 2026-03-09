"""
Unit tests for Stock Data Aggregator.

Run with: python -m pytest test_services.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import modules to test
from providers.yfinance_provider import YFinanceProvider
from providers.news_provider import NewsProvider
from providers.screener_provider import ScreenerProvider
from services.aggregator_service import AggregatorService
from models.stock_models import (
    MarketData, HistoricalPrice, Fundamentals, 
    NewsArticle, StockSummary, Market, Ratios, Financials,
    TechnicalIndicators
)
from utils.technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_sma,
    calculate_ema,
    calculate_vwap,
    get_rsi_signal,
    get_sma_crossover_signal
)


class TestYFinanceProvider:
    """Tests for YFinanceProvider."""
    
    @patch('providers.yfinance_provider.yf.Ticker')
    def test_get_stock_data_success(self, mock_ticker):
        """Test successful stock data retrieval."""
        # Mock yfinance response
        mock_stock = Mock()
        mock_stock.info = {
            'currentPrice': 2850.50,
            'marketCap': 1920000000000,
            'trailingPE': 24.5,
            'regularMarketPrice': 2850.50
        }
        mock_stock.history.return_value = MagicMock()
        
        mock_ticker.return_value = mock_stock
        
        provider = YFinanceProvider()
        result = provider.get_stock_data("RELIANCE.NS")
        
        assert result.ticker == "RELIANCE.NS"
        assert result.current_price == 2850.50
        assert result.market_cap == 1920000000000
        assert result.pe_ratio == 24.5
    
    @patch('providers.yfinance_provider.yf.Ticker')
    def test_get_stock_data_error_handling(self, mock_ticker):
        """Test error handling in stock data retrieval."""
        mock_ticker.side_effect = Exception("Connection error")
        
        provider = YFinanceProvider()
        result = provider.get_stock_data("INVALID.NS")
        
        # Should return default data on error
        assert result.ticker == "INVALID.NS"
        assert result.current_price == 0.0
        assert result.market_cap is None


class TestNewsProvider:
    """Tests for NewsProvider."""
    
    @patch('providers.news_provider.make_request')
    def test_get_news_success(self, mock_request):
        """Test successful news retrieval with relevance filtering."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'articles': [
                {
                    'title': 'Reliance Industries Reports Strong Q3 Results',
                    'description': 'Reliance stock surges on earnings beat',
                    'source': {'name': 'Test Source'},
                    'url': 'http://test.com',
                    'publishedAt': '2026-03-08T10:30:00Z'
                },
                {
                    'title': 'Unrelated Article About Animals',
                    'description': 'What animal has the strongest sense of smell?',
                    'source': {'name': 'Random Source'},
                    'url': 'http://random.com',
                    'publishedAt': '2026-03-08T10:30:00Z'
                }
            ]
        }
        mock_request.return_value = mock_response
        
        provider = NewsProvider(api_key="test_key")
        result = provider.get_news("Reliance", limit=10, ticker="RELIANCE.NS")
        
        # Should only return the relevant article, filtering out the irrelevant one
        assert len(result) == 1
        assert result[0].title == "Reliance Industries Reports Strong Q3 Results"
        assert result[0].source == "Test Source"
    
    def test_get_news_no_api_key(self):
        """Test news retrieval without API key."""
        provider = NewsProvider(api_key=None)
        result = provider.get_news("Reliance")
        
        assert result == []


class TestScreenerProvider:
    """Tests for ScreenerProvider."""
    
    @patch('providers.screener_provider.make_request')
    def test_get_fundamentals(self, mock_request):
        """Test fundamental data retrieval and parsing."""
        mock_response = Mock()
        html_content = """
        <html>
            <table>
                <tr><td>P/E</td><td>24.5</td></tr>
                <tr><td>ROCE</td><td>12.8</td></tr>
                <tr><td>Debt</td><td>0.15</td></tr>
            </table>
        </html>
        """
        mock_response.content = html_content.encode('utf-8')
        mock_request.return_value = mock_response
        
        provider = ScreenerProvider()
        result = provider.get_fundamentals("RELIANCE")
        
        assert result is not None
        assert isinstance(result, Fundamentals)
    
    def test_parse_float(self):
        """Test float parsing utility."""
        provider = ScreenerProvider()
        
        assert provider._parse_float("24.5") == 24.5
        assert provider._parse_float("24.5%") == 24.5
        assert provider._parse_float("1,000.50") == 1000.5
        assert provider._parse_float("N/A") is None
        assert provider._parse_float("-") is None


class TestAggregatorService:
    """Tests for AggregatorService."""
    
    @patch('services.aggregator_service.YFinanceProvider')
    @patch('services.aggregator_service.NewsProvider')
    @patch('services.aggregator_service.ScreenerProvider')
    def test_get_stock_summary(self, mock_screener, mock_news, mock_yfinance):
        """Test stock summary aggregation."""
        # Setup mocks
        mock_yf_instance = Mock()
        mock_yf_instance.get_stock_data.return_value = MarketData(
            ticker="RELIANCE.NS",
            current_price=2850.50
        )
        mock_yfinance.return_value = mock_yf_instance
        
        mock_news_instance = Mock()
        mock_news_instance.get_news.return_value = []
        mock_news.return_value = mock_news_instance
        
        mock_screener_instance = Mock()
        mock_screener_instance.get_all_company_data.return_value = {
            "fundamentals": Fundamentals(),
            "quarterly_results": [],
            "profit_loss": [],
            "balance_sheet": [],
            "cash_flow": [],
            "shareholding": [],
            "peer_comparison": []
        }
        mock_screener.return_value = mock_screener_instance
        
        # Test aggregation
        service = AggregatorService(news_api_key="test_key")
        result = service.get_stock_summary("RELIANCE.NS", "Reliance")
        
        assert isinstance(result, StockSummary)
        assert result.ticker == "RELIANCE.NS"
        assert result.company == "Reliance"
        assert result.market.price == 2850.50


class TestModels:
    """Tests for Pydantic models."""
    
    def test_market_data_model(self):
        """Test MarketData model validation."""
        data = MarketData(
            ticker="RELIANCE.NS",
            current_price=2850.50,
            market_cap=1920000000000,
            pe_ratio=24.5
        )
        
        assert data.ticker == "RELIANCE.NS"
        assert data.current_price == 2850.50
    
    def test_news_article_model(self):
        """Test NewsArticle model validation."""
        article = NewsArticle(
            title="Test Article",
            description="Test description",
            source="Test Source",
            url="http://test.com",
            published_at="2026-03-08T10:30:00Z"
        )
        
        assert article.title == "Test Article"
        assert article.source == "Test Source"
    
    def test_stock_summary_model(self):
        """Test StockSummary model validation."""
        summary = StockSummary(
            ticker="RELIANCE.NS",
            company="Reliance",
            market=Market(
                price=2850.50
            ),
            ratios=Ratios(),
            financials=Financials()
        )
        
        assert summary.ticker == "RELIANCE.NS"
        assert summary.company == "Reliance"
        assert isinstance(summary.timestamp, datetime)


class TestTechnicalIndicators:
    """Tests for technical indicator calculations."""
    
    # Sample price data for testing (realistic stock prices)
    SAMPLE_PRICES = [
        100.0, 101.5, 102.0, 100.5, 99.0, 98.5, 99.5, 101.0, 102.5, 103.0,
        104.0, 103.5, 102.0, 101.0, 100.0, 99.0, 98.0, 99.5, 101.0, 102.0,
        103.0, 104.5, 105.0, 106.0, 105.5, 104.0, 103.0, 102.5, 103.5, 104.0
    ]
    
    # Extended price data for SMA tests
    EXTENDED_PRICES = SAMPLE_PRICES * 10  # 300 data points
    
    def test_calculate_sma_basic(self):
        """Test basic SMA calculation."""
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        
        sma_3 = calculate_sma(prices, 3)
        assert sma_3 == 13.0  # (12 + 13 + 14) / 3
        
        sma_5 = calculate_sma(prices, 5)
        assert sma_5 == 12.0  # (10 + 11 + 12 + 13 + 14) / 5
    
    def test_calculate_sma_insufficient_data(self):
        """Test SMA returns None with insufficient data."""
        prices = [10.0, 11.0, 12.0]
        
        sma_50 = calculate_sma(prices, 50)
        assert sma_50 is None
    
    def test_calculate_ema_basic(self):
        """Test basic EMA calculation."""
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        
        ema = calculate_ema(prices, 3)
        assert ema is not None
        # EMA should be close to but not exactly the same as SMA
        assert 13.0 < ema < 15.0
    
    def test_calculate_rsi_basic(self):
        """Test RSI calculation with sample data."""
        rsi = calculate_rsi(self.SAMPLE_PRICES, period=14)
        
        assert rsi is not None
        assert 0 <= rsi <= 100
    
    def test_calculate_rsi_overbought(self):
        """Test RSI detects overbought conditions."""
        # Steadily rising prices should give high RSI
        rising_prices = [100.0 + i * 2 for i in range(20)]
        
        rsi = calculate_rsi(rising_prices, period=14)
        assert rsi is not None
        assert rsi > 70  # Should be overbought
    
    def test_calculate_rsi_oversold(self):
        """Test RSI detects oversold conditions."""
        # Steadily falling prices should give low RSI
        falling_prices = [200.0 - i * 2 for i in range(20)]
        
        rsi = calculate_rsi(falling_prices, period=14)
        assert rsi is not None
        assert rsi < 30  # Should be oversold
    
    def test_calculate_rsi_insufficient_data(self):
        """Test RSI returns None with insufficient data."""
        prices = [100.0, 101.0, 102.0]
        
        rsi = calculate_rsi(prices, period=14)
        assert rsi is None
    
    def test_calculate_macd_basic(self):
        """Test MACD calculation."""
        # Need at least 26 + 9 = 35 data points for MACD
        prices = self.SAMPLE_PRICES + [105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
        
        macd, signal, histogram = calculate_macd(prices)
        
        assert macd is not None
        assert signal is not None
        assert histogram is not None
        assert histogram == round(macd - signal, 4)
    
    def test_calculate_macd_insufficient_data(self):
        """Test MACD returns None with insufficient data."""
        prices = [100.0, 101.0, 102.0]
        
        macd, signal, histogram = calculate_macd(prices)
        
        assert macd is None
        assert signal is None
        assert histogram is None
    
    def test_calculate_vwap_basic(self):
        """Test VWAP calculation."""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        highs = [101.0, 102.0, 103.0, 104.0, 105.0]
        lows = [99.0, 100.0, 101.0, 102.0, 103.0]
        volumes = [1000, 1500, 2000, 1800, 1200]
        
        vwap = calculate_vwap(prices, highs, lows, volumes)
        
        assert vwap is not None
        assert 99.0 < vwap < 105.0  # VWAP should be within price range
    
    def test_calculate_vwap_empty_data(self):
        """Test VWAP returns None with empty data."""
        vwap = calculate_vwap([], [], [], [])
        assert vwap is None
    
    def test_calculate_vwap_zero_volume(self):
        """Test VWAP handles zero total volume."""
        prices = [100.0, 101.0]
        highs = [101.0, 102.0]
        lows = [99.0, 100.0]
        volumes = [0, 0]
        
        vwap = calculate_vwap(prices, highs, lows, volumes)
        assert vwap is None
    
    def test_get_rsi_signal(self):
        """Test RSI signal interpretation."""
        assert get_rsi_signal(25) == "oversold"
        assert get_rsi_signal(75) == "overbought"
        assert get_rsi_signal(50) == "neutral"
        assert get_rsi_signal(None) is None
    
    def test_get_sma_crossover_signal(self):
        """Test SMA crossover signal."""
        assert get_sma_crossover_signal(110, 100) == "bullish"  # Golden cross
        assert get_sma_crossover_signal(90, 100) == "bearish"   # Death cross
        assert get_sma_crossover_signal(100, 100) == "neutral"
        assert get_sma_crossover_signal(None, 100) is None
        assert get_sma_crossover_signal(100, None) is None
    
    def test_technical_indicators_model(self):
        """Test TechnicalIndicators dataclass."""
        indicators = TechnicalIndicators(
            rsi=55.5,
            macd=1.234,
            macd_signal=1.0,
            macd_histogram=0.234,
            sma_50=102.5,
            sma_200=98.3,
            vwap=101.0,
            rsi_signal="neutral",
            sma_crossover="bullish"
        )
        
        assert indicators.rsi == 55.5
        assert indicators.macd == 1.234
        assert indicators.sma_50 == 102.5
        assert indicators.sma_200 == 98.3
        assert indicators.sma_crossover == "bullish"
    
    def test_market_with_technical_indicators(self):
        """Test Market model includes technical indicators."""
        indicators = TechnicalIndicators(rsi=60.0, sma_50=100.0)
        market = Market(
            price=105.0,
            volume=1000000,
            technical_indicators=indicators
        )
        
        assert market.price == 105.0
        assert market.technical_indicators is not None
        assert market.technical_indicators.rsi == 60.0
        assert market.technical_indicators.sma_50 == 100.0


# Run tests with: python -m pytest test_services.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
