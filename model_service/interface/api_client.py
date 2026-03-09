"""
API client implementation for fetching stock data from the scraper service.
"""

import os
import requests
import logging
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from .base import AbstractDataFetcher

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Default API URL from environment
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")


class StockAPIClient(AbstractDataFetcher):
    """
    Simple HTTP client for fetching stock data from the FastAPI scraper service.
    
    Example:
        client = StockAPIClient()  # Uses API_URL from .env
        data = client.fetch("RELIANCE.NS", "Reliance")
    """
    
    def __init__(
        self, 
        base_url: Optional[str] = None,
        timeout: int = 30,
        retries: int = 3
    ):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the scraper service (defaults to API_URL from .env)
            timeout: Request timeout in seconds
            retries: Number of retry attempts for failed requests
        """
        url = base_url or DEFAULT_API_URL
        # Strip /stock endpoint if present
        self.base_url = url
        self.timeout = timeout
        self.retries = retries
        
    def fetch(self, ticker: str, company: str) -> Dict[str, Any]:
        """
        Fetch stock data from the API.
        
        Args:
            ticker: Stock ticker symbol (e.g., "RELIANCE.NS")
            company: Company name (e.g., "Reliance")
            
        Returns:
            Dictionary containing stock data from the API
            
        Raises:
            requests.RequestException: If the API request fails
        """
        url = f"{self.base_url}/stock"
        params = {"ticker": ticker, "company": company}
        
        last_exception = None
        for attempt in range(self.retries):
            try:
                logger.debug(f"Fetching data for {ticker} (attempt {attempt + 1}/{self.retries})")
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt < self.retries - 1:
                    continue
        
        logger.error(f"All {self.retries} attempts failed for {ticker}")
        raise last_exception
    
    def is_available(self) -> bool:
        """
        Check if the API service is available.
        
        Returns:
            True if the health check passes, False otherwise
        """
        try: 
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


class MockStockFetcher(AbstractDataFetcher):
    """
    Mock fetcher for testing purposes.
    Returns sample data without making actual API calls.
    """
    
    def __init__(self, sample_data: Optional[Dict[str, Any]] = None):
        """
        Initialize mock fetcher.
        
        Args:
            sample_data: Optional sample data to return. If None, generates default data.
        """
        self.sample_data = sample_data
    
    def fetch(self, ticker: str, company: str) -> Dict[str, Any]:
        """Return mock stock data."""
        if self.sample_data:
            return self.sample_data
        
        # Generate default mock data
        return {
            "ticker": ticker,
            "company": company,
            "market": {
                "price": 2500.0,
                "open": 2480.0,
                "high": 2520.0,
                "low": 2470.0,
                "volume": 1000000,
                "market_cap": 1500000000000,
                "technical_indicators": {
                    "rsi": 55.0,
                    "sma_50": 2450.0,
                    "sma_200": 2300.0,
                    "macd": 15.5,
                    "macd_signal": 12.3,
                    "macd_histogram": 3.2
                },
                "historical_prices": [
                    {
                        "date": "2024-01-01",
                        "open": 2400.0,
                        "high": 2450.0,
                        "low": 2380.0,
                        "close": 2420.0,
                        "volume": 900000,
                        "rsi": 52.0,
                        "sma_50": 2380.0,
                        "sma_200": 2250.0,
                        "macd": 10.0,
                        "macd_signal": 8.0,
                        "macd_histogram": 2.0
                    },
                    {
                        "date": "2024-01-02",
                        "open": 2420.0,
                        "high": 2480.0,
                        "low": 2410.0,
                        "close": 2460.0,
                        "volume": 950000,
                        "rsi": 54.0,
                        "sma_50": 2400.0,
                        "sma_200": 2270.0,
                        "macd": 12.0,
                        "macd_signal": 9.0,
                        "macd_histogram": 3.0
                    }
                ]
            },
            "ratios": {
                "pe": 25.0,
                "book_value": 1500.0,
                "dividend_yield": 1.2,
                "roce": 18.5,
                "roe": 22.0
            }
        }
    
    def is_available(self) -> bool:
        """Mock fetcher is always available."""
        return True
