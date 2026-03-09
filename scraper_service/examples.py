"""
Example usage of the Stock Data Aggregator service.

This script demonstrates how to use the microservice programmatically.
"""

import requests
import json
from typing import Dict, Any
import time


class StockAggregatorClient:
    """Client for interacting with Stock Data Aggregator API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_stock(self, ticker: str, company: str) -> Dict[str, Any]:
        """
        Get stock data for a single stock.
        
        Args:
            ticker: NSE ticker (e.g., RELIANCE.NS)
            company: Company name (e.g., Reliance)
        
        Returns:
            dict: Stock data
        """
        url = f"{self.base_url}/stock"
        params = {"ticker": ticker, "company": company}
        
        print(f"\n📊 Fetching data for {ticker}...")
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_multiple_stocks(self, stocks: list[Dict[str, str]]) -> list[Dict[str, Any]]:
        """
        Get data for multiple stocks.
        
        Args:
            stocks: List of dicts with 'ticker' and 'company' keys
        
        Returns:
            list: List of stock data
        """
        tickers = ",".join([s["ticker"] for s in stocks])
        companies = ",".join([s["company"] for s in stocks])
        
        url = f"{self.base_url}/stocks/batch"
        params = {"tickers": tickers, "companies": companies}
        
        print(f"\n📊 Fetching data for {len(stocks)} stocks...")
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            url = f"{self.base_url}/health"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def print_stock_summary(self, stock_data: Dict[str, Any]) -> None:
        """
        Print a formatted summary of stock data.
        
        Args:
            stock_data: Stock data from API
        """
        print(f"\n{'='*60}")
        print(f"Company: {stock_data['company']} ({stock_data['ticker']})")
        print(f"{'='*60}")
        
        # Market Data
        market_data = stock_data["market_data"]
        print(f"\n💹 Market Data:")
        print(f"  Current Price: ₹{market_data['current_price']:.2f}")
        if market_data.get("market_cap"):
            print(f"  Market Cap: ₹{market_data['market_cap']:.2e}")
        if market_data.get("pe_ratio"):
            print(f"  P/E Ratio: {market_data['pe_ratio']:.2f}")
        
        if market_data.get("historical_prices"):
            latest_price = market_data["historical_prices"][-1]
            print(f"  Latest Close (30d): ₹{latest_price['close']:.2f} ({latest_price['date']})")
        
        # Fundamentals
        fundamentals = stock_data["fundamentals"]
        print(f"\n📈 Fundamentals:")
        print(f"  P/E Ratio: {fundamentals.get('pe', 'N/A')}")
        print(f"  ROCE: {fundamentals.get('roce', 'N/A')}%")
        print(f"  Debt: {fundamentals.get('debt', 'N/A')}")
        print(f"  Sales Growth: {fundamentals.get('sales_growth', 'N/A')}%")
        print(f"  Profit Growth: {fundamentals.get('profit_growth', 'N/A')}%")
        
        # News
        news = stock_data.get("news", [])
        if news:
            print(f"\n📰 Latest News ({len(news)} articles):")
            for i, article in enumerate(news[:3], 1):
                print(f"  {i}. {article['title']}")
                print(f"     Source: {article['source']}")
                print(f"     URL: {article['url']}")


def example_single_stock():
    """Example: Fetch data for a single stock."""
    print("\n" + "="*60)
    print("Example 1: Single Stock")
    print("="*60)
    
    client = StockAggregatorClient()
    
    # Check health
    if not client.health_check():
        print("❌ API is not running. Start it with:")
        print("   uvicorn api.server:app --reload")
        return
    
    print("✓ API is healthy")
    
    # Get stock data
    try:
        stock_data = client.get_stock("RELIANCE.NS", "Reliance")
        client.print_stock_summary(stock_data)
        
        print(f"\n✅ Successfully fetched data for RELIANCE.NS")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_multiple_stocks():
    """Example: Fetch data for multiple stocks."""
    print("\n" + "="*60)
    print("Example 2: Multiple Stocks")
    print("="*60)
    
    client = StockAggregatorClient()
    
    if not client.health_check():
        print("❌ API is not running")
        return
    
    stocks = [
        {"ticker": "RELIANCE.NS", "company": "Reliance"},
        {"ticker": "INFY.NS", "company": "Infosys"},
        {"ticker": "TCS.NS", "company": "Tata Consultancy Services"},
    ]
    
    try:
        results = client.get_multiple_stocks(stocks)
        
        for stock_data in results:
            client.print_stock_summary(stock_data)
        
        print(f"\n✅ Successfully fetched data for {len(results)} stocks")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_raw_api_call():
    """Example: Direct API call using requests."""
    print("\n" + "="*60)
    print("Example 3: Raw API Call")
    print("="*60)
    
    try:
        response = requests.get(
            "http://localhost:8000/stock",
            params={"ticker": "HDFC.NS", "company": "HDFC Bank"}
        )
        response.raise_for_status()
        
        data = response.json()
        print("\n📦 Raw API Response:")
        print(json.dumps(data, indent=2)[:500] + "...")
        
        print(f"\n✅ API call successful")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_error_handling():
    """Example: Error handling."""
    print("\n" + "="*60)
    print("Example 4: Error Handling")
    print("="*60)
    
    client = StockAggregatorClient()
    
    if not client.health_check():
        print("❌ API is not running")
        print("   Start with: uvicorn api.server:app --reload")
        return
    
    # Try invalid ticker
    try:
        print("\n Testing with invalid ticker format...")
        stock_data = client.get_stock("INVALID", "Invalid Company")
        print("✓ Request succeeded (yfinance handled the invalid ticker gracefully)")
    except Exception as e:
        print(f"✓ Caught error: {e}")
    
    # Missing parameters
    try:
        print("\n Testing with missing parameters...")
        response = requests.get("http://localhost:8000/stock")
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"✓ Caught HTTP error: {response.status_code}")
        print(f"  Details: {response.json()}")


def example_performance():
    """Example: Performance testing."""
    print("\n" + "="*60)
    print("Example 5: Performance Testing")
    print("="*60)
    
    client = StockAggregatorClient()
    
    if not client.health_check():
        print("❌ API is not running")
        return
    
    stocks = [
        ("RELIANCE.NS", "Reliance"),
        ("INFY.NS", "Infosys"),
        ("TCS.NS", "Tata Consultancy Services"),
        ("HDFC.NS", "HDFC Bank"),
        ("ICICIBANK.NS", "ICICI Bank"),
    ]
    
    print(f"\nFetching data for {len(stocks)} stocks...")
    start_time = time.time()
    
    for ticker, company in stocks:
        try:
            client.get_stock(ticker, company)
            print(f"  ✓ {ticker}")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds")
    print(f"⏱️  Average per stock: {elapsed_time/len(stocks):.2f} seconds")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Stock Data Aggregator - Usage Examples")
    print("="*60)
    print("\nMake sure the API is running:")
    print("  uvicorn api.server:app --reload")
    
    # Run examples
    example_single_stock()
    example_multiple_stocks()
    example_raw_api_call()
    example_error_handling()
    example_performance()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nFor more information, see:")
    print("  - README.md (Full documentation)")
    print("  - QUICKSTART.md (Quick start guide)")
    print("  - API docs: http://localhost:8000/docs")
