

"""
DataFetcher module - provides high-level interface for fetching stock data.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from dotenv import load_dotenv

from .base import AbstractDataFetcher, StockDataPoint
from .api_client import StockAPIClient
from .pipeline import MultiThreadedPipeline, load_stocks_from_file, PipelineStats

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_STOCKS_FILE = Path(__file__).parent.parent / "training_stocks.txt"
DEFAULT_OUTPUT_FILE = Path(__file__).parent.parent / "training_data.csv"


class DataLoader:
    """
    High-level data loader that wraps the fetcher for simple use cases.
    """
    def __init__(self, stock_fetcher: AbstractDataFetcher):
        self.stock_fetcher = stock_fetcher

    def fetch_stock_data(self, ticker: str, company: str):
        return self.stock_fetcher.fetch(ticker, company)


class StockDataPipeline:
    """
    Convenience class for running the complete data fetching pipeline.
    
    Example:
        # Simple usage - uses defaults from .env and training_stocks.txt
        pipeline = StockDataPipeline()
        pipeline.run_if_needed()  # Only runs if CSV doesn't exist
        
        # Or force run
        pipeline.run_from_file("training_stocks.txt")
        
        # Or with explicit stock list
        stocks = [("RELIANCE.NS", "Reliance"), ("TCS.NS", "TCS")]
        pipeline.run(stocks)
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        output_file: Optional[str] = None,
        stocks_file: Optional[str] = None,
        max_workers: int = 5,
        timeout: int = 30,
        retries: int = 3
    ):
        """
        Initialize the pipeline.
        
        Args:
            api_url: URL of the stock data API (defaults to API_URL from .env)
            output_file: Path to output CSV file (defaults to training_data.csv)
            stocks_file: Path to stocks file (defaults to training_stocks.txt)
            max_workers: Number of concurrent threads for fetching
            timeout: API request timeout in seconds
            retries: Number of retry attempts for failed requests
        """
        # Load from environment if not specified
        self.api_url = api_url or os.getenv("API_URL", "http://localhost:8000/stock")
        # Strip the /stock endpoint if present (we add it in the client)
        self.api_url = self.api_url.replace("/stock", "").rstrip("/")
        
        self.output_file = output_file or str(DEFAULT_OUTPUT_FILE)
        self.stocks_file = stocks_file or str(DEFAULT_STOCKS_FILE)
        self.max_workers = max_workers
        self.timeout = timeout
        self.retries = retries
        
        self._client: Optional[StockAPIClient] = None
        self._pipeline: Optional[MultiThreadedPipeline] = None
    
    def _initialize(self) -> None:
        """Initialize the API client and pipeline."""
        self._client = StockAPIClient(
            base_url=self.api_url,
            timeout=self.timeout,
            retries=self.retries
        )
        self._pipeline = MultiThreadedPipeline(
            fetcher=self._client,
            output_file=self.output_file,
            max_workers=self.max_workers
        )
    
    def run(self, stocks: List[Tuple[str, str]]) -> PipelineStats:
        """
        Run pipeline for a list of stocks.
        
        Args:
            stocks: List of (ticker, company) tuples
            
        Returns:
            PipelineStats with execution results
        """
        self._initialize()
        try:
            return self._pipeline.run(stocks)
        finally:
            self.shutdown()
    
    def run_from_file(self, filepath: Optional[str] = None) -> PipelineStats:
        """
        Run pipeline using stocks from a text file.
        
        Args:
            filepath: Path to text file with stock names (one per line).
                      Defaults to training_stocks.txt
            
        Returns:
            PipelineStats with execution results
        """
        filepath = filepath or self.stocks_file
        stocks = load_stocks_from_file(filepath)
        logger.info(f"Loaded {len(stocks)} stocks from file: {filepath}")
        return self.run(stocks)
    
    def csv_exists(self) -> bool:
        """
        Check if the output CSV file already exists and has data.
        
        Returns:
            True if file exists and has content, False otherwise
        """
        output_path = Path(self.output_file)
        return output_path.exists() and output_path.stat().st_size > 0
    
    def run_if_needed(self, filepath: Optional[str] = None) -> Optional[PipelineStats]:
        """
        Run the pipeline only if the CSV file doesn't already exist.
        
        Args:
            filepath: Path to stocks file (defaults to training_stocks.txt)
            
        Returns:
            PipelineStats if pipeline ran, None if skipped (file exists)
        """
        if self.csv_exists():
            logger.info(f"CSV file already exists: {self.output_file}. Skipping data fetch.")
            return None
        
        logger.info(f"CSV file not found. Running pipeline to create: {self.output_file}")
        return self.run_from_file(filepath)
    
    def check_api_health(self) -> bool:
        """Check if the API is available."""
        client = StockAPIClient(base_url=self.api_url, timeout=5)
        return client.is_available()
    
    def shutdown(self) -> None:
        """Shutdown and cleanup resources."""
        if self._pipeline:
            self._pipeline.shutdown()
        
        self._pipeline = None
        self._client = None