"""
Multi-threaded data pipeline for fetching and processing stock data.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .base import (
    AbstractDataFetcher, 
    AbstractDataTransformer, 
    AbstractPipeline,
    StockDataPoint
)
from .csv_writer import ThreadSafeCSVWriter

logger = logging.getLogger(__name__)


class StockDataTransformer(AbstractDataTransformer):
    """
    Transforms raw API response into StockDataPoint objects.
    Calculates daily returns and assigns time indices.
    """
    
    def __init__(self, global_time_idx_lock: Optional[Lock] = None):
        """
        Initialize transformer.
        
        Args:
            global_time_idx_lock: Optional lock for global time index management
        """
        self._time_idx_lock = global_time_idx_lock or Lock()
        self._global_time_idx = 0
    
    def transform(
        self, 
        raw_data: Dict[str, Any], 
        ticker: str,
        start_time_idx: int = 0
    ) -> List[StockDataPoint]:
        """
        Transform raw API data into StockDataPoint objects.
        
        Args:
            raw_data: Raw data from the API
            ticker: Stock ticker symbol
            start_time_idx: Starting time index
            
        Returns:
            List of StockDataPoint objects
        """
        data_points = []
        
        try:
            market_data = raw_data.get("market", {})
            historical_prices = market_data.get("historical_prices", [])
            ratios = raw_data.get("ratios", {})
            
            # Extract fundamental ratios (constant across all historical records)
            pe_ratio = ratios.get("pe")
            book_value = ratios.get("book_value")
            dividend_yield = ratios.get("dividend_yield")
            roce = ratios.get("roce")
            roe = ratios.get("roe")
            eps = ratios.get("eps")
            debt_to_equity = ratios.get("debt_to_equity")
            face_value = ratios.get("face_value")
            market_cap = market_data.get("market_cap")
            
            # Get current technical indicators (for latest data point signals)
            current_tech = market_data.get("technical_indicators", {})
            current_vwap = current_tech.get("vwap")
            current_sma_crossover = current_tech.get("sma_crossover")
            current_rsi_signal = current_tech.get("rsi_signal")
            
            if not historical_prices:
                logger.warning(f"No historical prices found for {ticker}")
                return data_points
            
            # Sort by date to ensure correct order for daily return calculation
            historical_prices = sorted(historical_prices, key=lambda x: x.get("date", ""))
            
            previous_close = None
            
            for idx, price in enumerate(historical_prices):
                current_close = price.get("close", 0.0) or 0.0
                current_high = price.get("high", 0.0) or 0.0
                current_low = price.get("low", 0.0) or 0.0
                sma_50 = price.get("sma_50")
                sma_200 = price.get("sma_200")
                
                # Calculate daily return
                daily_return = self.calculate_daily_return(current_close, previous_close)
                
                # Calculate target (next day's close if available)
                target = None
                target_return = None
                if idx < len(historical_prices) - 1:
                    target = historical_prices[idx + 1].get("close")
                    if target and current_close:
                        target_return = ((target - current_close) / current_close) * 100
                
                # Calculate derived features
                price_to_sma50 = None
                if sma_50 and sma_50 != 0:
                    price_to_sma50 = current_close / sma_50
                
                price_to_sma200 = None
                if sma_200 and sma_200 != 0:
                    price_to_sma200 = current_close / sma_200
                
                volatility = None
                if current_close and current_close != 0:
                    volatility = ((current_high - current_low) / current_close) * 100
                
                # Determine SMA crossover signal based on current values
                sma_crossover = None
                if sma_50 and sma_200:
                    sma_crossover = "bullish" if sma_50 > sma_200 else "bearish"
                
                # Determine RSI signal
                rsi_val = price.get("rsi")
                rsi_signal = None
                if rsi_val is not None:
                    if rsi_val < 30:
                        rsi_signal = "oversold"
                    elif rsi_val > 70:
                        rsi_signal = "overbought"
                    else:
                        rsi_signal = "neutral"
                
                data_point = StockDataPoint(
                    ticker=ticker,
                    date=price.get("date", ""),
                    time_idx=start_time_idx + idx,
                    open=price.get("open", 0.0) or 0.0,
                    high=current_high,
                    low=current_low,
                    close=current_close,
                    volume=price.get("volume", 0) or 0,
                    rsi=rsi_val,
                    sma_50=sma_50,
                    sma_200=sma_200,
                    macd=price.get("macd"),
                    macd_signal=price.get("macd_signal"),
                    macd_histogram=price.get("macd_histogram"),
                    vwap=price.get("vwap") or current_vwap,
                    sma_crossover=sma_crossover,
                    rsi_signal=rsi_signal,
                    pe_ratio=pe_ratio,
                    book_value=book_value,
                    dividend_yield=dividend_yield,
                    roce=roce,
                    roe=roe,
                    eps=eps,
                    debt_to_equity=debt_to_equity,
                    face_value=face_value,
                    market_cap=market_cap,
                    target=target,
                    target_return=target_return,
                    daily_return=daily_return,
                    price_to_sma50=price_to_sma50,
                    price_to_sma200=price_to_sma200,
                    volatility=volatility,
                )
                
                data_points.append(data_point)
                previous_close = current_close
            
            logger.debug(f"Transformed {len(data_points)} data points for {ticker}")
            
        except Exception as e:
            logger.error(f"Error transforming data for {ticker}: {e}")
        
        return data_points
    
    def get_next_time_idx(self, count: int) -> int:
        """
        Get the next available time index range (thread-safe).
        
        Args:
            count: Number of time indices needed
            
        Returns:
            Starting time index for this batch
        """
        with self._time_idx_lock:
            start_idx = self._global_time_idx
            self._global_time_idx += count
            return start_idx


@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""
    total_stocks: int = 0
    successful: int = 0
    failed: int = 0
    total_records: int = 0
    elapsed_time: float = 0.0
    
    def __str__(self) -> str:
        return (
            f"Pipeline Stats: {self.successful}/{self.total_stocks} stocks processed, "
            f"{self.failed} failed, {self.total_records} records, "
            f"{self.elapsed_time:.2f}s elapsed"
        )


class MultiThreadedPipeline(AbstractPipeline):
    """
    Multi-threaded pipeline for fetching, transforming, and writing stock data.
    
    Features:
    - Configurable thread pool for concurrent API calls
    - Thread-safe CSV writing with locks
    - Progress tracking and statistics
    - Graceful shutdown support
    
    Example:
        from interface.api_client import StockAPIClient
        
        client = StockAPIClient("http://localhost:8000")
        pipeline = MultiThreadedPipeline(
            fetcher=client,
            output_file="training_data.csv",
            max_workers=5
        )
        
        stocks = [("RELIANCE.NS", "Reliance"), ("TCS.NS", "TCS")]
        pipeline.run(stocks)
        pipeline.shutdown()
    """
    
    def __init__(
        self,
        fetcher: AbstractDataFetcher,
        output_file: str,
        max_workers: int = 5,
        transformer: Optional[AbstractDataTransformer] = None,
        retry_failed: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            fetcher: Data fetcher implementation
            output_file: Path to output CSV file
            max_workers: Maximum number of concurrent threads
            transformer: Optional custom transformer (defaults to StockDataTransformer)
            retry_failed: Whether to retry failed fetches
        """
        self.fetcher = fetcher
        self.output_file = output_file
        self.max_workers = max_workers
        self.retry_failed = retry_failed
        
        # Initialize components with shared locks
        self._time_idx_lock = Lock()
        self.transformer = transformer or StockDataTransformer(self._time_idx_lock)
        self.writer = ThreadSafeCSVWriter(output_file)
        
        # Thread pool
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # Shutdown flag
        self._shutdown_event = Event()
        
        # Statistics
        self._stats = PipelineStats()
        self._stats_lock = Lock()
    
    def _fetch_and_process(self, ticker: str, company: str) -> Tuple[str, bool, int]:
        """
        Fetch and process data for a single stock.
        
        Args:
            ticker: Stock ticker
            company: Company name
            
        Returns:
            Tuple of (ticker, success, record_count)
        """
        if self._shutdown_event.is_set():
            return (ticker, False, 0)
        
        try:
            logger.info(f"Fetching data for {ticker} ({company})")
            
            # Fetch raw data from API
            raw_data = self.fetcher.fetch(ticker, company)
            
            # Get time index range for this batch
            historical_count = len(raw_data.get("market", {}).get("historical_prices", []))
            start_idx = self.transformer.get_next_time_idx(historical_count)
            
            # Transform to data points
            data_points = self.transformer.transform(raw_data, ticker, start_idx)
            
            if not data_points:
                logger.warning(f"No data points generated for {ticker}")
                return (ticker, False, 0)
            
            # Write to CSV (thread-safe)
            success = self.writer.write(data_points)
            
            if success:
                logger.info(f"Successfully processed {ticker}: {len(data_points)} records")
                return (ticker, True, len(data_points))
            else:
                logger.error(f"Failed to write data for {ticker}")
                return (ticker, False, 0)
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return (ticker, False, 0)
    
    def run(self, stocks: List[Tuple[str, str]]) -> PipelineStats:
        """
        Run the pipeline for a list of stocks.
        
        Args:
            stocks: List of (ticker, company) tuples
            
        Returns:
            PipelineStats with execution statistics
        """
        if not stocks:
            logger.warning("No stocks provided to pipeline")
            return self._stats
        
        self._stats = PipelineStats(total_stocks=len(stocks))
        start_time = time.time()
        
        print(f"\n[Pipeline] Starting with {len(stocks)} stocks, {self.max_workers} workers")
        logger.info(f"Starting pipeline with {len(stocks)} stocks, {self.max_workers} workers")
        
        # Check if API is available
        if not self.fetcher.is_available():
            print("[Pipeline] ERROR: API service is not available!")
            logger.error("API service is not available. Please ensure the server is running.")
            return self._stats
        
        print("[Pipeline] API is available, starting data fetch...\n")
        
        # Create thread pool
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        try:
            # Submit all tasks
            futures = {
                self._executor.submit(self._fetch_and_process, ticker, company): (ticker, company)
                for ticker, company in stocks
            }
            
            # Process completed tasks
            failed_stocks = []
            completed = 0
            total = len(stocks)
            
            for future in as_completed(futures):
                ticker, company = futures[future]
                
                try:
                    result_ticker, success, record_count = future.result()
                    completed += 1
                    
                    with self._stats_lock:
                        if success:
                            self._stats.successful += 1
                            self._stats.total_records += record_count
                            print(f"  [{completed}/{total}] ✓ {ticker}: {record_count:,} records")
                        else:
                            self._stats.failed += 1
                            print(f"  [{completed}/{total}] ✗ {ticker}: FAILED")
                            if self.retry_failed:
                                failed_stocks.append((ticker, company))
                                
                except Exception as e:
                    completed += 1
                    logger.error(f"Future failed for {ticker}: {e}")
                    print(f"  [{completed}/{total}] ✗ {ticker}: ERROR - {e}")
                    with self._stats_lock:
                        self._stats.failed += 1
            
            # Retry failed stocks once
            if failed_stocks and self.retry_failed:
                print(f"\n[Pipeline] Retrying {len(failed_stocks)} failed stocks...")
                logger.info(f"Retrying {len(failed_stocks)} failed stocks...")
                for i, (ticker, company) in enumerate(failed_stocks, 1):
                    if self._shutdown_event.is_set():
                        break
                    print(f"  [Retry {i}/{len(failed_stocks)}] {ticker}...", end=" ", flush=True)
                    result_ticker, success, record_count = self._fetch_and_process(ticker, company)
                    if success:
                        print(f"✓ {record_count:,} records")
                        with self._stats_lock:
                            self._stats.successful += 1
                            self._stats.failed -= 1
                            self._stats.total_records += record_count
                    else:
                        print("✗ FAILED")
        
        finally:
            self._stats.elapsed_time = time.time() - start_time
            self._executor.shutdown(wait=True)
            self._executor = None
        
        print(f"\n[Pipeline] Complete: {self._stats.successful}/{total} stocks, {self._stats.total_records:,} records in {self._stats.elapsed_time:.1f}s")
        logger.info(str(self._stats))
        return self._stats
    
    def shutdown(self) -> None:
        """Shutdown the pipeline gracefully."""
        logger.info("Shutting down pipeline...")
        self._shutdown_event.set()
        
        if self._executor:
            self._executor.shutdown(wait=False)
        
        self.writer.close()
        logger.info("Pipeline shutdown complete")
    
    @property
    def stats(self) -> PipelineStats:
        """Get current pipeline statistics."""
        return self._stats
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


def load_stocks_from_file(filepath: str) -> List[Tuple[str, str]]:
    """
    Load stock list from a text file.
    
    Args:
        filepath: Path to text file with stock names (one per line)
        
    Returns:
        List of (ticker, company) tuples
    """
    stocks = []
    
    with open(filepath, "r") as f:
        for line in f:
            company = line.strip()
            if company:
                # Generate ticker from company name
                ticker = f"{company.upper().replace(' ', '')}.NS"
                stocks.append((ticker, company))
    
    logger.info(f"Loaded {len(stocks)} stocks from {filepath}")
    return stocks
