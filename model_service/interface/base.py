"""
Abstract base classes for data fetching pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
from datetime import datetime


@dataclass
class StockDataPoint:
    """Represents a single data point for model training."""
    # Identifiers
    ticker: str
    date: str
    time_idx: int
    
    # OHLCV data
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Technical indicators
    rsi: Optional[float]
    sma_50: Optional[float]
    sma_200: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]
    vwap: Optional[float]
    sma_crossover: Optional[str]  # "bullish", "bearish", or None
    rsi_signal: Optional[str]  # "oversold", "overbought", or "neutral"
    
    # Fundamental ratios
    pe_ratio: Optional[float]
    book_value: Optional[float]
    dividend_yield: Optional[float]
    roce: Optional[float]  # Return on Capital Employed %
    roe: Optional[float]  # Return on Equity %
    eps: Optional[float]
    debt_to_equity: Optional[float]
    face_value: Optional[float]
    market_cap: Optional[float]
    
    # Derived features
    target: Optional[float]  # Next day's close price
    target_return: Optional[float]  # Next day's return %
    daily_return: Optional[float]  # (close - prev_close) / prev_close * 100
    price_to_sma50: Optional[float]  # close / sma_50 ratio
    price_to_sma200: Optional[float]  # close / sma_200 ratio
    volatility: Optional[float]  # (high - low) / close * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV writing."""
        return {
            "ticker": self.ticker,
            "date": self.date,
            "time_idx": self.time_idx,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "rsi": self.rsi,
            "sma_50": self.sma_50,
            "sma_200": self.sma_200,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "vwap": self.vwap,
            "sma_crossover": self.sma_crossover,
            "rsi_signal": self.rsi_signal,
            "pe_ratio": self.pe_ratio,
            "book_value": self.book_value,
            "dividend_yield": self.dividend_yield,
            "roce": self.roce,
            "roe": self.roe,
            "eps": self.eps,
            "debt_to_equity": self.debt_to_equity,
            "face_value": self.face_value,
            "market_cap": self.market_cap,
            "target": self.target,
            "target_return": self.target_return,
            "daily_return": self.daily_return,
            "price_to_sma50": self.price_to_sma50,
            "price_to_sma200": self.price_to_sma200,
            "volatility": self.volatility,
        }

    @staticmethod
    def csv_headers() -> List[str]:
        """Return CSV column headers."""
        return [
            "ticker", "date", "time_idx", "open", "high", "low", "close",
            "volume", "rsi", "sma_50", "sma_200", "macd", "macd_signal",
            "macd_histogram", "vwap", "sma_crossover", "rsi_signal",
            "pe_ratio", "book_value", "dividend_yield", "roce", "roe",
            "eps", "debt_to_equity", "face_value", "market_cap",
            "target", "target_return", "daily_return", "price_to_sma50",
            "price_to_sma200", "volatility"
        ]


class AbstractDataFetcher(ABC):
    """Abstract base class for fetching stock data from various sources."""
    
    @abstractmethod
    def fetch(self, ticker: str, company: str) -> Dict[str, Any]:
        """
        Fetch raw stock data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., "RELIANCE.NS")
            company: Company name (e.g., "Reliance")
            
        Returns:
            Dictionary containing raw stock data from the source
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is available."""
        pass


class AbstractDataTransformer(ABC):
    """Abstract base class for transforming raw data into StockDataPoint objects."""
    
    @abstractmethod
    def transform(
        self, 
        raw_data: Dict[str, Any], 
        ticker: str,
        start_time_idx: int = 0
    ) -> List[StockDataPoint]:
        """
        Transform raw API data into a list of StockDataPoint objects.
        
        Args:
            raw_data: Raw data from the fetcher
            ticker: Stock ticker symbol
            start_time_idx: Starting time index for this batch
            
        Returns:
            List of StockDataPoint objects ready for CSV writing
        """
        pass
    
    @staticmethod
    def calculate_daily_return(current_close: float, previous_close: float) -> Optional[float]:
        """Calculate daily return as percentage change."""
        if previous_close and previous_close != 0:
            return ((current_close - previous_close) / previous_close) * 100
        return None


class AbstractDataWriter(ABC):
    """Abstract base class for writing stock data to storage."""
    
    @abstractmethod
    def write(self, data_points: List[StockDataPoint]) -> bool:
        """
        Write data points to storage.
        
        Args:
            data_points: List of StockDataPoint objects to write
            
        Returns:
            True if write was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass


class AbstractPipeline(ABC):
    """Abstract base class for data processing pipeline."""
    
    @abstractmethod
    def run(self, stocks: List[tuple]) -> None:
        """
        Run the pipeline for a list of stocks.
        
        Args:
            stocks: List of (ticker, company) tuples
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the pipeline and clean up resources."""
        pass
