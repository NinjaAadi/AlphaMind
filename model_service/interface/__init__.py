"""
Interface module for stock data fetching and processing pipeline.

This module provides:
- Abstract base classes for data fetching, transformation, and writing
- API client for the scraper service
- Thread-safe CSV writer with proper locking
- Multi-threaded pipeline for efficient data collection

Usage:
    from interface import StockDataPipeline
    
    # Simple usage with file
    pipeline = StockDataPipeline(output_file="training_data.csv")
    pipeline.run_from_file("training_stocks.txt")
    
    # Or with explicit stocks
    stocks = [("RELIANCE.NS", "Reliance"), ("TCS.NS", "TCS")]
    pipeline.run(stocks)
"""

from .base import (
    AbstractDataFetcher,
    AbstractDataTransformer,
    AbstractDataWriter,
    AbstractPipeline,
    StockDataPoint,
)

from .api_client import (
    StockAPIClient,
    MockStockFetcher,
)

from .csv_writer import (
    ThreadSafeCSVWriter,
    CSVWriterPool,
)

from .pipeline import (
    StockDataTransformer,
    MultiThreadedPipeline,
    PipelineStats,
    load_stocks_from_file,
)

from .DataFetcher import (
    DataLoader,
    StockDataPipeline,
    DEFAULT_STOCKS_FILE,
    DEFAULT_OUTPUT_FILE,
)

__all__ = [
    # Base classes
    "AbstractDataFetcher",
    "AbstractDataTransformer", 
    "AbstractDataWriter",
    "AbstractPipeline",
    "StockDataPoint",
    # API client
    "StockAPIClient",
    "MockStockFetcher",
    # CSV writer
    "ThreadSafeCSVWriter",
    "CSVWriterPool",
    # Pipeline
    "StockDataTransformer",
    "MultiThreadedPipeline",
    "PipelineStats",
    "load_stocks_from_file",
    # High-level interface
    "DataLoader",
    "StockDataPipeline",
    # Default paths
    "DEFAULT_STOCKS_FILE",
    "DEFAULT_OUTPUT_FILE",
]
