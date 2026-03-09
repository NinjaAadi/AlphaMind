"""
Thread-safe CSV writer for stock data pipeline.
Uses locks to prevent data corruption during concurrent writes.
"""

import csv
import logging
from threading import Lock, RLock
from typing import List
from pathlib import Path

from .base import AbstractDataWriter, StockDataPoint

logger = logging.getLogger(__name__)


class ThreadSafeCSVWriter(AbstractDataWriter):
    """
    Thread-safe CSV writer that uses row-level locking.
    
    Features:
    - Row-level locking for fine-grained concurrency control
    - Each row write acquires and releases the lock independently
    - Auto-creates file with headers if it doesn't exist
    
    Example:
        writer = ThreadSafeCSVWriter("stock_data.csv")
        writer.write([data_point1, data_point2])
        writer.close()
    """
    
    def __init__(
        self, 
        filepath: str,
        auto_flush: bool = True
    ):
        """
        Initialize the CSV writer.
        
        Args:
            filepath: Path to the CSV file
            auto_flush: Whether to auto-flush after each row write
        """
        self.filepath = Path(filepath)
        self.auto_flush = auto_flush
        
        # Thread safety - row-level lock
        self._row_lock = RLock()
        
        # Internal state
        self._file_handle = None
        self._csv_writer = None
        self._is_closed = False
        self._total_written = 0
        
        # Initialize file
        self._initialize_file()
    
    def _initialize_file(self) -> None:
        """Initialize the CSV file with headers if it doesn't exist."""
        # Create parent directories if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        file_exists = self.filepath.exists() and self.filepath.stat().st_size > 0
        
        # Open file in append mode
        self._file_handle = open(self.filepath, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._file_handle, 
            fieldnames=StockDataPoint.csv_headers()
        )
        
        # Write headers if new file
        if not file_exists:
            with self._row_lock:
                self._csv_writer.writeheader()
                self._file_handle.flush()
            logger.info(f"Created new CSV file: {self.filepath}")
        else:
            logger.info(f"Appending to existing CSV file: {self.filepath}")
    
    def write_row(self, data_point: StockDataPoint) -> bool:
        """
        Write a single row to CSV with row-level locking.
        
        Args:
            data_point: Single StockDataPoint to write
            
        Returns:
            True if write was successful, False otherwise
        """
        if self._is_closed:
            logger.error("Cannot write to closed writer")
            return False
        
        try:
            # Acquire lock for this single row write
            with self._row_lock:
                self._csv_writer.writerow(data_point.to_dict())
                self._total_written += 1
                
                if self.auto_flush:
                    self._file_handle.flush()
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing row to CSV: {e}")
            return False
    
    def write(self, data_points: List[StockDataPoint]) -> bool:
        """
        Write multiple data points to CSV with per-row locking.
        Each row acquires and releases the lock independently.
        
        Args:
            data_points: List of StockDataPoint objects to write
            
        Returns:
            True if all writes were successful, False otherwise
        """
        if self._is_closed:
            logger.error("Cannot write to closed writer")
            return False
        
        if not data_points:
            return True
        
        success = True
        written_count = 0
        
        # Write each row with individual lock acquisition
        for dp in data_points:
            if self.write_row(dp):
                written_count += 1
            else:
                success = False
        
        logger.debug(f"Wrote {written_count} records (total: {self._total_written})")
        return success
    
    def flush(self) -> None:
        """Manually flush file to disk."""
        with self._row_lock:
            if self._file_handle and not self._is_closed:
                self._file_handle.flush()
    
    def close(self) -> None:
        """Close the writer and release resources."""
        if self._is_closed:
            return
        
        with self._row_lock:
            self._is_closed = True
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
                self._csv_writer = None
        
        logger.info(f"Closed CSV writer. Total records written: {self._total_written}")
    
    @property
    def total_written(self) -> int:
        """Return total number of records written."""
        return self._total_written
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class CSVWriterPool:
    """
    Pool of CSV writers for writing to multiple files.
    Useful when partitioning data by ticker or date.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize writer pool.
        
        Args:
            output_dir: Directory to store CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._writers: dict[str, ThreadSafeCSVWriter] = {}
        self._pool_lock = Lock()
    
    def get_writer(self, name: str) -> ThreadSafeCSVWriter:
        """
        Get or create a writer for the given name.
        
        Args:
            name: Name/key for the writer (used as filename)
            
        Returns:
            ThreadSafeCSVWriter instance
        """
        with self._pool_lock:
            if name not in self._writers:
                filepath = self.output_dir / f"{name}.csv"
                self._writers[name] = ThreadSafeCSVWriter(str(filepath))
            return self._writers[name]
    
    def close_all(self) -> None:
        """Close all writers in the pool."""
        with self._pool_lock:
            for writer in self._writers.values():
                writer.close()
            self._writers.clear()
        
        logger.info("Closed all writers in pool")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
        return False
