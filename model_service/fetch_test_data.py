#!/usr/bin/env python3
"""
Script to fetch testing data using the same pipeline as training data.

Usage:
    python fetch_test_data.py
    
This script:
    1. Reads stocks from testing_stocks.txt
    2. Fetches data from the stock API
    3. Saves to output/testing_data.csv
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from interface.DataFetcher import StockDataPipeline


def main():
    """Fetch testing data using the pipeline."""
    
    # Configuration
    testing_stocks_file = "testing_stocks.txt"
    output_file = "output/testing_data.csv"
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FETCHING TESTING DATA")
    print("=" * 60)
    print(f"Stocks file: {testing_stocks_file}")
    print(f"Output file: {output_file}")
    
    # Create pipeline with test output path
    pipeline = StockDataPipeline(
        output_file=output_file,
        stocks_file=testing_stocks_file,
        max_workers=3,
        timeout=120,
        retries=3
    )
    
    # Check if test data already exists
    if pipeline.csv_exists():
        print(f"\nTest data already exists at {output_file}")
        user_input = input("Do you want to regenerate? (y/N): ").strip().lower()
        if user_input != 'y':
            print("Skipping data fetch.")
            return
    
    # Run the pipeline
    print("\nStarting data fetch...")
    stats = pipeline.run_from_file(testing_stocks_file)
    
    # Print results
    print("\n" + "=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)
    print(f"Total stocks: {stats.total_stocks}")
    print(f"Successful: {stats.successful}")
    print(f"Failed: {stats.failed}")
    print(f"Total records: {stats.total_records:,}")
    print(f"Duration: {stats.elapsed_time:.2f}s")
    
    print(f"\nTest data saved to: {output_file}")


if __name__ == "__main__":
    main()
