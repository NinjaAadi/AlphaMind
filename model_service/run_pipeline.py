#!/usr/bin/env python3
"""
Main script to run the stock data fetching pipeline.

This script fetches stock data from the scraper service API and saves it
to a CSV file for model training.

Usage:
    python run_pipeline.py                          # Only runs if CSV doesn't exist
    python run_pipeline.py --force                  # Force run even if CSV exists
    python run_pipeline.py --output data.csv        # Custom output file
    python run_pipeline.py --workers 10             # More threads

Configuration:
    API_URL is loaded from .env file
    Stock names are loaded from training_stocks.txt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add interface module to path
sys.path.insert(0, str(Path(__file__).parent))

from interface import StockDataPipeline, load_stocks_from_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch stock data from API and save to CSV for model training"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="training_data.csv",
        help="Output CSV file path (default: training_data.csv)"
    )
    
    parser.add_argument(
        "--stocks-file", "-f",
        default="training_stocks.txt",
        help="File containing stock names (default: training_stocks.txt)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=5,
        help="Number of concurrent worker threads (default: 5)"
    )
    
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=30,
        help="API request timeout in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--retries", "-r",
        type=int,
        default=3,
        help="Number of retry attempts for failed requests (default: 3)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force run even if CSV file already exists"
    )
    
    parser.add_argument(
        "--wait-for-api",
        type=int,
        default=30,
        help="Seconds to wait for API to be ready (default: 30, for Docker)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get API URL from environment
    api_url = os.getenv("API_URL", "http://localhost:8000/stock")
    
    logger.info("=" * 60)
    logger.info("Stock Data Pipeline")
    logger.info("=" * 60)
    logger.info(f"API URL: {api_url} (from .env)")
    logger.info(f"Stocks file: {args.stocks_file}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Force run: {args.force}")
    logger.info("=" * 60)
    
    # Initialize pipeline (reads API_URL from .env automatically)
    pipeline = StockDataPipeline(
        output_file=args.output,
        stocks_file=args.stocks_file,
        max_workers=args.workers,
        timeout=args.timeout,
        retries=args.retries
    )
    
    # Check if CSV already exists (unless --force)
    if not args.force and pipeline.csv_exists():
        logger.info(f"CSV file already exists: {args.output}")
        logger.info("Use --force to overwrite or delete the file to regenerate.")
        return
    
    # Wait for API to be ready (useful in Docker)
    import time
    logger.info("Checking API health123...")
    api_ready = False
    for attempt in range(args.wait_for_api):

        if pipeline.check_api_health():
            api_ready = True
            break
        logger.info(f"Waiting for API... ({attempt + 1}/{args.wait_for_api})")
        time.sleep(1)
    
    if not api_ready:
        logger.error("API is not available. Please ensure the scraper service is running.")
        logger.error(f"Expected API at: {api_url}")
        logger.error("Start the API with: cd scraper_service && python -m uvicorn api.server:app")
        sys.exit(1)
    
    logger.info("API is healthy")
    
    # Run pipeline
    try:
        stats = pipeline.run_from_file()
        
        logger.info("=" * 60)
        logger.info("Pipeline Complete")
        logger.info("=" * 60)
        logger.info(f"Total stocks: {stats.total_stocks}")
        logger.info(f"Successful: {stats.successful}")
        logger.info(f"Failed: {stats.failed}")
        logger.info(f"Total records: {stats.total_records}")
        logger.info(f"Elapsed time: {stats.elapsed_time:.2f}s")
        logger.info(f"Output saved to: {args.output}")
        logger.info("=" * 60)
        
        if stats.failed > 0:
            logger.warning(f"{stats.failed} stocks failed to process")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        pipeline.shutdown()
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
