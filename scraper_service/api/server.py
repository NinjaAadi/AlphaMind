"""
FastAPI server for stock data aggregation microservice.
"""

import os
import logging
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import sys
from dotenv import load_dotenv


# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.aggregator_service import AggregatorService
from models.stock_models import StockSummary, ErrorResponse
from utils.constants import LOG_FORMAT, MSG_SERVICE_HEALTHY, SERVICE_NAME


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)


# Global aggregator service
aggregator_service: Optional[AggregatorService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Initializes resources on startup and cleans up on shutdown.
    """
    # Startup
    global aggregator_service
    load_dotenv()
    news_api_key = os.getenv("NEWS_API_KEY")
    
    if(not news_api_key):
        logger.warning("NEWS_API_KEY not found in environment. News data will be unavailable.")
    aggregator_service = AggregatorService(news_api_key=news_api_key)
    logger.info("Aggregator service initialized")
    
    yield
     
    # Shutdown
    logger.info("Shutting down aggregator service")


# Create FastAPI app
app = FastAPI(
    title="Stock Data Aggregator",
    description="Microservice for collecting and aggregating Indian stock market data",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static files setup
STATIC_DIR = Path(__file__).parent.parent / "static"


@app.get("/", tags=["UI"])
async def serve_dashboard():
    """Serve the main dashboard UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Health status
    """
    return {"status": MSG_SERVICE_HEALTHY, "service": SERVICE_NAME}


@app.get("/stock", response_model=StockSummary, tags=["Stock Data"])
async def get_stock(
    company: str = Query(..., description="Company name (e.g., Reliance, Infosys, TCS)"),
    ticker: str = Query(None, description="Stock ticker (optional - auto-generated from company name if not provided)")
):
    """
    Get aggregated stock data from multiple sources.
    
    This endpoint combines data from:
    - YFinance (market data)
    - Screener.in (fundamentals)
    - NewsAPI (news articles)
    
    Args:
        company: Company name (e.g., Reliance, Infosys, TCS)
        ticker: Optional stock ticker. If not provided, auto-generated as COMPANY.NS
    
    Returns:
        StockSummary: Aggregated stock information
    
    Raises:
        HTTPException: If data retrieval fails or invalid inputs provided
    
    Example:
        GET /stock?company=Reliance
        GET /stock?company=Infosys&ticker=INFY.NS
    """
    try:
        # Validate company name
        if not company or not company.strip():
            raise HTTPException(
                status_code=400,
                detail="'company' parameter is required"
            )
        
        company = company.strip()
        
        # Auto-generate ticker if not provided
        if not ticker:
            # Generate ticker: uppercase company name + .NS suffix
            ticker = f"{company.upper().replace(' ', '')}.NS"
            logger.info(f"Auto-generated ticker: {ticker}")
        else:
            ticker = ticker.strip().upper()
            # Add .NS suffix if not present
            if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
                ticker = f"{ticker}.NS"
                logger.info(f"Added .NS suffix to ticker: {ticker}")
        
        logger.info(f"Request for stock data: ticker={ticker}, company={company}")
        
        # Get aggregated data
        stock_summary = aggregator_service.get_stock_summary(ticker, company)
        
        logger.info(f"Successfully retrieved stock data for {ticker}")
        return stock_summary
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving stock data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stock data: {str(e)}"
        )


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        dict: API information and available endpoints
    """
    return {
        "name": "Stock Data Aggregator",
        "version": "1.0.0",
        "description": "Microservice for collecting Indian stock market data",
        "endpoints": {
            "health": "/health",
            "stock_data": "/stock?ticker=RELIANCE.NS&company=Reliance",
            "docs": "/docs"
        },
        "data_sources": [
            "YFinance (market data)",
            "Screener.in (fundamentals)",
            "NewsAPI (news articles)"
        ]
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload
    )
