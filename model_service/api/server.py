"""
FastAPI server for stock prediction using the trained TFT model.

This service:
1. Takes a stock name/ticker as input
2. Calls the scraper_service API to fetch stock data
3. Preprocesses the data
4. Runs prediction using the trained TFT model
5. Returns the prediction results
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import torch
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer, NaNLabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = Path(__file__).parent.parent / "models"
CHECKPOINT_DIR = Path(__file__).parent.parent / "lightning_logs"
SCRAPER_API_URL = os.getenv("SCRAPER_API_URL", "http://localhost:8000")
MAX_ENCODER_LENGTH = 30
MAX_PREDICTION_LENGTH = 5

# Set default dtype for MPS compatibility
torch.set_default_dtype(torch.float32)


# ============================================================================
# Pydantic Models (Request/Response)
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for stock prediction."""
    stock: str = Field(..., description="Stock name or ticker (e.g., 'Reliance' or 'RELIANCE.NS')")


class HorizonPrediction(BaseModel):
    """Prediction for a single time horizon."""
    horizon: int = Field(..., description="Days ahead (1=tomorrow, 2=day after, etc.)")
    predicted_return: float = Field(..., description="Predicted daily return in percentage points (e.g., 0.5 = 0.5%)")
    predicted_price: float = Field(..., description="Predicted price (median estimate) in INR")
    price_low: float = Field(..., description="10th percentile price - 90% chance actual price is above this")
    price_high: float = Field(..., description="90th percentile price - 90% chance actual price is below this")
    direction: str = Field(..., description="Predicted direction: UP (positive return) or DOWN (negative return)")


class PredictionResponse(BaseModel):
    """Response model for stock prediction."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., RELIANCE.NS)")
    company: str = Field(..., description="Company name")
    current_price: Optional[float] = Field(None, description="Current market price in INR")
    predictions: List[HorizonPrediction] = Field(..., description="Predictions for next 1-5 days")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")
    field_descriptions: Dict[str, str] = Field(
        default={
            "horizon": "Number of days ahead (1=tomorrow, 5=5 days from now)",
            "predicted_return": "Expected daily return in percentage points",
            "predicted_price": "Expected price (median/50th percentile estimate)",
            "price_low": "Conservative price estimate (10th percentile - 90% chance price is above this)",
            "price_high": "Optimistic price estimate (90th percentile - 90% chance price is below this)",
            "direction": "Predicted price movement direction (UP or DOWN)",
            "confidence_interval": "80% confidence interval between price_low and price_high"
        },
        description="Explanation of prediction fields"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Prediction timestamp (UTC)")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    scraper_api: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages TFT model loading and inference."""
    
    def __init__(self):
        self.model: Optional[TemporalFusionTransformer] = None
        self.model_path: Optional[str] = None
        self.is_loaded = False
        
    def find_best_checkpoint(self) -> Optional[Path]:
        """Find the best checkpoint in lightning_logs."""
        if not CHECKPOINT_DIR.exists():
            logger.warning(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
            return None
        
        # Look for checkpoints in version directories
        checkpoints = list(CHECKPOINT_DIR.glob("*/checkpoints/*.ckpt"))
        if not checkpoints:
            logger.warning("No checkpoint files found")
            return None
        
        # Return the most recent checkpoint
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found checkpoint: {latest}")
        return latest
    
    def load_model(self) -> bool:
        """Load the trained TFT model."""
        try:
            # First try saved model weights
            model_file = MODEL_DIR / "tft_stock_model.pt"
            checkpoint_path = self.find_best_checkpoint()
            
            if checkpoint_path and checkpoint_path.exists():
                logger.info(f"Loading model from checkpoint: {checkpoint_path}")
                self.model = TemporalFusionTransformer.load_from_checkpoint(
                    checkpoint_path, 
                    map_location="cpu"
                )
                self.model.eval()
                self.model_path = str(checkpoint_path)
                self.is_loaded = True
                logger.info("Model loaded successfully from checkpoint")
                return True
            elif model_file.exists():
                logger.info(f"Model weights file found at {model_file}")
                logger.warning("Cannot load .pt file directly - need checkpoint. Looking for checkpoints...")
                return False
            else:
                logger.error("No model found. Train the model first using train_tft.py")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Run prediction on preprocessed data.
        
        Uses the model's stored dataset_parameters to create the dataset.
        
        Args:
            data: Preprocessed DataFrame with required features
            
        Returns:
            Predictions array [batch, horizon] for the last window
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Get encoder/prediction lengths from model's stored parameters
        params = self.model.dataset_parameters
        max_encoder = params.get("max_encoder_length", MAX_ENCODER_LENGTH)
        max_pred = params.get("max_prediction_length", MAX_PREDICTION_LENGTH)
        
        # We need at least max_encoder_length + max_prediction_length rows
        min_rows = max_encoder + max_pred
        if len(data) < min_rows:
            raise ValueError(f"Need at least {min_rows} data points, got {len(data)}")
        
        # Use only the last (min_rows + 5) rows for efficiency
        data = data.tail(min_rows + 5).copy()
        data = data.reset_index(drop=True)
        data["time_idx"] = range(len(data))
        
        # Ensure float32
        float_cols = data.select_dtypes(include=['float64']).columns
        data[float_cols] = data[float_cols].astype('float32')
        
        # Create dataset using model's stored parameters
        # Use min_encoder_length=1 to allow the dataset to be created
        dataset_params = params.copy()
        dataset_params["min_encoder_length"] = 1  # Allow shorter for prediction
        dataset_params["min_prediction_length"] = 1
        dataset_params["predict_mode"] = False  # We'll create samples from actual data
        
        logger.info(f"Creating prediction dataset with {len(data)} rows")
        
        # Create dataset from scratch using parameters
        dataset = TimeSeriesDataSet(
            data,
            time_idx=dataset_params["time_idx"],
            target=dataset_params["target"],
            group_ids=dataset_params["group_ids"],
            min_encoder_length=1,
            max_encoder_length=dataset_params["max_encoder_length"],
            min_prediction_length=1,
            max_prediction_length=dataset_params["max_prediction_length"],
            static_categoricals=dataset_params.get("static_categoricals", []),
            time_varying_known_categoricals=dataset_params.get("time_varying_known_categoricals", []),
            time_varying_known_reals=dataset_params.get("time_varying_known_reals", []),
            time_varying_unknown_reals=dataset_params.get("time_varying_unknown_reals", []),
            target_normalizer=TorchNormalizer(method="robust", center=True),
            categorical_encoders={
                "sma_crossover": NaNLabelEncoder(add_nan=True),
                "rsi_signal": NaNLabelEncoder(add_nan=True)
            },
            allow_missing_timesteps=True,
        )
        
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        
        # Run prediction - get only the last batch (most recent prediction)
        self.model.eval()
        last_prediction = None
        
        with torch.no_grad():
            for batch in dataloader:
                x, _ = batch
                pred = self.model(x)
                if hasattr(pred, 'prediction'):
                    last_prediction = pred.prediction.cpu().numpy()
                else:
                    last_prediction = pred.cpu().numpy()
        
        if last_prediction is None:
            raise RuntimeError("No predictions generated")
        
        # Return the last prediction (most recent window)
        return last_prediction


# ============================================================================
# Data Fetcher (calls scraper_service)
# ============================================================================

class ScraperClient:
    """Client for calling the scraper_service API."""
    
    def __init__(self, base_url: str = SCRAPER_API_URL):
        self.base_url = base_url.rstrip("/")
        
    async def get_stock_data(self, stock: str) -> Dict[str, Any]:
        """
        Fetch stock data from scraper service.
        
        Args:
            stock: Stock name or ticker
            
        Returns:
            Stock data dictionary
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Determine if it's a ticker or company name
                if stock.upper().endswith(".NS") or stock.upper().endswith(".BO"):
                    url = f"{self.base_url}/stock?company={stock.replace('.NS', '').replace('.BO', '')}&ticker={stock}"
                else:
                    url = f"{self.base_url}/stock?company={stock}"
                
                logger.info(f"Fetching stock data from: {url}")
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error from scraper: {e.response.status_code}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Scraper service error: {e.response.status_code}"
                )
            except httpx.RequestError as e:
                logger.error(f"Request error to scraper: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Cannot reach scraper service at {self.base_url}"
                )


# ============================================================================
# Data Preprocessor
# ============================================================================

class DataPreprocessor:
    """Preprocess scraped data for model input."""
    
    @staticmethod
    def process(stock_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert API response to model-ready DataFrame.
        
        Args:
            stock_data: Response from scraper service
            
        Returns:
            Preprocessed DataFrame
        """
        ticker = stock_data.get("ticker", "UNKNOWN")
        market = stock_data.get("market", {})
        ratios = stock_data.get("ratios", {})
        historical = market.get("historical_prices", [])
        
        min_rows = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH + 10
        if not historical or len(historical) < min_rows:
            raise ValueError(
                f"Insufficient historical data. Need {min_rows} days, "
                f"got {len(historical) if historical else 0}"
            )
        
        # Convert historical prices to DataFrame
        df = pd.DataFrame(historical)
        
        # Add ticker
        df["ticker"] = ticker
        
        # Ensure date column exists and is sorted
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        
        # Add time_idx
        df["time_idx"] = range(len(df))
        
        # Add fundamental ratios (static for all rows)
        df["pe_ratio"] = ratios.get("pe", 0) or 0
        df["book_value"] = ratios.get("book_value", 0) or 0
        df["dividend_yield"] = ratios.get("dividend_yield", 0) or 0
        df["roce"] = ratios.get("roce", 0) or 0
        df["roe"] = ratios.get("roe", 0) or 0
        df["eps"] = ratios.get("eps", 0) or 0
        df["debt_to_equity"] = ratios.get("debt_to_equity", 0) or 0
        df["face_value"] = ratios.get("face_value", 0) or 0
        df["market_cap"] = market.get("market_cap", 0) or 0
        
        # Calculate derived features
        df["daily_return"] = df["close"].pct_change().fillna(0) * 100
        df["target_return"] = df["daily_return"].shift(-1).fillna(0)  # Next day return
        
        # Price to SMA ratios
        if "sma_50" in df.columns and df["sma_50"].notna().any():
            df["price_to_sma50"] = (df["close"] / df["sma_50"].replace(0, np.nan)).fillna(1)
        else:
            df["sma_50"] = df["close"].rolling(50, min_periods=1).mean()
            df["price_to_sma50"] = (df["close"] / df["sma_50"]).fillna(1)
        
        if "sma_200" in df.columns and df["sma_200"].notna().any():
            df["price_to_sma200"] = (df["close"] / df["sma_200"].replace(0, np.nan)).fillna(1)
        else:
            df["sma_200"] = df["close"].rolling(200, min_periods=1).mean()
            df["price_to_sma200"] = (df["close"] / df["sma_200"]).fillna(1)
        
        # Volatility (20-day rolling std of returns)
        df["volatility"] = df["daily_return"].rolling(20, min_periods=1).std().fillna(0)
        
        # VWAP if not present
        if "vwap" not in df.columns or df["vwap"].isna().all():
            df["vwap"] = ((df["high"] + df["low"] + df["close"]) / 3).fillna(0)
        
        # RSI if not present
        if "rsi" not in df.columns or df["rsi"].isna().all():
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            df["rsi"] = 100 - (100 / (1 + rs))
            df["rsi"] = df["rsi"].fillna(50)
        
        # MACD if not present
        if "macd" not in df.columns or df["macd"].isna().all():
            ema_12 = df["close"].ewm(span=12, adjust=False).mean()
            ema_26 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = ema_12 - ema_26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        # Fill any remaining NaN
        for col in ["macd", "macd_signal", "macd_histogram", "vwap"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Categorical signals
        if "sma_crossover" not in df.columns or df["sma_crossover"].isna().all():
            df["sma_crossover"] = np.where(
                df["sma_50"] > df["sma_200"], "bullish",
                np.where(df["sma_50"] < df["sma_200"], "bearish", "neutral")
            )
        df["sma_crossover"] = df["sma_crossover"].fillna("neutral").astype(str)
        
        if "rsi_signal" not in df.columns or df["rsi_signal"].isna().all():
            df["rsi_signal"] = np.where(
                df["rsi"] < 30, "oversold",
                np.where(df["rsi"] > 70, "overbought", "neutral")
            )
        df["rsi_signal"] = df["rsi_signal"].fillna("neutral").astype(str)
        
        # Convert to float32
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        logger.info(f"Preprocessed {len(df)} rows for {ticker}")
        return df


# ============================================================================
# Global instances
# ============================================================================

model_manager = ModelManager()
scraper_client = ScraperClient()
preprocessor = DataPreprocessor()


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager - load model on startup."""
    logger.info("Starting Model Prediction Service...")
    
    # Try to load model
    if model_manager.load_model():
        logger.info("Model loaded successfully on startup")
    else:
        logger.warning("Model not loaded - predictions will fail until model is available")
    
    yield
    
    logger.info("Shutting down Model Prediction Service")


app = FastAPI(
    title="AlphaMind Model Prediction Service",
    description="API for stock price predictions using Temporal Fusion Transformer",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded,
        model_path=model_manager.model_path,
        scraper_api=SCRAPER_API_URL
    )


@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_stock(stock: str = Query(..., description="Stock name or ticker (e.g., 'Reliance' or 'RELIANCE.NS')")):
    """
    Predict future returns for a stock.
    
    This endpoint:
    1. Fetches current stock data from the scraper service
    2. Preprocesses the data
    3. Runs the TFT model for prediction
    4. Returns predictions for the next 5 time periods
    
    Args:
        stock: Stock name or ticker as query parameter
        
    Returns:
        PredictionResponse with predictions and metadata
    """
    # Check if model is loaded
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    stock = stock.strip()
    if not stock:
        raise HTTPException(status_code=400, detail="Stock name/ticker is required")
    
    logger.info(f"Prediction request for: {stock}")
    
    try:
        # Step 1: Fetch stock data from scraper
        stock_data = await scraper_client.get_stock_data(stock)
        ticker = stock_data.get("ticker", stock)
        company = stock_data.get("company", stock)
        current_price = stock_data.get("market", {}).get("price")
        
        # Step 2: Preprocess data
        df = preprocessor.process(stock_data)
        
        # Step 3: Run prediction
        predictions = model_manager.predict(df)
        
        # Step 4: Format response
        # predictions shape: [1, horizon, quantiles] where quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        # Index 1 = 10th percentile (low), Index 3 = 50th (median), Index 5 = 90th (high)
        
        horizon_predictions = []
        base_price = current_price if current_price else 0
        cumulative_return_low = 0.0
        cumulative_return_med = 0.0
        cumulative_return_high = 0.0
        
        for i in range(min(MAX_PREDICTION_LENGTH, predictions.shape[1])):
            if len(predictions.shape) == 3 and predictions.shape[2] >= 6:
                # Use quantile predictions from model
                # Quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
                return_low = float(predictions[0, i, 1])    # 10th percentile
                return_med = float(predictions[0, i, 3])    # 50th percentile (median)
                return_high = float(predictions[0, i, 5])   # 90th percentile
            else:
                # Fallback if no quantiles
                return_med = float(predictions[0, i] if len(predictions.shape) > 1 else predictions[i])
                return_low = return_med
                return_high = return_med
            
            # Cumulative returns for multi-day predictions
            cumulative_return_low += return_low
            cumulative_return_med += return_med
            cumulative_return_high += return_high
            
            # Calculate prices from cumulative returns
            predicted_price = base_price * (1 + cumulative_return_med / 100) if base_price else 0
            price_low = base_price * (1 + cumulative_return_low / 100) if base_price else 0
            price_high = base_price * (1 + cumulative_return_high / 100) if base_price else 0
            
            horizon_predictions.append(HorizonPrediction(
                horizon=i + 1,
                predicted_return=return_med,
                predicted_price=round(predicted_price, 2),
                price_low=round(price_low, 2),
                price_high=round(price_high, 2),
                direction="UP" if return_med > 0 else "DOWN"
            ))
        
        return PredictionResponse(
            ticker=ticker,
            company=company,
            current_price=current_price,
            predictions=horizon_predictions,
            model_info={
                "model_type": "Temporal Fusion Transformer (TFT)",
                "lookback_days": MAX_ENCODER_LENGTH,
                "forecast_days": MAX_PREDICTION_LENGTH,
                "quantiles": [0.1, 0.5, 0.9],
                "quantile_meaning": "10th, 50th (median), 90th percentile",
                "data_points_used": len(df),
                "model_path": model_manager.model_path
            }
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
