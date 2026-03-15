"""
Temporal Fusion Transformer (TFT) for Stock Price Prediction.

This module provides a class-based approach to:
1. Load and preprocess stock data
2. Clean and validate the dataset
3. Create TFT model and train
4. Save and evaluate the model
"""

import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Tuple

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data.encoders import TorchNormalizer, EncoderNormalizer
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Register pytorch_forecasting classes as safe globals for checkpoint loading (PyTorch 2.6+)
torch.serialization.add_safe_globals([
    GroupNormalizer,
    NaNLabelEncoder, 
    TorchNormalizer,
    EncoderNormalizer,
    QuantileLoss,
])

# Set default dtype to float32 for MPS (Apple Silicon) compatibility
torch.set_default_dtype(torch.float32)

# Configuration constants
NUM_WORKERS = 7  # Number of workers for data loading
BATCH_SIZE = 64  # Default batch size


class StockDataPreprocessor:
    """
    Preprocessor for stock data before TFT training.
    
    Handles:
    - Loading CSV data
    - Cleaning missing/invalid values
    - Feature engineering
    - Data validation
    - Descriptive statistics
    """
    
    # Columns required for training
    REQUIRED_COLUMNS = [
        "ticker", "date", "time_idx", "open", "high", "low", "close", "volume",
        "rsi", "sma_50", "sma_200", "macd", "macd_signal", "macd_histogram",
        "target_return", "daily_return"
    ]
    
    # Numeric columns that need cleaning
    NUMERIC_COLUMNS = [
        "open", "high", "low", "close", "volume", "rsi", "sma_50", "sma_200",
        "macd", "macd_signal", "macd_histogram", "vwap", "pe_ratio", "book_value",
        "dividend_yield", "roce", "roe", "eps", "debt_to_equity", "face_value",
        "market_cap", "target", "target_return", "daily_return", "price_to_sma50",
        "price_to_sma200", "volatility"
    ]
    
    # Columns to use for TFT training - numeric time-varying features
    TIME_VARYING_FEATURES = [
        # OHLCV
        "open", "high", "low", "close", "volume",
        # Technical indicators
        "rsi", "sma_50", "sma_200", "macd", "macd_signal", "macd_histogram", "vwap",
        # Fundamental ratios
        "pe_ratio", "book_value", "dividend_yield", "roce", "roe", "eps",
        "debt_to_equity", "face_value", "market_cap",
        # Derived features
        "daily_return", "price_to_sma50", "price_to_sma200", "volatility"
    ]
    
    # Categorical features for TFT
    CATEGORICAL_FEATURES = ["sma_crossover", "rsi_signal"]
    
    def __init__(self, data_path: str):
        """
        Initialize preprocessor.
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self._stats: dict = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
        
        return self.df
    
    def describe_raw_data(self) -> None:
        """Print descriptive statistics for raw data."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "=" * 60)
        print("RAW DATA DESCRIPTION")
        print("=" * 60)
        
        # Basic info
        print(f"\nShape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
        print(f"\nColumns: {list(self.df.columns)}")
        
        # Unique tickers
        n_tickers = self.df["ticker"].nunique()
        print(f"\nUnique tickers: {n_tickers}")
        print(f"Tickers: {sorted(self.df['ticker'].unique())[:10]}..." if n_tickers > 10 else f"Tickers: {sorted(self.df['ticker'].unique())}")
        
        # Date range
        if "date" in self.df.columns:
            print(f"\nDate range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        # Missing values
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            "Missing": missing,
            "Percent": missing_pct
        })
        missing_df = missing_df[missing_df["Missing"] > 0].sort_values("Missing", ascending=False)
        if len(missing_df) > 0:
            print(missing_df.to_string())
        else:
            print("No missing values!")
        
        # Numeric statistics
        print("\n--- Numeric Column Statistics ---")
        numeric_cols = [c for c in self.NUMERIC_COLUMNS if c in self.df.columns]
        print(self.df[numeric_cols].describe().round(2).to_string())
        
        # Records per ticker
        print("\n--- Records per Ticker ---")
        ticker_counts = self.df.groupby("ticker").size()
        print(f"Min records: {ticker_counts.min()}")
        print(f"Max records: {ticker_counts.max()}")
        print(f"Mean records: {ticker_counts.mean():.0f}")
        print(f"Median records: {ticker_counts.median():.0f}")
    
    def clean_data(
        self,
        drop_na_columns: Optional[List[str]] = None,
        fill_na_columns: Optional[dict] = None,
        remove_zero_volume: bool = True,
        remove_negative_prices: bool = True,
        min_records_per_ticker: int = 100
    ) -> pd.DataFrame:
        """
        Clean the dataset.
        
        Args:
            drop_na_columns: Columns where NA rows should be dropped
            fill_na_columns: Dict of {column: fill_value} for filling NAs
            remove_zero_volume: Remove rows with zero volume
            remove_negative_prices: Remove rows with negative prices
            min_records_per_ticker: Minimum records required per ticker
            
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "=" * 60)
        print("CLEANING DATA")
        print("=" * 60)
        
        df = self.df.copy()
        initial_rows = len(df)
        
        # 1. Sort by ticker and time_idx
        print("\n1. Sorting by ticker and time_idx...")
        df = df.sort_values(["ticker", "time_idx"]).reset_index(drop=True)
        
        # 2. Convert date column
        if "date" in df.columns:
            print("2. Converting date column...")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        # 3. Fill NA values for specified columns
        if fill_na_columns:
            print(f"3. Filling NA values for {list(fill_na_columns.keys())}...")
            for col, value in fill_na_columns.items():
                if col in df.columns:
                    df[col] = df[col].fillna(value)
        
        # 3b. Fill categorical columns with "unknown"
        categorical_cols = ["sma_crossover", "rsi_signal"]
        print(f"3b. Filling categorical columns: {categorical_cols}")
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("unknown").astype(str)
        
        # 3c. Fill fundamental ratio NAs with 0 (they may be missing for some stocks)
        fundamental_cols = [
            "pe_ratio", "book_value", "dividend_yield", "roce", "roe",
            "eps", "debt_to_equity", "face_value", "market_cap",
            "price_to_sma50", "price_to_sma200"
        ]
        print(f"3c. Filling fundamental ratio NAs with 0...")
        for col in fundamental_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # 4. Drop rows with NA in critical columns
        critical_columns = drop_na_columns or [
            "ticker", "time_idx", "open", "high", "low", "close", "volume",
            "target_return", "daily_return"
        ]
        critical_columns = [c for c in critical_columns if c in df.columns]
        print(f"4. Dropping rows with NA in critical columns: {critical_columns}")
        before = len(df)
        df = df.dropna(subset=critical_columns)
        print(f"   Dropped {before - len(df):,} rows")
        
        # 5. Remove zero volume rows
        if remove_zero_volume and "volume" in df.columns:
            print("5. Removing zero volume rows...")
            before = len(df)
            df = df[df["volume"] > 0]
            print(f"   Dropped {before - len(df):,} rows")
        
        # 6. Remove negative prices
        if remove_negative_prices:
            print("6. Removing negative price rows...")
            before = len(df)
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                if col in df.columns:
                    df = df[df[col] > 0]
            print(f"   Dropped {before - len(df):,} rows")
        
        # 7. Remove infinite values
        print("7. Removing infinite values...")
        before = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=numeric_cols)
        print(f"   Dropped {before - len(df):,} rows")
        
        # 8. Filter tickers with insufficient data
        print(f"8. Filtering tickers with < {min_records_per_ticker} records...")
        before_tickers = df["ticker"].nunique()
        ticker_counts = df.groupby("ticker").size()
        valid_tickers = ticker_counts[ticker_counts >= min_records_per_ticker].index
        df = df[df["ticker"].isin(valid_tickers)]
        after_tickers = df["ticker"].nunique()
        print(f"   Removed {before_tickers - after_tickers} tickers")
        
        # 9. Reassign time_idx per ticker (ensure continuous)
        print("9. Reassigning continuous time_idx per ticker...")
        df["time_idx"] = df.groupby("ticker").cumcount()
        
        # Summary
        print("\n--- Cleaning Summary ---")
        print(f"Initial rows: {initial_rows:,}")
        print(f"Final rows: {len(df):,}")
        print(f"Rows removed: {initial_rows - len(df):,} ({(initial_rows - len(df)) / initial_rows * 100:.1f}%)")
        print(f"Final tickers: {df['ticker'].nunique()}")
        
        self.df_clean = df
        return df
    
    def describe_clean_data(self) -> None:
        """Print descriptive statistics for cleaned data."""
        if self.df_clean is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")
        
        df = self.df_clean
        
        print("\n" + "=" * 60)
        print("CLEANED DATA DESCRIPTION")
        print("=" * 60)
        
        # Basic info
        print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"Unique tickers: {df['ticker'].nunique()}")
        
        # Date range
        if "date" in df.columns:
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Missing values check
        missing = df.isnull().sum().sum()
        print(f"\nTotal missing values: {missing}")
        
        # Target variable statistics
        print("\n--- Target Variable (target_return) ---")
        print(df["target_return"].describe().round(4).to_string())
        
        # Feature statistics
        print("\n--- Feature Statistics ---")
        feature_cols = [c for c in self.TIME_VARYING_FEATURES if c in df.columns]
        print(df[feature_cols].describe().round(2).to_string())
        
        # Correlation with target
        print("\n--- Top Correlations with target_return ---")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()["target_return"].drop("target_return").abs().sort_values(ascending=False)
        print(correlations.head(10).round(4).to_string())
        
        # Records per ticker distribution
        print("\n--- Records per Ticker ---")
        ticker_counts = df.groupby("ticker").size()
        print(f"Min: {ticker_counts.min()}, Max: {ticker_counts.max()}, Mean: {ticker_counts.mean():.0f}")
        
        # Store stats
        self._stats = {
            "n_rows": len(df),
            "n_tickers": df["ticker"].nunique(),
            "n_features": len(feature_cols),
            "target_mean": df["target_return"].mean(),
            "target_std": df["target_return"].std(),
        }
    
    def get_clean_data(self) -> pd.DataFrame:
        """Get the cleaned DataFrame."""
        if self.df_clean is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")
        return self.df_clean
    
    def save_clean_data(self, output_path: str) -> None:
        """Save cleaned data to CSV."""
        if self.df_clean is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")
        
        self.df_clean.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")


class TFTStockModel:
    """
    Temporal Fusion Transformer model for stock prediction.
    
    Handles:
    - Creating TimeSeriesDataSet
    - Building TFT model
    - Training
    - Saving/loading model
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        max_encoder_length: int = 30,
        max_prediction_length: int = 5
    ):
        """
        Initialize TFT model.
        
        Args:
            df: Cleaned DataFrame
            max_encoder_length: Number of past time steps to use (30 = ~6 trading weeks)
            max_prediction_length: Number of future steps to predict
        """
        # Convert float64 to float32 for MPS (Apple Silicon) compatibility
        float_cols = df.select_dtypes(include=['float64']).columns
        df = df.copy()
        df[float_cols] = df[float_cols].astype('float32')
        
        self.df = df
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        
        self.training_dataset: Optional[TimeSeriesDataSet] = None
        self.validation_dataset: Optional[TimeSeriesDataSet] = None
        self.model: Optional[TemporalFusionTransformer] = None
    
    def create_datasets(
        self,
        train_ratio: float = 0.8,
        batch_size: int = 64
    ) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Create training and validation datasets."""
        
        # Split by time within each ticker
        training_cutoff = self.df.groupby("ticker")["time_idx"].transform(
            lambda x: x.quantile(train_ratio)
        )
        
        train_df = self.df[self.df["time_idx"] <= training_cutoff]
        val_df = self.df[self.df["time_idx"] > training_cutoff]
        
        print(f"Training samples: {len(train_df):,}")
        print(f"Validation samples: {len(val_df):,}")
        
        # Create training dataset with all available features
        # Note: ticker is used only as group_id for data organization, NOT as a feature
        # This allows the model to generalize to unseen tickers based on features alone
        self.training_dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="target_return",
            group_ids=["ticker"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            # No static_categoricals - model learns from features, not ticker identity
            static_categoricals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_known_categoricals=["sma_crossover", "rsi_signal"],
            time_varying_unknown_reals=[
                # OHLCV
                "open", "high", "low", "close", "volume",
                # Technical indicators
                "rsi", "sma_50", "sma_200", "macd", "macd_signal",
                "macd_histogram", "vwap",
                # Fundamental ratios (static per ticker but can vary over time)
                "pe_ratio", "book_value", "dividend_yield", "roce", "roe",
                "eps", "debt_to_equity", "face_value", "market_cap",
                # Derived features
                "daily_return", "price_to_sma50", "price_to_sma200", "volatility"
            ],
            # Use global normalization so model can predict on any ticker
            target_normalizer=TorchNormalizer(
                method="robust",  # Robust to outliers, uses median/IQR
                center=True,
            ),
            categorical_encoders={
                "sma_crossover": NaNLabelEncoder(add_nan=True),
                "rsi_signal": NaNLabelEncoder(add_nan=True)
            },
            allow_missing_timesteps=True  # Allow gaps in time series
        )
        
        # Create validation dataset
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            val_df,
            predict=True,
            stop_randomization=True
        )
        
        return self.training_dataset, self.validation_dataset
    
    def build_model(
        self,
        learning_rate: float = 0.005,
        hidden_size: int = 32,
        attention_head_size: int = 4,
        dropout: float = 0.1
    ) -> TemporalFusionTransformer:
        """Build TFT model from dataset."""
        
        if self.training_dataset is None:
            raise ValueError("Dataset not created. Call create_datasets() first.")
        
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=16,
            loss=QuantileLoss(),
            log_interval=10
        )
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        return self.model
    
    def train(
        self,
        max_epochs: int = 20,
        batch_size: int = 64,
        ckpt_path: Optional[str] = None
    ) -> Trainer:
        """Train the model.
        
        Args:
            max_epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            ckpt_path: Path to checkpoint file to resume training from
        """
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        train_dataloader = self.training_dataset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=NUM_WORKERS
        )
        val_dataloader = self.validation_dataset.to_dataloader(
            train=False, batch_size=batch_size, num_workers=NUM_WORKERS
        )
        
        # Callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10,  # Increased patience to allow more training
            verbose=True,
            mode="min"
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True
        )
        
        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            callbacks=[early_stop_callback, checkpoint_callback],
            gradient_clip_val=0.1,
            enable_progress_bar=True
        )
        
        # Use weights_only=False when resuming since checkpoint contains serialized objects
        trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path,
            weights_only=False if ckpt_path else True
        )
        
        return trainer
    
    def save_model(self, path: str) -> None:
        """Save model weights and dataset parameters so the API server can load the .pt file."""
        if self.model is None:
            raise ValueError("Model not built.")
        if self.training_dataset is None:
            raise ValueError("Dataset not created. Call create_datasets() first.")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "state_dict": self.model.state_dict(),
            "dataset_parameters": self.training_dataset.get_parameters(),
        }
        torch.save(bundle, path)
        print(f"Model and dataset parameters saved to {path}")
    
    def load_model(self, path: str) -> TemporalFusionTransformer:
        """
        Load model weights from file.
        
        Args:
            path: Path to the saved model weights
            
        Returns:
            Loaded model
        """
        if self.training_dataset is None:
            raise ValueError("Dataset not created. Call create_datasets() first.")
        
        if self.model is None:
            self.build_model()
        
        # Load checkpoint (support both bundle format and plain state_dict)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "state_dict" in loaded:
            state_dict = loaded["state_dict"]
        else:
            state_dict = loaded
        model_state = self.model.state_dict()
        
        # Filter out keys with size mismatch (e.g., ticker embeddings when add_nan=True changes size)
        filtered_state = {}
        skipped_keys = []
        for key, value in state_dict.items():
            if key in model_state:
                if value.shape == model_state[key].shape:
                    filtered_state[key] = value
                else:
                    skipped_keys.append(f"{key}: checkpoint {value.shape} vs model {model_state[key].shape}")
            else:
                skipped_keys.append(f"{key}: not in model")
        
        if skipped_keys:
            print(f"Warning: Skipped {len(skipped_keys)} incompatible weights:")
            for key in skipped_keys[:5]:
                print(f"  - {key}")
            if len(skipped_keys) > 5:
                print(f"  ... and {len(skipped_keys) - 5} more")
        
        # Load compatible weights
        self.model.load_state_dict(filtered_state, strict=False)
        self.model.eval()
        print(f"Model loaded from {path} ({len(filtered_state)}/{len(state_dict)} weights)")
        return self.model
    
    def create_test_dataset(
        self,
        test_df: pd.DataFrame
    ) -> TimeSeriesDataSet:
        """
        Create a test dataset from new data.
        
        Args:
            test_df: Cleaned test DataFrame
            
        Returns:
            Test TimeSeriesDataSet
        """
        if self.training_dataset is None:
            raise ValueError("Training dataset not created. Call create_datasets() first.")
        
        # Convert float64 to float32 for MPS compatibility
        float_cols = test_df.select_dtypes(include=['float64']).columns
        test_df = test_df.copy()
        test_df[float_cols] = test_df[float_cols].astype('float32')
        
        # Log ticker info (model can predict on any ticker since it uses features, not ticker identity)
        training_tickers = set(self.df["ticker"].unique())
        test_tickers = set(test_df["ticker"].unique())
        unknown_tickers = test_tickers - training_tickers
        known_tickers = test_tickers & training_tickers
        
        print(f"Test tickers: {len(test_tickers)} total, {len(known_tickers)} seen in training, {len(unknown_tickers)} new")
        
        # Create test dataset based on training dataset structure
        # Use predict=False to get all possible windows, not just the last one per ticker
        test_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            test_df,
            predict=False,  # Changed: get all windows for proper evaluation
            stop_randomization=True,
            allow_missing_timesteps=True  # Handle gaps in test data
        )
        
        print(f"Test dataset created: {len(test_dataset):,} samples (from {len(test_df):,} rows)")
        return test_dataset
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        batch_size: int = 64
    ) -> dict:
        """
        Evaluate model on test data.
        
        Args:
            test_df: Cleaned test DataFrame
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create test dataset
        test_dataset = self.create_test_dataset(test_df)
        test_dataloader = test_dataset.to_dataloader(
            train=False, batch_size=batch_size, num_workers=NUM_WORKERS
        )
        
        print(f"Dataloader batches: {len(test_dataloader)}, expected samples: ~{len(test_dataloader) * batch_size}")
        
        # Get predictions
        self.model.eval()
        predictions = []
        actuals = []
        
        print("Running evaluation...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                x, y = batch
                # Get prediction (median quantile)
                pred = self.model(x)
                predictions.append(pred.prediction.cpu())
                actuals.append(y[0].cpu())
        
        # Concatenate all predictions
        predictions = torch.cat(predictions, dim=0)
        actuals = torch.cat(actuals, dim=0)
        
        # Calculate metrics
        # For quantile predictions, use median (index 3 for 7 quantiles)
        if len(predictions.shape) == 3:  # [batch, horizon, quantiles]
            pred_median = predictions[:, :, predictions.shape[2] // 2]
        else:
            pred_median = predictions
        
        # Flatten for metrics
        pred_flat = pred_median.flatten().numpy()
        actual_flat = actuals.flatten().numpy()
        
        # Mean Absolute Error
        mae = np.abs(pred_flat - actual_flat).mean()
        
        # Root Mean Squared Error
        rmse = np.sqrt(((pred_flat - actual_flat) ** 2).mean())
        
        # Mean Absolute Percentage Error (handle zeros)
        mask = actual_flat != 0
        if mask.sum() > 0:
            mape = np.abs((pred_flat[mask] - actual_flat[mask]) / actual_flat[mask]).mean() * 100
        else:
            mape = float('nan')
        
        # Directional Accuracy (for return prediction)
        direction_correct = ((pred_flat > 0) == (actual_flat > 0)).mean() * 100
        
        # R-squared
        ss_res = ((actual_flat - pred_flat) ** 2).sum()
        ss_tot = ((actual_flat - actual_flat.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
        
        results = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "direction_accuracy": direction_correct,
            "r2": r2,
            "n_samples": len(pred_flat),
            "n_tickers": test_df["ticker"].nunique()
        }
        
        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Samples tested: {results['n_samples']:,}")
        print(f"Tickers tested: {results['n_tickers']}")
        print(f"\nMetrics:")
        print(f"  MAE:                {results['mae']:.6f}")
        print(f"  RMSE:               {results['rmse']:.6f}")
        print(f"  MAPE:               {results['mape']:.2f}%")
        print(f"  Direction Accuracy: {results['direction_accuracy']:.2f}%")
        print(f"  R²:                 {results['r2']:.4f}")
        
        return results
    
    def predict(
        self,
        test_df: pd.DataFrame,
        batch_size: int = 64,
        return_quantiles: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions on test data.
        
        Args:
            test_df: Cleaned test DataFrame
            batch_size: Batch size for prediction
            return_quantiles: If True, return all quantile predictions
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create test dataset
        test_dataset = self.create_test_dataset(test_df)
        test_dataloader = test_dataset.to_dataloader(
            train=False, batch_size=batch_size, num_workers=NUM_WORKERS
        )
        
        # Get predictions
        self.model.eval()
        all_predictions = []
        
        print("Making predictions...")
        with torch.no_grad():
            for batch in test_dataloader:
                x, _ = batch
                pred = self.model(x)
                all_predictions.append(pred.prediction.cpu())
        
        # Concatenate predictions
        predictions = torch.cat(all_predictions, dim=0).numpy()
        
        # Create results DataFrame
        n_samples = predictions.shape[0]
        horizon = predictions.shape[1] if len(predictions.shape) > 1 else 1
        
        if return_quantiles and len(predictions.shape) == 3:
            # Return all quantiles
            quantile_labels = ["q10", "q20", "q30", "q50", "q60", "q70", "q90"]
            results = []
            for i in range(n_samples):
                for h in range(horizon):
                    row = {"sample_idx": i, "horizon": h + 1}
                    for q_idx, q_label in enumerate(quantile_labels):
                        row[q_label] = predictions[i, h, q_idx]
                    results.append(row)
            result_df = pd.DataFrame(results)
        else:
            # Return median predictions only
            if len(predictions.shape) == 3:
                pred_median = predictions[:, :, predictions.shape[2] // 2]
            else:
                pred_median = predictions
            
            results = []
            for i in range(n_samples):
                for h in range(horizon):
                    results.append({
                        "sample_idx": i,
                        "horizon": h + 1,
                        "prediction": pred_median[i, h] if len(pred_median.shape) > 1 else pred_median[i]
                    })
            result_df = pd.DataFrame(results)
        
        print(f"Generated {len(result_df):,} predictions")
        return result_df


# ------------------------
# Main execution
# ------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TFT Stock Model")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint file to resume training from")
    parser.add_argument("--max-epochs", type=int, default=20,
                        help="Maximum number of epochs to train")
    args = parser.parse_args()
    
    # Configuration - use absolute paths relative to model_training directory
    BASE_DIR = Path(__file__).parent.parent  # model_training/
    TRAIN_DATA_PATH = BASE_DIR / "output" / "training_data.csv"
    TEST_DATA_PATH = BASE_DIR / "output" / "testing_data.csv"
    MODEL_PATH = BASE_DIR / "models" / "tft_stock_model.pt"
    
    # Step 1: Load and preprocess training data
    print("\n" + "=" * 60)
    print("STEP 1: LOADING TRAINING DATA")
    print("=" * 60)
    
    preprocessor = StockDataPreprocessor(str(TRAIN_DATA_PATH))
    preprocessor.load_data()
    
    # Step 2: Describe raw data
    print("\n" + "=" * 60)
    print("STEP 2: DESCRIBING RAW DATA")
    print("=" * 60)
    
    preprocessor.describe_raw_data()
    
    # Step 3: Clean data
    print("\n" + "=" * 60)
    print("STEP 3: CLEANING DATA")
    print("=" * 60)
    
    preprocessor.clean_data(
        fill_na_columns={
            "vwap": 0,
            "pe_ratio": 0,
            "book_value": 0,
            "dividend_yield": 0,
        },
        min_records_per_ticker=100
    )
    
    # Step 4: Describe cleaned data
    print("\n" + "=" * 60)
    print("STEP 4: DESCRIBING CLEANED DATA")
    print("=" * 60)
    
    preprocessor.describe_clean_data()
    
    # Step 5: Save cleaned data (optional)
    # preprocessor.save_clean_data("dataset/training_data_clean.csv")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    # ------------------------
    # TRAINING
    # ------------------------
    df_clean = preprocessor.get_clean_data()
    
    # Check if model already exists
    model_exists = MODEL_PATH.exists()
    
    if model_exists and not args.resume:
        print(f"\nModel already exists at {MODEL_PATH}")
        print("Skipping training. Delete the model file to retrain.")
        print("Or use --resume <checkpoint_path> to continue training.")
    else:
        print("\n" + "=" * 60)
        if args.resume:
            print("STEP 5: RESUMING MODEL TRAINING")
            print(f"Checkpoint: {args.resume}")
        else:
            print("STEP 5: TRAINING MODEL")
        print("=" * 60)
        
        tft_model = TFTStockModel(df_clean)
        tft_model.create_datasets()
        tft_model.build_model()
        tft_model.train(max_epochs=args.max_epochs, ckpt_path=args.resume)
        tft_model.save_model(str(MODEL_PATH))
    
    # ------------------------
    # TESTING
    # ------------------------
    print("\n" + "=" * 60)
    print("STEP 6: TESTING MODEL")
    print("=" * 60)
    
    # Check if test data exists
    if not TEST_DATA_PATH.exists():
        print(f"\nTest data not found at {TEST_DATA_PATH}")
        print("Run fetch_test_data.py first to generate test data.")
    else:
        # Load and preprocess test data
        print("\nLoading test data...")
        test_preprocessor = StockDataPreprocessor(str(TEST_DATA_PATH))
        test_preprocessor.load_data()
        test_preprocessor.clean_data(
            fill_na_columns={"vwap": 0, "pe_ratio": 0, "book_value": 0, "dividend_yield": 0},
            min_records_per_ticker=100
        )
        test_df = test_preprocessor.get_clean_data()
        
        # Create model and load weights
        print("\nLoading trained model...")
        tft_model = TFTStockModel(df_clean)
        tft_model.create_datasets()
        tft_model.load_model(str(MODEL_PATH))
        
        # Evaluate on test data
        results = tft_model.evaluate(test_df)
        
        # Make predictions
        predictions_df = tft_model.predict(test_df)
        predictions_path = BASE_DIR / "output" / "test_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"\nPredictions saved to {predictions_path}")
        
        # Print prediction summary
        print("\n" + "=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)
        print(f"\nTotal predictions: {len(predictions_df):,}")
        print(f"\nPrediction statistics:")
        print(predictions_df.describe().round(4).to_string())
        print(f"\nFirst 20 predictions:")
        print(predictions_df.head(20).to_string())
        
        # Print accuracy summary
        print("\n" + "=" * 60)
        print("ACCURACY SUMMARY")
        print("=" * 60)
        print(f"\nDirection Accuracy: {results['direction_accuracy']:.2f}%")
        print(f"  (Correctly predicted up/down direction)")
        print(f"\nIncorrect: {100 - results['direction_accuracy']:.2f}%")
        print(f"\nOther Metrics:")
        print(f"  MAE:  {results['mae']:.6f}")
        print(f"  RMSE: {results['rmse']:.6f}")
        print(f"  R²:   {results['r2']:.4f}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)