# AlphaMind Model Training Service

## Overview

The Model Training Service trains a **Temporal Fusion Transformer (TFT)** model for stock price prediction. The model learns from historical price data, technical indicators, and fundamental ratios to predict future returns.

## Architecture

### Model: Temporal Fusion Transformer (TFT)

| Component | Value |
|-----------|-------|
| Type | Temporal Fusion Transformer |
| Framework | PyTorch Forecasting + Lightning |
| Parameters | 118,192 |
| Encoder Length | 30 timesteps (~6 trading weeks) |
| Prediction Horizon | 5 timesteps |
| Target | `target_return` (future price return) |

### Key Design Decision: No Ticker Embeddings

The model uses **global normalization** and **no ticker-specific embeddings**, allowing it to generalize to **any stock** - including stocks not seen during training.

**What this means:**
- ✅ Can predict on completely new/unseen tickers
- ✅ Learns universal price patterns (RSI, MACD, etc.)
- ✅ No overfitting to specific stock behaviors
- ❌ Cannot capture ticker-specific anomalies

### Features Used

| Category | Features |
|----------|----------|
| OHLCV | open, high, low, close, volume |
| Technical Indicators | rsi, sma_50, sma_200, macd, macd_signal, macd_histogram, vwap |
| Fundamental Ratios | pe_ratio, book_value, dividend_yield, roce, roe, eps, debt_to_equity, face_value, market_cap |
| Derived Features | daily_return, price_to_sma50, price_to_sma200, volatility |
| Categorical | sma_crossover, rsi_signal |

---

## Training Results

### Latest Training Run (March 2026)

| Metric | Value |
|--------|-------|
| Training Epochs | 14 (early stopped at epoch 13) |
| Best Validation Loss | 0.927 (epoch 3) |
| Final Training Loss | 0.809 |
| Training Samples | 168,157 |
| Validation Samples | 42,026 |
| Training Tickers | 98 |

### Training Progress

```
Epoch 0:  val_loss=0.945, train_loss=0.822 ✓ (best)
Epoch 1:  val_loss=0.928, train_loss=0.815 ✓ (improved)
Epoch 3:  val_loss=0.927, train_loss=0.814 ✓ (best)
Epoch 4-13: val_loss did not improve
Early stopping triggered at epoch 13
```

---

## Evaluation Results

### Test Set Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Test Samples | 420,750 | ~84K windows × 5 horizons |
| Test Tickers | 23 | All unseen during training |
| **Direction Accuracy** | **50.77%** | Slightly better than random (50%) |
| MAE | 1.527 | Mean absolute error |
| RMSE | 2.246 | Root mean squared error |
| MAPE | 110.67% | Mean absolute percentage error |
| R² | -0.0012 | Near zero (poor fit) |

### Prediction Statistics

```
       sample_idx      horizon   prediction
count  420750.0000  420750.0000  420750.0000
mean    42074.5000       3.0000       0.0179
std     24292.0414       1.4142       0.2507
min         0.0000       1.0000      -7.0411
25%     21037.0000       2.0000      -0.0277
50%     42074.5000       3.0000       0.0227
75%     63112.0000       4.0000       0.0640
max     84149.0000       5.0000       5.9762
```

### Interpretation

- **Direction Accuracy ~50.77%**: Model is slightly better than random at predicting up/down direction
- **Negative R² (-0.0012)**: Model predictions cluster near zero and don't capture the true magnitude of returns. Even though direction is ~50% correct, the predicted *values* deviate from actuals more than simply predicting the mean would.
- **High MAPE**: Percentage errors are large due to magnitude mismatch

**Why R² is negative with 50% direction accuracy:**
Direction accuracy only checks if the sign (+/-) is correct. R² checks if the actual values match. The model predicts small values (mean ~0.02, std ~0.25) while actual returns have larger variance. The model is "playing it safe" near zero.

---

## Usage

### Training a New Model

```bash
cd /Users/aadityapal/Documents/AlphaMind/model_training

# Clean old artifacts
rm -f models/tft_stock_model.pt
rm -rf model_train/lightning_logs

# Train
python model_train/train_tft.py --max-epochs 40
```

### Resume Training from Checkpoint

```bash
python model_train/train_tft.py \
  --resume model_train/lightning_logs/version_X/checkpoints/epoch=Y-step=Z.ckpt \
  --max-epochs 50
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-epochs` | 20 | Maximum training epochs |
| `--resume` | None | Path to checkpoint to resume from |

---

## File Structure

```
model_training/
├── model_train/
│   ├── train_tft.py          # Main training script
│   ├── quick_test.py         # Quick evaluation script
│   └── lightning_logs/       # Training checkpoints & logs
│       └── version_X/
│           ├── checkpoints/
│           ├── hparams.yaml
│           └── metrics.csv
├── output/
│   ├── training_data.csv     # Preprocessed training data
│   ├── testing_data.csv      # Preprocessed test data
│   └── test_predictions.csv  # Model predictions
├── models/
│   └── tft_stock_model.pt    # Saved model weights
├── training_stocks.txt       # List of training tickers
├── testing_stocks.txt        # List of test tickers
├── fetch_test_data.py        # Script to fetch test data
└── requirements.txt          # Python dependencies
```

---

## Configuration

### Model Hyperparameters

Located in `train_tft.py`:

```python
# TFTStockModel defaults
max_encoder_length = 30      # Days of history used
max_prediction_length = 5    # Days to predict ahead

# Model architecture
learning_rate = 0.005
hidden_size = 32
attention_head_size = 4
dropout = 0.1

# Training
batch_size = 64
num_workers = 7
patience = 10  # Early stopping patience
```

### Dataset Configuration

```python
# Target normalization (global, not per-ticker)
target_normalizer = TorchNormalizer(method="robust", center=True)

# No ticker embeddings - model generalizes to any stock
static_categoricals = []  # Empty - no ticker identity used
```

---

## Dependencies

```
pytorch-forecasting>=1.0.0
lightning>=2.0.0
torch>=2.0.0
pandas
numpy
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Future Improvements

1. **More Training Data**: Add more stocks and longer history
2. **Feature Engineering**: Add sector embeddings, market regime indicators
3. **Hyperparameter Tuning**: Use Optuna for automated tuning
4. **Ensemble Methods**: Combine multiple models
5. **Alternative Targets**: Predict classification (up/down) instead of returns
6. **Walk-Forward Validation**: More realistic backtesting

---

## Notes

- Model is trained on **Indian NSE stocks** (tickers ending in `.NS`)
- All times are in IST (Indian Standard Time)
- Data is hourly candle data aggregated from intraday
- Model checkpoint saved at best validation loss, not final epoch

---

## Prediction API

The model service exposes a FastAPI server for real-time stock predictions.

### Starting the API Server

```bash
cd /Users/aadityapal/Documents/AlphaMind/model_service

# Make sure scraper_service is running on port 8000
# Then start the model service:
python -m api.server

# Or with uvicorn directly:
uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCRAPER_API_URL` | `http://localhost:8000` | URL of the scraper service |
| `PORT` | `8001` | Port to run the prediction API |

### API Endpoints

#### Health Check
```
GET /health
```
Returns service health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/path/to/checkpoint.ckpt",
  "scraper_api": "http://localhost:8000"
}
```

#### Predict Stock (POST)
```
POST /predict
Content-Type: application/json

{
  "stock": "Reliance"
}
```

#### Predict Stock (GET)
```
GET /predict/{stock}
```

**Example:**
```bash
curl http://localhost:8001/predict/RELIANCE
```

**Response:**
```json
{
  "ticker": "RELIANCE.NS",
  "company": "Reliance",
  "current_price": 2450.50,
  "predictions": [
    {"horizon": 1, "predicted_return": 0.15, "direction": "UP"},
    {"horizon": 2, "predicted_return": 0.08, "direction": "UP"},
    {"horizon": 3, "predicted_return": -0.02, "direction": "DOWN"},
    {"horizon": 4, "predicted_return": 0.12, "direction": "UP"},
    {"horizon": 5, "predicted_return": 0.05, "direction": "UP"}
  ],
  "model_info": {
    "encoder_length": 30,
    "prediction_length": 5,
    "model_path": "/path/to/checkpoint.ckpt",
    "data_points_used": 31
  },
  "timestamp": "2026-03-09T10:30:00Z"
}
```

#### Reload Model
```
POST /reload-model
```
Reloads the model from disk (useful after retraining).

### How It Works

1. **Request received** with stock name (e.g., "Reliance")
2. **Calls scraper_service** API to fetch current market data and fundamentals
3. **Preprocesses data** - calculates technical indicators, normalizes features
4. **Runs TFT model** inference on the last 30 days of data
5. **Returns predictions** for the next 5 time periods with direction

### Error Handling

| Status Code | Meaning |
|-------------|---------|
| 200 | Success |
| 400 | Invalid request (missing stock name, insufficient data) |
| 502 | Scraper service error |
| 503 | Model not loaded or scraper unreachable |
| 500 | Internal prediction error |

### API Documentation

Interactive Swagger docs available at:
- `http://localhost:8001/docs` (Swagger UI)
- `http://localhost:8001/redoc` (ReDoc)
