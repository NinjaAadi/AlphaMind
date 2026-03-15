# Scraper Service (AlphaMind)

**Port: 8000** — Stock data aggregation and **main UI** for AlphaMind.

Part of [AlphaMind](../README.md): this service aggregates Indian stock market data from YFinance, Screener.in, and NewsAPI; serves the dashboard and "Ask the LLM" chat; and proxies LLM requests to the RAG service (8002). The **model service** (8001) and **RAG service** (8002) depend on this API for live stock data.

---

## Architecture

```
scraper_service/
├── providers/                 # Data collection modules
│   ├── yfinance_provider.py   # Yahoo Finance market data
│   ├── news_provider.py       # NewsAPI integration
│   └── screener_provider.py   # Screener.in scraping
│
├── models/                    # Pydantic data models
│   └── stock_models.py
│
├── services/                  # Business logic
│   └── aggregator_service.py
│
├── api/                       # FastAPI endpoints
│   └── server.py
│
├── utils/                     # Helper utilities
│   └── http_utils.py
│
└── requirements.txt
```

## Data Sources

1. **YFinance** - Real-time market data, historical prices, market cap, PE ratio
2. **NewsAPI** - Stock-related news articles and updates
3. **Screener.in** - Company fundamentals, ratios (PE, ROCE, Debt, Growth rates)

## Features

- ✅ Clean architecture with modular providers
- ✅ Type hints throughout codebase
- ✅ Pydantic validation for all data models
- ✅ Comprehensive error handling
- ✅ Retry logic for HTTP requests
- ✅ Logging for debugging
- ✅ CORS support
- ✅ Batch API endpoint
- ✅ Production-ready code

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone or create the project directory:**

   ```bash
   cd /path/to/scraper_service
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables:**

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and add your NewsAPI key:
   - Get a free API key from [https://newsapi.org/](https://newsapi.org/)

   ```env
   NEWS_API_KEY=your_actual_api_key
   ```

## Running the Service

### Development

```bash
cd scraper_service
source venv/bin/activate   # or venv\Scripts\activate on Windows
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

- **API:** http://localhost:8000  
- **UI (dashboard + Ask the LLM):** http://localhost:8000  
- **Docs:** http://localhost:8000/docs  

### Production

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### 1. Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "service": "stock-data-aggregator"
}
```

### 2. Get Single Stock Data

```
GET /stock?company=Reliance
GET /stock?company=TCS&ticker=TCS.NS
```

Parameters:
- `company`: **Required.** Company name (e.g., Reliance, TCS, Infosys).
- `ticker`: Optional. NSE ticker (e.g. RELIANCE.NS). If omitted, auto-generated as `{COMPANY}.NS`.

Example Response:
```json
{
  "ticker": "RELIANCE.NS",
  "company": "Reliance",
  "market_data": {
    "ticker": "RELIANCE.NS",
    "current_price": 2850.50,
    "market_cap": 1920000000000,
    "pe_ratio": 24.5,
    "historical_prices": [
      {
        "date": "2026-02-06",
        "close": 2820.25,
        "volume": 45320000
      },
      ...
    ]
  },
  "fundamentals": {
    "pe": 24.5,
    "roce": 12.8,
    "debt": 0.15,
    "sales_growth": 8.5,
    "profit_growth": 6.2
  },
  "news": [
    {
      "title": "Reliance Industries reports strong Q4 earnings",
      "description": "Company beats market expectations...",
      "source": "Economic Times",
      "url": "https://...",
      "published_at": "2026-03-08T10:30:00Z"
    },
    ...
  ],
  "timestamp": "2026-03-08T12:45:30.123456"
}
```

### 3. Get Multiple Stocks (Batch)

```
GET /stocks/batch?tickers=RELIANCE.NS,INFY.NS&companies=Reliance,Infosys
```

Parameters:
- `tickers`: Comma-separated NSE ticker symbols
- `companies`: Comma-separated company names

Returns:
```json
[
  { /* StockSummary for RELIANCE */ },
  { /* StockSummary for INFY */ }
]
```

### 4. API Documentation

```
GET /docs         # Swagger UI
GET /redoc        # ReDoc
```

## NSE Ticker Symbols

Common Indian stocks:
- `RELIANCE.NS` - Reliance Industries
- `INFY.NS` - Infosys
- `TCS.NS` - Tata Consultancy Services
- `HDFC.NS` - HDFC Bank
- `ICICIBANK.NS` - ICICI Bank
- `MARUTI.NS` - Maruti Suzuki
- `WIPRO.NS` - Wipro
- `HCLTECH.NS` - HCL Technologies
- `SUNPHARMA.NS` - Sun Pharmaceutical
- `ASIANPAINT.NS` - Asian Paints

## Code Structure

### Providers

Each provider is an independent module that handles data collection from a specific source:

**YFinanceProvider:**
- Fetches real-time market data, historical prices
- Handles timeout and error cases gracefully

**NewsProvider:**
- Integrates with NewsAPI for stock-related news
- Returns normalized article data
- Gracefully handles missing API key

**ScreenerProvider:**
- Scrapes company pages from Screener.in
- Extracts key financial ratios
- Uses BeautifulSoup for HTML parsing
- Includes fallback parsing strategies

### Models

All data models use Pydantic for:
- Type validation
- JSON serialization
- Automatic documentation

### Services

**AggregatorService:**
- Orchestrates data collection from all providers
- Combines results into unified response
- Handles errors from individual providers
- Continues on partial failures

### API Layer

**FastAPI Server:**
- Clean REST API endpoints
- Automatic API documentation
- Request validation
- Global exception handling
- CORS support

## Error Handling

The service implements comprehensive error handling:

1. **Provider-level errors** - Returns partial data when available
2. **Request errors** - Retry logic with exponential backoff
3. **Parsing errors** - Graceful degradation
4. **API errors** - HTTP 400/500 responses with details
5. **Validation errors** - Pydantic validation errors

## Logging

All modules include detailed logging:

```python
logger.info(f"Fetching yfinance data for ticker: {ticker}")
logger.error(f"Error fetching data: {str(e)}")
logger.warning(f"Missing API key")
```

View logs to debug issues.

## Performance Considerations

1. **Parallel requests** - Screener scraping is fastest when combined
2. **Caching** - Consider adding Redis for frequent queries
3. **Rate limiting** - NewsAPI has rate limits (check documentation)
4. **Timeouts** - All requests have configurable timeouts
5. **Retries** - Automatic retry on transient failures

## Future Enhancements

- [ ] Add Redis caching for 5-minute TTL
- [ ] Implement WebSocket support for real-time updates
- [ ] Add database storage (PostgreSQL)
- [ ] Add authentication/authorization
- [ ] Add request rate limiting
- [ ] Add metrics and monitoring (Prometheus)
- [ ] Add support for more data sources
- [ ] Add historical data analysis endpoints

## Troubleshooting

### Port Already in Use

```bash
# Change port in command
uvicorn api.server:app --port 8001
```

### NewsAPI Key Issues

- Verify API key is correct
- Check API plan has sufficient quota
- Get key from https://newsapi.org/

### Screener.in Scraping Fails

- Website structure may have changed
- Check HTML parsing logic in `screener_provider.py`
- Verify User-Agent headers are correct

### yfinance Connection Issues

- Check internet connection
- Verify ticker symbol format (e.g., RELIANCE.NS not just RELIANCE)
- Try yfinance directly: `python -c "import yfinance; print(yfinance.Ticker('RELIANCE.NS').info)"`

## Production Deployment

### Using Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY scraper_service .

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.server:app
```

### Environment Variables

```env
NEWS_API_KEY=your_key
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false  # Set to false in production
```

## Testing

To test the API:

```python
import requests

response = requests.get(
    "http://localhost:8000/stock",
    params={
        "ticker": "RELIANCE.NS",
        "company": "Reliance"
    }
)

print(response.json())
```

Or use cURL:

```bash
curl "http://localhost:8000/stock?ticker=RELIANCE.NS&company=Reliance"
```

## License

MIT License - feel free to use and modify

## Contributing

Contributions welcome! Please ensure:
- Type hints are used
- Tests are added
- Code follows PEP 8
- Docstrings are provided
