"""
Constants for the scraper service.
"""

# =============================================================================
# API URLs
# =============================================================================
SCREENER_BASE_URL = "https://www.screener.in"
SCREENER_COMPANY_URL = f"{SCREENER_BASE_URL}/company"
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

# =============================================================================
# HTTP Configuration
# =============================================================================
DEFAULT_TIMEOUT = 10
RETRY_COUNT = 3
RETRY_BACKOFF_FACTOR = 1
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# HTTP Headers
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)
ACCEPT_HEADER = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
ACCEPT_LANGUAGE = "en-US,en;q=0.5"

# =============================================================================
# Screener.in HTML Selectors
# =============================================================================
SCREENER_TOP_RATIOS_ID = "top-ratios"
SCREENER_PROS_CLASS = "pros"
SCREENER_CONS_CLASS = "cons"

# Section IDs for financial tables
SCREENER_SECTION_QUARTERS = "quarters"
SCREENER_SECTION_PROFIT_LOSS = "profit-loss"
SCREENER_SECTION_BALANCE_SHEET = "balance-sheet"
SCREENER_SECTION_CASH_FLOW = "cash-flow"
SCREENER_SECTION_RATIOS = "ratios"
SCREENER_SECTION_SHAREHOLDING = "shareholding"
SCREENER_SECTION_PEERS = "peers"

# =============================================================================
# YFinance Field Mappings
# =============================================================================
YFINANCE_CURRENT_PRICE_FIELDS = ["currentPrice", "regularMarketPrice"]
YFINANCE_OPEN_FIELDS = ["regularMarketOpen", "open"]
YFINANCE_HIGH_FIELDS = ["regularMarketDayHigh", "dayHigh"]
YFINANCE_LOW_FIELDS = ["regularMarketDayLow", "dayLow"]
YFINANCE_VOLUME_FIELDS = ["regularMarketVolume", "volume"]

# =============================================================================
# Date Formats
# =============================================================================
DATE_FORMAT = "%Y-%m-%d"
HISTORICAL_DAYS = 3650  # ~240 trading days - enough for SMA 200

# =============================================================================
# Technical Indicator Parameters
# =============================================================================
RSI_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
SMA_SHORT_PERIOD = 50
SMA_LONG_PERIOD = 200

# =============================================================================
# API Response Messages
# =============================================================================
ERROR_TICKER_REQUIRED = "Ticker and company name are required"
ERROR_INVALID_TICKER = "Invalid ticker format"
MSG_SERVICE_HEALTHY = "healthy"
SERVICE_NAME = "stock-data-aggregator"

# =============================================================================
# Logging Formats
# =============================================================================
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# Default Values
# =============================================================================
DEFAULT_NEWS_LIMIT = 10
UNKNOWN_SOURCE = "Unknown"
NOT_AVAILABLE_VALUES = ["N/A", "-", ""]

# Column name for metric/row labels in financial tables
METRIC_COLUMN_NAME = "metric"

# Characters to strip when parsing numbers
NUMBER_STRIP_CHARS = [",", "%", "Cr.", "cr", "₹", " "]
