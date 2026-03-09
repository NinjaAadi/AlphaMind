from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# --- Clean API Response Models ---


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators computed from historical prices."""
    rsi: Optional[float] = None  # Relative Strength Index (0-100)
    macd: Optional[float] = None  # MACD line value
    macd_signal: Optional[float] = None  # MACD signal line
    macd_histogram: Optional[float] = None  # MACD histogram (MACD - Signal)
    sma_50: Optional[float] = None  # 50-day Simple Moving Average
    sma_200: Optional[float] = None  # 200-day Simple Moving Average
    vwap: Optional[float] = None  # Volume Weighted Average Price
    # Additional signals
    sma_crossover: Optional[str] = None  # "bullish" (50 > 200), "bearish" (50 < 200), or None
    rsi_signal: Optional[str] = None  # "oversold" (< 30), "overbought" (> 70), or "neutral"


@dataclass
class Market:
    """Market data from yfinance - real-time trading information."""
    price: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    technical_indicators: Optional[TechnicalIndicators] = None
    historical_prices: List['HistoricalPrice'] = field(default_factory=list)


@dataclass
class Ratios:
    """Key financial ratios from Screener.in."""
    pe: Optional[float] = None  # Stock P/E
    book_value: Optional[float] = None
    dividend_yield: Optional[float] = None  # percentage
    roce: Optional[float] = None  # Return on Capital Employed %
    roe: Optional[float] = None  # Return on Equity %
    face_value: Optional[float] = None
    eps: Optional[float] = None
    debt_to_equity: Optional[float] = None
    high_low: Optional[str] = None  # e.g., "1,612/1,115"


@dataclass
class Financials:
    """Financial statements from Screener.in - all values converted to numbers."""
    quarterly_results: List[Dict[str, Any]] = field(default_factory=list)
    profit_loss: List[Dict[str, Any]] = field(default_factory=list)
    balance_sheet: List[Dict[str, Any]] = field(default_factory=list)
    cash_flow: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StockSummary:
    """Clean API response structure optimized for RAG/ML/trading models."""
    ticker: str
    company: str
    market: Market
    ratios: Ratios
    financials: Financials
    shareholding: List[Dict[str, Any]] = field(default_factory=list)
    peer_comparison: List[Dict[str, Any]] = field(default_factory=list)
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    news: List["NewsArticle"] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --- Supporting Models ---


@dataclass
class MarketData:
    """Legacy model - kept for backwards compatibility with yfinance provider."""
    ticker: str
    symbol: Optional[str] = None
    open_price: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[int] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    forward_eps: Optional[float] = None
    debt_to_equity: Optional[float] = None
    book_value: Optional[float] = None
    historical_prices: List['HistoricalPrice'] = field(default_factory=list)
    technical_indicators: Optional['TechnicalIndicators'] = None


@dataclass
class Fundamentals:
    """Legacy model - kept for backwards compatibility with screener provider."""
    symbol: Optional[str] = None
    market_cap: Optional[float] = None
    current_price: Optional[float] = None
    high_low: Optional[str] = None
    pe_ratio: Optional[float] = None
    book_value: Optional[float] = None
    dividend_yield: Optional[float] = None
    roce: Optional[float] = None
    roe: Optional[float] = None
    face_value: Optional[float] = None
    eps: Optional[float] = None
    debt_equity: Optional[float] = None
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    quarterly_results: List[Dict[str, Any]] = field(default_factory=list)
    profit_loss: List[Dict[str, Any]] = field(default_factory=list)
    balance_sheet: List[Dict[str, Any]] = field(default_factory=list)
    cash_flow: List[Dict[str, Any]] = field(default_factory=list)
    ratios: List[Dict[str, Any]] = field(default_factory=list)
    shareholding: List[Dict[str, Any]] = field(default_factory=list)
    peer_comparison: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class HistoricalPrice:
    """Historical price data for a stock (OHLCV) with technical indicators."""
    date: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float = 0.0
    volume: int = 0
    # Technical indicators per day (for charting)
    rsi: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None


@dataclass
class NewsArticle:
    title: str
    url: str
    description: Optional[str] = None
    source: Optional[str] = None
    published_at: Optional[str] = None


@dataclass
class CompanyFundamentals:
    """Container for all data scraped from a Screener.in company page."""
    company_info: Dict[str, Any] = field(default_factory=dict)
    fundamentals: Dict[str, Any] = field(default_factory=dict)
    pros_cons: Dict[str, List[str]] = field(
        default_factory=lambda: {"pros": [], "cons": []}
    )
    peer_comparison: List[Dict[str, Any]] = field(default_factory=list)
    quarterly_results: List[Dict[str, Any]] = field(default_factory=list)
    profit_loss: List[Dict[str, Any]] = field(default_factory=list)
    balance_sheet: List[Dict[str, Any]] = field(default_factory=list)
    cash_flow: List[Dict[str, Any]] = field(default_factory=list)
    ratios: List[Dict[str, Any]] = field(default_factory=list)
    shareholding: List[Dict[str, Any]] = field(default_factory=list)
    

@dataclass
class ErrorResponse:
    """Error response model."""
    error: str
    message: str