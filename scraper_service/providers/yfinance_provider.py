"""
YFinance provider for fetching market data from Yahoo Finance.
"""

from pathlib import Path
from typing import List, Optional
import logging
import certifi
import yfinance as yf
from curl_cffi.requests import Session as CurlCffiSession

# Use project-local cache to avoid permission errors (e.g. macOS Library/Caches)
_cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "py-yfinance"
_cache_dir.mkdir(parents=True, exist_ok=True)
try:
    yf.set_tz_cache_location(str(_cache_dir))
except Exception:
    pass  # Non-fatal if cache location cannot be set

from datetime import datetime, timedelta
from models.stock_models import MarketData, HistoricalPrice, TechnicalIndicators, NewsArticle
from utils.constants import (
    DATE_FORMAT, 
    HISTORICAL_DAYS, 
    DEFAULT_TIMEOUT,
    RSI_PERIOD,
    SMA_SHORT_PERIOD,
    SMA_LONG_PERIOD,
)
from utils.technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_sma,
    calculate_vwap,
    get_rsi_signal,
    get_sma_crossover_signal,
    calculate_rsi_series,
    calculate_sma_series,
    calculate_macd_series,
)


logger = logging.getLogger(__name__)


def _yf_session():
    """Build a curl_cffi Session for yfinance (Yahoo requires it). Uses certifi CA bundle for SSL verification."""
    return CurlCffiSession(impersonate="chrome", verify=certifi.where())


class YFinanceProvider:
    """Provider for fetching stock market data from yfinance."""
    
    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize YFinance provider.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
    
    def get_stock_data(self, ticker: str) -> MarketData:
        """
        Fetch stock data from yfinance.
        
        Args:
            ticker: Stock ticker symbol (e.g., RELIANCE.NS)
        
        Returns:
            MarketData: Market data object with current price, market cap, PE ratio, and historical prices
        
        Raises:
            Exception: If data fetching fails
        """
        try:
            logger.info(f"Fetching yfinance data for ticker: {ticker}")
            session = _yf_session()
            stock = yf.Ticker(ticker, session=session)
            
            # Get current data from info
            info = stock.info
            current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0.0)
            market_cap = info.get("marketCap")
            pe_ratio = info.get("trailingPE") or info.get("forwardPE")
            
            # Get open, high, low, volume from info
            open_price = info.get("regularMarketOpen") or info.get("open")
            day_high = info.get("regularMarketDayHigh") or info.get("dayHigh")
            day_low = info.get("regularMarketDayLow") or info.get("dayLow")
            volume = info.get("regularMarketVolume") or info.get("volume")
            
            # Get EPS and debt info
            trailing_eps = info.get("trailingEps")
            forward_eps = info.get("forwardEps")
            debt_to_equity = info.get("debtToEquity")
            book_value = info.get("bookValue")
            
            # Get historical prices (last HISTORICAL_DAYS days for SMA 200)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=HISTORICAL_DAYS)
            
            hist = stock.history(start=start_date, end=end_date)
            
            # Collect raw OHLCV data
            dates: List[str] = []
            opens: List[float] = []
            closes: List[float] = []
            highs: List[float] = []
            lows: List[float] = []
            volumes: List[int] = []
            
            for date, row in hist.iterrows():
                dates.append(date.strftime(DATE_FORMAT))
                opens.append(float(row["Open"]))
                closes.append(float(row["Close"]))
                highs.append(float(row["High"]))
                lows.append(float(row["Low"]))
                volumes.append(int(row["Volume"]))
            
            # Calculate technical indicator series for charting
            rsi_series = calculate_rsi_series(closes, RSI_PERIOD)
            sma_50_series = calculate_sma_series(closes, SMA_SHORT_PERIOD)
            sma_200_series = calculate_sma_series(closes, SMA_LONG_PERIOD)
            macd_series, macd_signal_series, macd_hist_series = calculate_macd_series(closes)
            
            # Build historical prices with technical indicators per day
            historical_prices: List[HistoricalPrice] = []
            for i in range(len(dates)):
                historical_prices.append(
                    HistoricalPrice(
                        date=dates[i],
                        open=opens[i],
                        high=highs[i],
                        low=lows[i],
                        close=closes[i],
                        volume=volumes[i],
                        rsi=rsi_series[i],
                        sma_50=sma_50_series[i],
                        sma_200=sma_200_series[i],
                        macd=macd_series[i],
                        macd_signal=macd_signal_series[i],
                        macd_histogram=macd_hist_series[i]
                    )
                )
            
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(
                closes, highs, lows, volumes
            )
            
            logger.info(f"Successfully fetched yfinance data for {ticker}")
            
            return MarketData(
                ticker=ticker,
                symbol=ticker,
                open_price=float(open_price) if open_price else None,
                high=float(day_high) if day_high else None,
                low=float(day_low) if day_low else None,
                volume=int(volume) if volume else None,
                current_price=float(current_price) if current_price else 0.0,
                market_cap=float(market_cap) if market_cap else None,
                pe_ratio=float(pe_ratio) if pe_ratio else None,
                eps=float(trailing_eps) if trailing_eps else None,
                forward_eps=float(forward_eps) if forward_eps else None,
                debt_to_equity=float(debt_to_equity) if debt_to_equity else None,
                book_value=float(book_value) if book_value else None,
                historical_prices=historical_prices,
                technical_indicators=technical_indicators
            )
        
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {ticker}: {str(e)}")
            # Return empty/default data on error
            return MarketData(
                ticker=ticker,
                symbol=ticker,
                current_price=0.0,
                market_cap=None,
                pe_ratio=None,
                historical_prices=[]
            )
    
    def _calculate_technical_indicators(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[int]
    ) -> Optional[TechnicalIndicators]:
        """
        Calculate technical indicators from historical price data.
        
        Args:
            closes: List of closing prices (oldest to newest)
            highs: List of high prices
            lows: List of low prices
            volumes: List of volumes
        
        Returns:
            TechnicalIndicators object with RSI, MACD, SMA, VWAP
        """
        if not closes or len(closes) < RSI_PERIOD:
            logger.warning("Insufficient data for technical indicators")
            return None
        
        try:
            # Calculate RSI
            rsi = calculate_rsi(closes, RSI_PERIOD)
            
            # Calculate MACD
            macd, macd_signal, macd_histogram = calculate_macd(closes)
            
            # Calculate SMAs
            sma_50 = calculate_sma(closes, SMA_SHORT_PERIOD)
            sma_200 = calculate_sma(closes, SMA_LONG_PERIOD)
            
            # Calculate VWAP
            vwap = calculate_vwap(closes, highs, lows, volumes)
            
            # Get signals
            rsi_signal = get_rsi_signal(rsi)
            sma_crossover = get_sma_crossover_signal(sma_50, sma_200)
            
            indicators = TechnicalIndicators(
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                sma_50=round(sma_50, 2) if sma_50 else None,
                sma_200=round(sma_200, 2) if sma_200 else None,
                vwap=vwap,
                rsi_signal=rsi_signal,
                sma_crossover=sma_crossover
            )
            
            logger.info(f"Calculated technicals: RSI={rsi}, MACD={macd}, SMA50={sma_50}, SMA200={sma_200}")
            return indicators
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {str(e)}")
            return None

    def get_news(self, ticker: str) -> List[NewsArticle]:
        """
        Fetch news articles from yfinance.
        
        Args:
            ticker: Stock ticker symbol (e.g., RELIANCE.NS)
        
        Returns:
            List of NewsArticle objects
        """
        try:
            logger.info(f"Fetching yfinance news for ticker: {ticker}")
            session = _yf_session()
            stock = yf.Ticker(ticker, session=session)
            news = stock.news or []
            
            articles = []
            for item in news:
                try:
                    content = item.get('content', {})
                    if not content:
                        continue
                        
                    title = content.get('title', '')
                    if not title:
                        continue
                    
                    # Get URL from clickThroughUrl or canonicalUrl
                    url_info = content.get('clickThroughUrl') or content.get('canonicalUrl', {})
                    url = url_info.get('url', '') if isinstance(url_info, dict) else ''
                    
                    # Get provider/source
                    provider = content.get('provider', {})
                    source = provider.get('displayName', 'Yahoo Finance') if isinstance(provider, dict) else 'Yahoo Finance'
                    
                    # Get published date
                    pub_date = content.get('pubDate', '')
                    
                    # Get description/summary
                    description = content.get('summary') or content.get('description', '')
                    
                    articles.append(NewsArticle(
                        title=title,
                        url=url,
                        description=description,
                        source=source,
                        published_at=pub_date
                    ))
                except Exception as e:
                    logger.debug(f"Error parsing news item: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} news articles from yfinance for {ticker}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching yfinance news for {ticker}: {str(e)}")
            return []
