"""
Aggregator service for combining data from multiple providers.
"""

import logging
from typing import Optional, List
from providers.yfinance_provider import YFinanceProvider
from providers.news_provider import NewsProvider
from providers.screener_provider import ScreenerProvider
from models.stock_models import (
    StockSummary, 
    Market, 
    Ratios, 
    Financials,
    MarketData, 
    Fundamentals, 
    NewsArticle
)


logger = logging.getLogger(__name__)


class AggregatorService:
    """Service for aggregating stock data from multiple providers."""
    
    def __init__(self, news_api_key: Optional[str] = None):
        """
        Initialize aggregator service with providers.
        
        Args:
            news_api_key: Optional API key for NewsAPI
        """
        self.yfinance_provider = YFinanceProvider()
        self.news_provider = NewsProvider(api_key=news_api_key)
        self.screener_provider = ScreenerProvider()
    
    def get_stock_summary(self, ticker: str, company: str) -> StockSummary:
        """
        Get aggregated stock summary from all providers.
        
        Returns clean data structure optimized for RAG/ML/trading models:
        - market: Real-time trading data from yfinance
        - ratios: Key financial ratios from Screener.in
        - financials: Financial statements with numeric values
        - shareholding: Shareholding pattern
        - news: Filtered relevant news articles
        
        Args:
            ticker: Stock ticker (e.g., RELIANCE.NS) - can be empty to auto-generate
            company: Company name (e.g., Reliance)
        
        Returns:
            StockSummary: Aggregated stock information
        """
        try:
            # Strip whitespace from inputs
            company = company.strip() if company else ""
            ticker = ticker.strip() if ticker else ""
            
            # Validate company name is required
            if not company:
                raise ValueError("Company name is required")
            
            # Auto-generate ticker if not provided
            if not ticker:
                ticker = f"{company.upper().replace(' ', '')}.NS"
                logger.info(f"Auto-generated ticker: {ticker}")
            else:
                # Ensure ticker is uppercase and has proper suffix
                ticker = ticker.upper()
                if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
                    ticker = f"{ticker}.NS"
            
            logger.info(f"Getting stock summary for {ticker} ({company})")
            
            # Extract symbol from ticker (e.g., RELIANCE from RELIANCE.NS)
            symbol = ticker.split(".")[0]
            
            # Fetch data from all providers
            market_data = self._get_market_data(ticker)
            company_data = self._get_all_company_data(symbol)
            fundamentals = company_data.get("fundamentals", Fundamentals(symbol=symbol))
            
            # Build clean Market object from yfinance data
            market = Market(
                price=market_data.current_price,
                open=market_data.open_price,
                high=market_data.high,
                low=market_data.low,
                volume=market_data.volume,
                market_cap=market_data.market_cap,
                technical_indicators=market_data.technical_indicators,
                historical_prices=market_data.historical_prices
            )
            
            # Build clean Ratios object from Screener.in data
            # Prefer Screener data, fallback to yfinance where missing
            ratios = Ratios(
                pe=fundamentals.pe_ratio,
                book_value=fundamentals.book_value or market_data.book_value,
                dividend_yield=fundamentals.dividend_yield,
                roce=fundamentals.roce,
                roe=fundamentals.roe,
                face_value=fundamentals.face_value,
                eps=fundamentals.eps or market_data.eps,
                debt_to_equity=fundamentals.debt_equity or market_data.debt_to_equity,
                high_low=fundamentals.high_low
            )
            
            # Build clean Financials object with numeric values
            financials = Financials(
                quarterly_results=company_data.get("quarterly_results", []),
                profit_loss=company_data.get("profit_loss", []),
                balance_sheet=company_data.get("balance_sheet", []),
                cash_flow=company_data.get("cash_flow", [])
            )
            
            # Get filtered relevant news
            news = self._get_news(company, ticker)
            
            # Create aggregated response with clean structure
            summary = StockSummary(
                ticker=ticker,
                company=company,
                market=market,
                ratios=ratios,
                financials=financials,
                shareholding=company_data.get("shareholding", []),
                peer_comparison=company_data.get("peer_comparison", []),
                pros=fundamentals.pros,
                cons=fundamentals.cons,
                news=news,
            )
            
            logger.info(f"Successfully aggregated stock summary for {ticker}")
            return summary
        
        except Exception as e:
            logger.error(f"Error getting stock summary for {ticker}: {str(e)}")
            raise
    
    def _get_market_data(self, ticker: str) -> MarketData:
        """
        Get market data from yfinance provider.
        
        Args:
            ticker: Stock ticker
        
        Returns:
            MarketData: Market data object
        """
        try:
            return self.yfinance_provider.get_stock_data(ticker)
        except Exception as e:
            logger.warning(f"Failed to get market data for {ticker}: {str(e)}")
            return MarketData(ticker=ticker, current_price=0.0)
    
    def _get_fundamentals(self, symbol: str) -> Fundamentals:
        """
        Get fundamentals from screener provider.
        
        Args:
            symbol: Company symbol
        
        Returns:
            Fundamentals: Fundamental ratios
        """
        try:
            return self.screener_provider.get_fundamentals(symbol)
        except Exception as e:
            logger.warning(f"Failed to get fundamentals for {symbol}: {str(e)}")
            return Fundamentals()
    
    def _get_all_company_data(self, symbol: str) -> dict:
        """
        Get all company data from screener provider including financial tables.
        
        Args:
            symbol: Company symbol
        
        Returns:
            dict: Dictionary with fundamentals and all financial tables
        """
        try:
            return self.screener_provider.get_all_company_data(symbol)
        except Exception as e:
            logger.warning(f"Failed to get all company data for {symbol}: {str(e)}")
            return {"fundamentals": Fundamentals(symbol=symbol)}
    
    def _get_news(self, company: str, ticker: str = "") -> List[NewsArticle]:
        """
        Get news from both yfinance and NewsAPI, combined and deduplicated.
        
        Args:
            company: Company name
            ticker: Stock ticker for relevance filtering
        
        Returns:
            List[NewsArticle]: Combined list of relevant news articles
        """
        all_news = []
        seen_titles = set()
        
        # Get news from yfinance (free, no API key needed)
        try:
            yf_news = self.yfinance_provider.get_news(ticker)
            for article in yf_news:
                title_lower = article.title.lower()
                if title_lower not in seen_titles:
                    seen_titles.add(title_lower)
                    all_news.append(article)
        except Exception as e:
            logger.warning(f"Failed to get yfinance news for {ticker}: {str(e)}")
        
        # Get news from NewsAPI (if API key is configured)
        try:
            newsapi_news = self.news_provider.get_news(company, limit=10, ticker=ticker)
            for article in newsapi_news:
                title_lower = article.title.lower()
                if title_lower not in seen_titles:
                    seen_titles.add(title_lower)
                    all_news.append(article)
        except Exception as e:
            logger.warning(f"Failed to get NewsAPI news for {company}: {str(e)}")
        
        # Sort by published date (newest first)
        all_news.sort(key=lambda x: x.published_at or '', reverse=True)
        
        logger.info(f"Combined {len(all_news)} news articles for {company}")
        return all_news[:15]  # Return top 15 articles
