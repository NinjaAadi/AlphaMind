"""
News provider for fetching stock-related news from NewsAPI.
"""

import os
import logging
import re
from typing import List, Optional
from utils.http_utils import make_request
from utils.constants import NEWS_API_BASE_URL, DEFAULT_TIMEOUT, DEFAULT_NEWS_LIMIT, UNKNOWN_SOURCE
from models.stock_models import NewsArticle


logger = logging.getLogger(__name__)


class NewsProvider:
    """Provider for fetching news from NewsAPI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize News provider.
        Args:
            api_key: NewsAPI API key. If not provided, will try to get from environment variable
        """
        self.api_key = api_key 
        self.base_url = NEWS_API_BASE_URL
        self.timeout = DEFAULT_TIMEOUT
    
    def _is_relevant(self, article: dict, company: str, ticker: str) -> bool:
        """
        Check if an article is relevant to the company/stock.
        
        Args:
            article: Raw article dict from NewsAPI
            company: Company name (e.g., "Reliance")
            ticker: Stock ticker (e.g., "RELIANCE.NS")
        
        Returns:
            True if article appears relevant, False otherwise
        """
        title = (article.get("title") or "").lower()
        description = (article.get("description") or "").lower()
        content = title + " " + description
        
        # Extract symbol from ticker (e.g., RELIANCE from RELIANCE.NS)
        symbol = ticker.split(".")[0].lower() if ticker else ""
        company_lower = company.lower() if company else ""
        
        # Check if company name or symbol appears in title or description
        if company_lower and company_lower in content:
            return True
        if symbol and symbol in content:
            return True
        
        # Also check for common financial keywords to catch related news
        financial_keywords = ["stock", "share", "market", "investor", "trading", "nse", "bse"]
        has_financial_context = any(kw in content for kw in financial_keywords)
        
        # If has financial context, be more lenient with partial matches
        if has_financial_context and company_lower:
            # Check for partial company name match (at least first word)
            first_word = company_lower.split()[0] if company_lower else ""
            if first_word and len(first_word) > 3 and first_word in content:
                return True
        
        return False
    
    def get_news(self, query: str, limit: int = DEFAULT_NEWS_LIMIT, ticker: str = "") -> List[NewsArticle]:
        """
        Fetch news articles related to a stock query with relevance filtering.
        
        Args:
            query: Search query (company name)
            limit: Maximum number of articles to return
            ticker: Stock ticker for additional relevance filtering
        
        Returns:
            List[NewsArticle]: List of relevant news articles
        """
        try:
            logger.info(f"Fetching news for query: {query}")
            
            if not self.api_key:
                logger.warning("NewsAPI key not configured. Returning empty news list.")
                return []
            
            # Fetch more articles than needed to account for filtering
            fetch_limit = min(limit * 3, 50)
            
            params = {
                "q": query,
                "sortBy": "publishedAt",
                "pageSize": fetch_limit,
                "apiKey": self.api_key
            }
            
            response = make_request(
                url=self.base_url,
                params=params,
                timeout=self.timeout
            )
            
            if response is None:
                logger.error(f"Failed to fetch news for {query}")
                return []
            
            data = response.json()
            articles = data.get("articles", [])
            
            news_list: List[NewsArticle] = []
            filtered_count = 0
            
            for article in articles:
                # Apply relevance filtering
                if not self._is_relevant(article, query, ticker):
                    filtered_count += 1
                    continue
                
                try:
                    news_article = NewsArticle(
                        title=article.get("title", ""),
                        description=article.get("description"),
                        source=article.get("source", {}).get("name", UNKNOWN_SOURCE),
                        url=article.get("url", ""),
                        published_at=article.get("publishedAt", "")
                    )
                    news_list.append(news_article)
                    
                    # Stop when we have enough relevant articles
                    if len(news_list) >= limit:
                        break
                except Exception as e:
                    logger.warning(f"Error parsing article: {str(e)}")
                    continue
            
            logger.info(f"Fetched {len(news_list)} relevant articles for {query} (filtered {filtered_count} irrelevant)")
            return news_list
        
        except Exception as e:
            logger.error(f"Error fetching news for {query}: {str(e)}")
            return []
