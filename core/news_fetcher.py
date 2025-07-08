"""
Elite Financial News Fetcher
Multi-source news ingestion with advanced filtering and preprocessing
Supports NewsAPI, RSS feeds, and web scraping for maximum coverage
"""

import os
import requests
import feedparser
import json
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import logging
from urllib.parse import urljoin, urlparse
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Structured representation of a financial news article"""
    headline: str
    content: str
    source: str
    url: str
    published_at: datetime
    author: Optional[str] = None
    category: Optional[str] = None
    tickers: List[str] = None
    hash_id: str = None
    
    def __post_init__(self):
        """Generate unique hash ID for deduplication"""
        if not self.hash_id:
            content_hash = hashlib.md5(
                f"{self.headline}{self.source}".encode('utf-8')
            ).hexdigest()
            self.hash_id = content_hash

class FinancialNewsFetcher:
    """
    Advanced news fetcher with multiple sources and intelligent filtering
    """
    
    def __init__(self, api_key: str = None, config: Dict = None):
        self.api_key = api_key or os.getenv('NEWSAPI_KEY')
        self.config = config or self._default_config()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FinancialSentimentAI/1.0 (Professional NLP Research)'
        })
        self.seen_articles = set()  # For deduplication
        
    def _default_config(self) -> Dict:
        """Default configuration for news fetching"""
        return {
            'newsapi_url': 'https://newsapi.org/v2/everything',
            'financial_keywords': [
                'stocks', 'market', 'trading', 'investment', 'earnings', 'revenue',
                'profit', 'loss', 'economy', 'inflation', 'federal reserve', 'fed',
                'interest rates', 'nasdaq', 'dow jones', 's&p 500', 'bitcoin',
                'cryptocurrency', 'forex', 'commodities', 'oil prices', 'gold',
                'merger', 'acquisition', 'ipo', 'dividend', 'quarterly results',
                'financial', 'economic', 'fiscal', 'monetary', 'banking'
            ],
            'high_quality_sources': [
                'bloomberg.com', 'reuters.com', 'wsj.com', 'ft.com',
                'cnbc.com', 'marketwatch.com', 'yahoo.com', 'investing.com',
                'seekingalpha.com', 'fool.com', 'benzinga.com'
            ],
            'rss_feeds': [
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://moxie.foxbusiness.com/google-publisher/markets.xml',
                'https://www.marketwatch.com/rss/topstories',
                'https://finance.yahoo.com/news/rssindex'
            ],
            'exclusion_patterns': [
                r'\[.*removed.*\]',
                r'subscribe to continue',
                r'sign up.*free',
                r'advertisement'
            ],
            'min_headline_length': 10,
            'max_articles_per_source': 50,
            'request_timeout': 15,
            'rate_limit_delay': 0.5
        }
    
    def fetch_from_newsapi(self, 
                          query: str = None, 
                          days_back: int = 1,
                          max_articles: int = 100) -> List[NewsArticle]:
        """
        Fetch news from NewsAPI with advanced filtering
        """
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError("Valid NewsAPI key required for real-time analysis")
        
        # Build sophisticated query
        if not query:
            query = ' OR '.join(self.config['financial_keywords'][:10])
        
        # Calculate date range
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'language': 'en',
            'sortBy': 'publishedAt',
            'from': from_date,
            'to': to_date,
            'pageSize': min(max_articles, 100),
            'apiKey': self.api_key
        }
        
        try:
            logger.info(f"🔄 Fetching from NewsAPI: {query[:50]}...")
            response = self.session.get(
                self.config['newsapi_url'], 
                params=params,
                timeout=self.config['request_timeout']
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'ok':
                raise Exception(f"NewsAPI Error: {data.get('message', 'Unknown error')}")
            
            articles = []
            for article_data in data.get('articles', []):
                article = self._parse_newsapi_article(article_data)
                if article and self._is_high_quality_article(article):
                    articles.append(article)
            
            logger.info(f"✅ NewsAPI: {len(articles)} high-quality articles")
            return articles
            
        except Exception as e:
            logger.error(f"❌ NewsAPI fetch failed: {e}")
            return []
    
    def fetch_from_rss(self, max_articles: int = 50) -> List[NewsArticle]:
        """
        Fetch from multiple RSS feeds with intelligent parsing
        """
        all_articles = []
        
        for feed_url in self.config['rss_feeds']:
            try:
                logger.info(f"🔄 Fetching RSS: {urlparse(feed_url).netloc}")
                
                # Add delay to respect rate limits
                time.sleep(self.config['rate_limit_delay'])
                
                feed = feedparser.parse(feed_url)
                
                if not feed.entries:
                    logger.warning(f"⚠️ No entries found in {feed_url}")
                    continue
                
                feed_articles = []
                for entry in feed.entries[:max_articles]:
                    article = self._parse_rss_entry(entry, feed_url)
                    if article and self._is_high_quality_article(article):
                        feed_articles.append(article)
                
                all_articles.extend(feed_articles)
                logger.info(f"✅ RSS {urlparse(feed_url).netloc}: {len(feed_articles)} articles")
                
            except Exception as e:
                logger.error(f"❌ RSS fetch failed for {feed_url}: {e}")
                continue
        
        return all_articles
    
    def _parse_newsapi_article(self, article_data: Dict) -> Optional[NewsArticle]:
        """Parse NewsAPI article format"""
        try:
            headline = article_data.get('title', '').strip()
            content = article_data.get('description', '') or article_data.get('content', '')
            
            if not headline or headline == "[Removed]":
                return None
            
            # Parse published date
            published_str = article_data.get('publishedAt', '')
            try:
                published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            except:
                published_at = datetime.now()
            
            return NewsArticle(
                headline=headline,
                content=content.strip() if content else "",
                source=article_data.get('source', {}).get('name', 'Unknown'),
                url=article_data.get('url', ''),
                published_at=published_at,
                author=article_data.get('author')
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse NewsAPI article: {e}")
            return None
    
    def _parse_rss_entry(self, entry, feed_url: str) -> Optional[NewsArticle]:
        """Parse RSS feed entry"""
        try:
            headline = entry.get('title', '').strip()
            content = entry.get('summary', '') or entry.get('description', '')
            
            if not headline:
                return None
            
            # Parse published date
            published_at = datetime.now()
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published_at = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                published_at = datetime(*entry.updated_parsed[:6])
            
            # Extract source from feed URL
            source = urlparse(feed_url).netloc.replace('www.', '').replace('feeds.', '')
            
            return NewsArticle(
                headline=headline,
                content=content.strip() if content else "",
                source=source,
                url=entry.get('link', ''),
                published_at=published_at,
                author=entry.get('author')
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse RSS entry: {e}")
            return None
    
    def _is_high_quality_article(self, article: NewsArticle) -> bool:
        """
        Advanced quality filtering for financial relevance
        """
        headline = article.headline.lower()
        
        # Length check
        if len(headline) < self.config['min_headline_length']:
            return False
        
        # Exclusion patterns
        for pattern in self.config['exclusion_patterns']:
            if re.search(pattern, headline, re.IGNORECASE):
                return False
        
        # Deduplication
        if article.hash_id in self.seen_articles:
            return False
        
        # Financial relevance check
        financial_score = self._calculate_financial_relevance(headline)
        if financial_score < 0.3:  # Threshold for financial relevance
            return False
        
        # Source quality check
        if article.source.lower() in ['[removed]', 'unknown', '']:
            return False
        
        self.seen_articles.add(article.hash_id)
        return True
    
    def _calculate_financial_relevance(self, text: str) -> float:
        """
        Calculate financial relevance score using keyword matching
        """
        text_lower = text.lower()
        keyword_matches = 0
        
        # Primary financial keywords (higher weight)
        primary_keywords = [
            'stock', 'market', 'trading', 'earnings', 'revenue', 'profit',
            'economy', 'inflation', 'fed', 'interest rate', 'investment'
        ]
        
        # Secondary keywords (lower weight)
        secondary_keywords = [
            'financial', 'economic', 'fiscal', 'monetary', 'banking',
            'nasdaq', 'dow', 's&p', 'bitcoin', 'crypto', 'forex'
        ]
        
        # Count primary keyword matches (weight: 1.0)
        for keyword in primary_keywords:
            if keyword in text_lower:
                keyword_matches += 1.0
        
        # Count secondary keyword matches (weight: 0.5)
        for keyword in secondary_keywords:
            if keyword in text_lower:
                keyword_matches += 0.5
        
        # Normalize score
        max_possible_score = len(primary_keywords) + len(secondary_keywords) * 0.5
        return min(keyword_matches / max_possible_score * 2, 1.0)  # Scale to 0-1
    
    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from text using pattern matching
        """
        # Pattern for stock tickers (e.g., AAPL, TSLA, BTC-USD)
        ticker_pattern = r'\b[A-Z]{1,5}(?:-[A-Z]{1,3})?\b'
        
        # Common false positives to exclude
        exclude_list = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL',
            'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET',
            'USE', 'MAN', 'NEW', 'NOW', 'WAY', 'MAY', 'SAY', 'CEO',
            'CFO', 'CTO', 'IPO', 'SEC', 'FDA', 'GDP', 'CPI', 'API'
        }
        
        tickers = re.findall(ticker_pattern, text.upper())
        # Filter out common false positives and short words
        filtered_tickers = [
            ticker for ticker in tickers 
            if ticker not in exclude_list and len(ticker) >= 2
        ]
        
        return list(set(filtered_tickers))  # Remove duplicates
    
    def fetch_comprehensive(self, 
                          max_articles: int = 100,
                          include_rss: bool = True,
                          custom_query: str = None) -> List[NewsArticle]:
        """
        Comprehensive news fetching from all sources with deduplication
        """
        all_articles = []
        
        # Fetch from NewsAPI
        try:
            newsapi_articles = self.fetch_from_newsapi(
                query=custom_query,
                max_articles=max_articles // 2
            )
            all_articles.extend(newsapi_articles)
        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
        
        # Fetch from RSS feeds
        if include_rss:
            try:
                rss_articles = self.fetch_from_rss(max_articles // 2)
                all_articles.extend(rss_articles)
            except Exception as e:
                logger.error(f"RSS fetch failed: {e}")
        
        # Add ticker extraction
        for article in all_articles:
            article.tickers = self.extract_tickers(article.headline + " " + article.content)
        
        # Sort by published date (newest first)
        all_articles.sort(key=lambda x: x.published_at, reverse=True)
        
        # Limit to max_articles
        final_articles = all_articles[:max_articles]
        
        logger.info(f"🎯 Total high-quality articles: {len(final_articles)}")
        return final_articles
    
    def get_statistics(self) -> Dict:
        """Get fetcher statistics"""
        return {
            'unique_articles_seen': len(self.seen_articles),
            'configured_sources': len(self.config['high_quality_sources']),
            'rss_feeds': len(self.config['rss_feeds']),
            'financial_keywords': len(self.config['financial_keywords'])
        }

def fetch_financial_news(keywords: List[str] = None, max_articles: int = 20, api_key: str = None) -> List[NewsArticle]:
    """
    Convenience function to fetch financial news
    
    Args:
        keywords: List of keywords to search for
        max_articles: Maximum number of articles to fetch
        api_key: Optional API key (uses environment variable if not provided)
        
    Returns:
        List of news articles
    """
    fetcher = FinancialNewsFetcher(api_key=api_key)
    return fetcher.fetch_comprehensive(max_articles=max_articles)

def test_news_fetcher():
    """Test the financial news fetcher"""
    import os
    
    # Get API key from environment or config
    api_key = os.getenv('NEWS_API_KEY', 'bbfea5df5aee4175a9f5c08d04f99d1e')
    
    if not api_key or api_key == 'your_api_key_here':
        print("❌ Please set NEWS_API_KEY environment variable")
        return
    
    fetcher = FinancialNewsFetcher(api_key)
    
    print("🧪 Testing Elite Financial News Fetcher")
    print("=" * 60)
    
    # Test comprehensive fetch
    articles = fetcher.fetch_comprehensive(max_articles=10)
    
    print(f"\n📊 Results Summary:")
    print(f"   Total Articles: {len(articles)}")
    
    if articles:
        print(f"\n📰 Sample Articles:")
        for i, article in enumerate(articles[:5], 1):
            print(f"\n{i}. 📈 {article.headline}")
            print(f"   🏢 Source: {article.source}")
            print(f"   🕐 Published: {article.published_at.strftime('%Y-%m-%d %H:%M')}")
            if article.tickers:
                print(f"   💰 Tickers: {', '.join(article.tickers)}")
            print(f"   🔗 URL: {article.url[:80]}...")
    
    # Display statistics
    stats = fetcher.get_statistics()
    print(f"\n📊 Fetcher Statistics:")
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    test_news_fetcher()
