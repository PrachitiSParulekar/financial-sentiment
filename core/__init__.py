"""
Financial Sentiment Analysis AI - Core Module
Advanced NLP components for financial sentiment analysis
"""

__version__ = "1.0.0"
__author__ = "Financial Sentiment AI Team"

# Core components
from .news_fetcher import FinancialNewsFetcher, fetch_financial_news
from .text_cleaner import FinancialTextCleaner, clean_financial_text
from .sentiment_vader import FinancialVaderAnalyzer, analyze_vader_sentiment
from .sentiment_finbert_clean import CleanFinBERTAnalyzer as FinBERTAnalyzer, analyze_finbert

__all__ = [
    "FinancialNewsFetcher",
    "FinancialTextCleaner", 
    "FinancialVaderAnalyzer",
    "FinBERTAnalyzer",
    "fetch_financial_news",
    "clean_financial_text",
    "analyze_vader_sentiment",
    "analyze_finbert"
]
